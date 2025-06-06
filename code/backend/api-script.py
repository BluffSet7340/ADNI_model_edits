import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import subprocess
import nibabel as nib
from PIL import Image
from io import BytesIO
from torchvision import models, transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from skimage import exposure
from PIL import Image
from io import BytesIO
from torchvision.models import vit_b_16, ViT_B_16_Weights

app = FastAPI(title="MRI Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_labels = ['AD', 'CN', 'MCI']
UPLOAD_DIR = Path("./uploads")
STRIPPED_DIR = Path("./skull_stripped")
REORIENTED_DIR = Path("./reoriented")
REGISTERED_DIR = Path("./registered")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
synthstrip_model_path = "/mnt/c/Users/sbm76/Downloads/synthstrip.1.pt" 
MNI_TEMPLATE = "/home/saeed/fsl/data/standard/MNI152_T1_1mm_brain"

# Define model (ViT with 3-channel input for grayscale images)
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ViTClassifier, self).__init__()
        # Load the pre-trained ViT model
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # Modify the final fully connected layer to match the number of classes
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Define the model (InceptionV3 with 3-channel input for grayscale images)
class InceptionV3Classifier(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3Classifier, self).__init__()
        # Load InceptionV3 with updated pre-trained weights
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        # Disable auxiliary outputs for simplicity
        self.model.aux_logits = False
        self.model.AuxLogits = None
        # Modify the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

import torch
import numpy as np
import cv2
class GradCAM:
    def __init__(self, vit_classifier):
        self.model = vit_classifier
        self.vit = vit_classifier.model  # This is torchvision's ViT model

        # Use the last transformer block (ViT has a `blocks` attribute)
        self.target_layer = self.vit.encoder.layers[-6]

        self.gradients = None
        self.activations = None

        # Register hooks on the target layer
        self.target_layer.register_forward_hook(
            lambda module, input, output: self.save_activations(module, input, output)
        )
        self.target_layer.register_full_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(module, grad_input, grad_output)
        )

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class):
        output = self.model(input_tensor)
        self.model.zero_grad()

        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients[0].cpu().numpy()  # (tokens, channels)
        activations = self.activations[0].cpu().numpy()  # (tokens, channels)

        # Remove class token
        gradients, activations = gradients[1:], activations[1:]

        # Compute weights
        weights = np.mean(np.abs(gradients), axis=0)  # (channels,)

        # Weighted sum of activations
        cam = np.dot(activations, weights)

        num_patches = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(num_patches, num_patches)

        # Resize to 224x224
        cam = cv2.resize(cam, (224, 224))

        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

        return cam


# model = InceptionV3Classifier(num_classes=3).to(device)

model = ViTClassifier(num_classes=3).to(device)

# model.load_state_dict(torch.load("../../InceptionV3_model.pth", map_location=device))

model.load_state_dict(torch.load("../../vit_final_model_step_size.pth", map_location=device))

# model.load_state_dict(torch.load("../../vit_final_model.pth", map_location=device))


model.eval()
# gradcam = GradCAM(model, model.model.Mixed_7c) # for inceptionV3

# For ViT, use the last transformer layer in the encoder
gradcam = GradCAM(model)

def run_synthstrip(input_path, output_path):
    subprocess.run(["nipreps-synthstrip", "-i", str(input_path), "-o", str(output_path), "--model", synthstrip_model_path], check=True)

def run_fsl_reorient(input_path, output_path):
    subprocess.run(["fslreorient2std", str(input_path), str(output_path)], check=True)

def run_flirt(input_path, output_path):
    subprocess.run([
        "flirt", "-in", str(input_path),
        "-ref", MNI_TEMPLATE,
        "-out", str(output_path),
        "-dof", "12"
    ], check=True)

def normalize_slice(slice_data):
    p1, p99 = np.percentile(slice_data, (1, 99))
    clipped = np.clip(slice_data, p1, p99)
    normalized = exposure.rescale_intensity(clipped, out_range=(0, 255))
    return normalized.astype(np.uint8)

def extract_middle_slice(nifti_path):
    img = nib.load(str(nifti_path))
    data = img.get_fdata()

    if len(data.shape) != 3:
        raise ValueError("Only 3D NIfTI images are supported.")

    middle_idx = data.shape[-1] // 2
    # sagittal, coronal, and axial
    slice_data = np.rot90(data[:, :, middle_idx])
    normalized = normalize_slice(slice_data)

    pil_image = Image.fromarray(normalized).convert("L")
    pil_image = pil_image.resize((224, 224), Image.Resampling.BILINEAR)  # Specify resampling filter
    return pil_image

@app.get("/")
async def home():
    print("Welcome The model is here. REady to rock and roll")
@app.post("/preprocess/")
async def preprocess(file: UploadFile = File(...)):
    try:
        UPLOAD_DIR.mkdir(exist_ok=True)
        STRIPPED_DIR.mkdir(exist_ok=True)
        REORIENTED_DIR.mkdir(exist_ok=True)
        REGISTERED_DIR.mkdir(exist_ok=True)

        filename = ''.join(c if c.isalnum() or c in '_.' else '_' for c in file.filename)
        base_filename = filename.replace(".nii.gz", "").replace(".nii", "")
        file_id = f"{base_filename}_{os.urandom(4).hex()}"
        
        uploaded_path = UPLOAD_DIR / f"{file_id}.nii"
        with open(uploaded_path, "wb") as f:
            content = await file.read()
            f.write(content)

        stripped_path = STRIPPED_DIR / f"{file_id}.nii"
        run_synthstrip(uploaded_path, stripped_path)

        reoriented_path = REORIENTED_DIR / f"{file_id}.nii"
        run_fsl_reorient(stripped_path, reoriented_path)

        registered_path = REGISTERED_DIR / f"{file_id}.nii.gz"
        run_flirt(reoriented_path, registered_path)

        pil_image = extract_middle_slice(registered_path)
        # Convert PIL Image to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "file_id": file_id,
            "image_base64": img_base64
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Preprocessing failed: {str(e)}"}
        )

@app.get("/predict/{file_id}")
async def predict(file_id: str):
    try:
        image_path = REGISTERED_DIR / f"{file_id}.nii.gz"

        pil_image = extract_middle_slice(image_path)

        # transform = transforms.Compose([
        #     transforms.Grayscale(num_output_channels=3),
        #     transforms.Resize((299, 299)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        # ViT Transformations
        # Transformations: replicate grayscale channels to match input requirements (3 channels)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
            transforms.Resize((224, 224)),               # Resize to ResNet18 input size
            transforms.ToTensor(),                       # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # ImageNet std
        ])
        
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        return {
            "predicted_class": class_labels[pred_idx.item()],
            "probability": float(confidence),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )

@app.get("/gradcam/{file_id}")
async def gradcam_endpoint(file_id: str):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

        image_path = REGISTERED_DIR / f"{file_id}.nii.gz"
        
        # Extract middle slice and ensure RGB conversion
        pil_image = extract_middle_slice(image_path).convert("RGB")
        
        # Prepare for visualization - store original before transforms
        original_img = np.array(pil_image.resize((224, 224)))
        
        # ViT transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        pred_class = torch.argmax(model(input_tensor)).item()
        
        # Generate CAM
        cam = gradcam.generate_cam(input_tensor, pred_class)
        
        # Normalize CAM for overlay
        cam_normalized = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB for overlay

        # Overlay the heatmap onto the original image
        overlay = cv2.addWeighted(original_img.astype(np.float32), 0.6, heatmap.astype(np.float32), 0.4, 0).astype(np.uint8)

        # Create a plot with the overlay and color bar
        fig, ax = plt.subplots(figsize=(6, 6))
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap('jet')

        # Display the overlay image
        im = ax.imshow(overlay, cmap=cmap, norm=norm)
        plt.axis('off')  # Turn off axes for the overlay

        # Add color bar to the right
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation Intensity', rotation=270, labelpad=15)

        # Save the plot to a buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        plt.close(fig)

        # Encode the overlay with color bar as base64
        img_with_colorbar = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "heatmap": img_with_colorbar
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"GradCAM failed: {str(e)}"}
        )
