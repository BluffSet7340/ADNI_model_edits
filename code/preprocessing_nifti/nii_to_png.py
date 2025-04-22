import os
import nibabel as nib
import numpy as np
import imageio
from skimage import exposure

def normalize_slice(slice_data):
    """Normalize MRI slice intensities and enhance contrast"""
    p1, p99 = np.percentile(slice_data, (1, 99))
    clipped = np.clip(slice_data, p1, p99)
    normalized = exposure.rescale_intensity(clipped, out_range=(0, 255))
    return normalized.astype(np.uint8)

def process_nii(input_path, output_root, input_root, rotation=90):
    """Process NIfTI file and save middle axial slice"""
    img = nib.load(input_path)
    data = img.get_fdata()
    
    if len(data.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI, got {len(data.shape)}D: {input_path}")

    # Calculate middle slice index
    num_slices = data.shape[-1]
    middle_idx = num_slices // 2  # Integer division for middle index
    
    # Get original filename without extensions
    original_name = os.path.basename(input_path)
    base_name = original_name.replace('.nii.gz', '').replace('.nii', '')
    
    # Create output path with original name
    rel_path = os.path.relpath(os.path.dirname(input_path), start=input_root)
    output_dir = os.path.join(output_root, rel_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and save middle slice
    slice_data = np.rot90(data[:, :, middle_idx], k=rotation//90)
    normalized = normalize_slice(slice_data)
    
    output_path = os.path.join(output_dir, f"{base_name}.png")
    imageio.imwrite(output_path, normalized)
    print(f"Saved middle slice {middle_idx} for {original_name}")

# Configuration
input_root = "./flirt_registered_T1_1mm_parallel/MCI"  # Input directory containing NIfTI files
output_root = "png_output_1mm_parallel_MCI"

# Process files with original naming
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith(('.nii', '.nii.gz')):
            input_path = os.path.join(root, file)
            process_nii(input_path, output_root, input_root)