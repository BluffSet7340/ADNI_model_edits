import requests
import base64

# Base URL of your running FastAPI server
BASE_URL = "http://127.0.0.1:8000"

# Path to a test .nii.gz file (replace this with your actual file path)
NIFTI_FILE_PATH = "./mri-images/Test Cases for T1_augmented_hflip/Unprocessed NIFTI Files/MCI_ADNI_035_S_0033_MR_MPR-R__GradWarp__B1_Correction_Br_20070808131913345_S33648_I65851.nii"

# 1️⃣ Test Preprocessing Route
def test_preprocess():
    with open(NIFTI_FILE_PATH, 'rb') as f:
        files = {'file': (NIFTI_FILE_PATH, f, 'application/octet-stream')}
        response = requests.post(f"{BASE_URL}/preprocess/", files=files)
    
    print("Preprocess Response Status Code:", response.status_code)
    print("Preprocess Raw Text Response:", response.text)

    # Check if response is JSON before trying to parse it
    try:
        json_response = response.json()
        print("Parsed JSON:", json_response)
        return json_response["file_id"]
    except Exception as e:
        print("Failed to parse JSON response:", e)
        return None


# 2️⃣ Test Prediction Route
def test_predict(file_id):
    response = requests.get(f"{BASE_URL}/predict/{file_id}")
    print("Predict Response:", response.status_code, response.json())

# 3️⃣ Test GradCAM Heatmap Route
def test_gradcam(file_id):
    response = requests.get(f"{BASE_URL}/gradcam/{file_id}")
    result = response.json()
    print("GradCAM Response:", response.status_code)

    # Save base64 image to a file for viewing
    with open(f"{file_id}_heatmap.png", "wb") as f:
        f.write(base64.b64decode(result["heatmap"]))
    print(f"GradCAM heatmap saved as {file_id}_heatmap.png")

# Run full test
if __name__ == "__main__":
    file_id = test_preprocess()   # Run preprocessing
    test_predict(file_id)         # Predict class
    test_gradcam(file_id)         # Get GradCAM heatmap
