import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


API_KEY=os.environ.get("API_KEY")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODEL_DIR = "saved_models/fine_tuned_blip"
IMAGES_FOLDER = "./Images/Images"
CAPTIONS_FILE = "./Images/captions.txt"
CLEANED_CAPTIONS_FILE = "./Images/cleaned_captions.txt"


def check_api_healthy():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200 and response.json().get("model_loaded")
    except requests.exceptions.ConnectionError:
        return False


# If no trained model exists, show an info banner (the API server
# already falls back to the public base model automatically).
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    st.info(
        "ℹ️ No fine-tuned model found at `saved_models/fine_tuned_blip/`. "
        "The API is using the public **Salesforce/blip-image-captioning-base** model. "
        "To train your own, run:  \n"
        "`python -c \"from training.model_training import train_model; "
        "train_model('./Images/cleaned_captions.txt', './Images/Images', epochs=5)\"`"
    )


# Title of the application
st.title("BLIP Image Captioning App")
st.write("Upload an image, and I'll generate a caption for it!")

# Check API connectivity
if not check_api_healthy():
    st.error("API server is not reachable. Please start it with: `uvicorn api.inference:app --port 8000`")
    st.stop()

# File uploader widget for image input
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Send image to FastAPI and get caption
    with st.spinner("Generating caption..."):
        response = requests.post(
            f"{API_URL}/caption",
            files={"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)},
            headers={"X-API-Key": API_KEY}
        )

    if response.status_code == 200:
        caption = response.json()["caption"]
        st.subheader("Generated Caption:")
        st.write(caption)
    else:
        st.error(f"Error from API: {response.json().get('detail', 'Unknown error')}")

else:
    st.write("Please upload an image to generate a caption.")
