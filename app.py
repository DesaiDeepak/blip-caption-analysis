import os
import streamlit as st
from PIL import Image
import torch
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from create_project_folders import ensure_dataset
from cleaning.data_cleaning import clean_captions_file
from training.model_training import train_model

load_dotenv()

MODEL_DIR = "saved_models/fine_tuned_blip"
IMAGES_FOLDER = "./Images/Images"
CAPTIONS_FILE = "./Images/captions.txt"
CLEANED_CAPTIONS_FILE = "./Images/cleaned_captions.txt"


@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    processor = BlipProcessor.from_pretrained(MODEL_DIR)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.to(device)
    return processor, model, device


# If no trained model exists, run the full pipeline before loading
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    st.title("BLIP Image Captioning App")
    st.warning("No trained model found. Running the training pipeline automatically...")

    st.write("**Step 1:** Checking dataset...")
    with st.spinner("Downloading dataset if needed..."):
        ensure_dataset(images_folder=IMAGES_FOLDER, captions_file=CAPTIONS_FILE)
    st.success("Dataset ready.")

    st.write("**Step 2:** Cleaning captions...")
    with st.spinner("Cleaning captions..."):
        clean_captions_file(
            captions_file=CAPTIONS_FILE,
            cleaned_captions_file=CLEANED_CAPTIONS_FILE,
            images_folder=IMAGES_FOLDER,
        )
    st.success("Captions cleaned.")

    st.write("**Step 3:** Training model — this will take a while...")
    with st.spinner("Training in progress, please wait..."):
        train_model(
            captions_file=CLEANED_CAPTIONS_FILE,
            images_folder=IMAGES_FOLDER,
            hf_token=os.environ.get("HF_TOKEN"),
        )
    st.success("Model trained and saved! Reloading app...")
    st.experimental_rerun()


processor, model, device = load_model()

# Title of the application
st.title("BLIP Image Captioning App")
st.write("Upload an image, and I'll generate a caption for it!")

# File uploader widget for image input
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "gif", "bmp", "pdf"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Process the image and generate the caption
    if uploaded_image.type == "application/pdf":
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(uploaded_image.read())
        image = images[0]
    else:
        image = Image.open(uploaded_image).convert("RGB")

    # Preprocess the image using the BLIP processor
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

    # Show the generated caption
    st.subheader("Generated Caption:")
    st.write(caption)

else:
    st.write("Please upload an image to generate a caption.")
