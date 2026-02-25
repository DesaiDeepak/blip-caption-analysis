import os
import torch
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from cleaning.data_cleaning import clean_captions_file
from training.model_training import train_model, check_trained_model
from evaluation.model_evaluation import evaluate_model
from create_project_folders import ensure_dataset

load_dotenv()

def main():
    # Define file paths
    captions_file = "./Images/captions.txt"
    cleaned_captions_file = "./Images/cleaned_captions.txt"
    images_folder = "./Images/Images"
    model_save_dir = "saved_models/fine_tuned_blip"

    # Step 0: Ensure dataset is present, download if missing
    print("\n--- Step 0: Checking Dataset ---")
    ensure_dataset(images_folder=images_folder, captions_file=captions_file)

    # Step 1: Clean the captions file
    print("\n--- Step 1: Cleaning Captions File ---")
    clean_captions_file(
        captions_file=captions_file,
        cleaned_captions_file=cleaned_captions_file,
        images_folder=images_folder,
    )

    # Step 2: Check if the model is already trained
    print("\n--- Step 2: Checking Trained Model ---")
    if not check_trained_model(model_save_dir):
        print("\n--- Step 3: Training Model ---")
        train_model(
            captions_file=cleaned_captions_file,
            images_folder=images_folder,
            epochs=1,
            batch_size=4,
            lr=5e-5,
            gradient_accumulation_steps=4,
            log_interval=100,
            save_dir=model_save_dir,
            hf_token=os.environ.get("HF_TOKEN"),
        )
    else:
        print("Model already trained. Proceeding to evaluation.")

    # Step 3: Initialize processor and model
    print("\n--- Step 3: Initializing Processor and Model ---")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(model_save_dir)

    # Use MPS if available, else fall back to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Step 4: Running Evaluation
    print("\n--- Step 4: Running Evaluation ---")
    evaluate_model(
        model=model,
        processor=processor,
        images_folder=images_folder,
        captions_file=cleaned_captions_file,
        device=device,
    )

if __name__ == "__main__":
    main()
