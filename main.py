import os
import yaml
import torch
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from cleaning.data_cleaning import clean_captions_file
from training.model_training import train_model, check_trained_model
from evaluation.model_evaluation import evaluate_model
from create_project_folders import ensure_dataset
from utils.logger import get_logger

load_dotenv()
logger=get_logger("main")

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    paths = config["paths"]
    captions_file = paths["captions_file"]
    cleaned_captions_file = paths["cleaned_captions_file"]
    images_folder = paths["images_folder"]
    model_save_dir = paths["model_save_dir"]

    training = config["training"]
    model_cfg = config["model"]
    evaluation = config["evaluation"]

    # Step 0: Ensure dataset is present, download if missing
    logger.info("\n--- Step 0: Checking Dataset ---")
    ensure_dataset(images_folder=images_folder, captions_file=captions_file)

    # Step 1: Clean the captions file
    logger.info("\n--- Step 1: Cleaning Captions File ---")
    clean_captions_file(
        captions_file=captions_file,
        cleaned_captions_file=cleaned_captions_file,
        images_folder=images_folder,
    )

    # Step 2: Check if the model is already trained
    logger.info("\n--- Step 2: Checking Trained Model ---")
    if not check_trained_model(model_save_dir):
        logger.info("\n--- Step 3: Training Model ---")
        train_model(
            captions_file=cleaned_captions_file,
            images_folder=images_folder,
            epochs=training["epochs"],
            batch_size=training["batch_size"],
            lr=training["learning_rate"],
            gradient_accumulation_steps=training["gradient_accumulation_steps"],
            log_interval=training["log_interval"],
            save_dir=model_save_dir,
            hf_token=os.environ.get("HF_TOKEN"),
            model_name=model_cfg["base_model"],
            max_samples=training["max_samples"],
        )
    else:
        logger.info("Model already trained. Proceeding to evaluation.")

    # Step 3: Initialize processor and model
    logger.info("\n--- Step 3: Initializing Processor and Model ---")
    processor = BlipProcessor.from_pretrained(model_cfg["base_model"])
    model = BlipForConditionalGeneration.from_pretrained(model_save_dir)

    # MPS causes bus errors with BLIP inference - use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Step 4: Running Evaluation
    logger.info("\n--- Step 4: Running Evaluation ---")
    evaluate_model(
        model=model,
        processor=processor,
        images_folder=images_folder,
        captions_file=cleaned_captions_file,
        device=device,
        batch_size=evaluation["batch_size"],
        max_samples=evaluation["max_samples"],
    )

if __name__ == "__main__":
    main()
