import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from torch.utils.data import DataLoader, Subset
from transformers import BlipProcessor, BlipForConditionalGeneration
from data.dataset import FlickrDataset, collate_fn
from torch.cuda.amp import GradScaler, autocast
from dotenv import load_dotenv
from utils.logger import get_logger

logger=get_logger("training")

load_dotenv()


# Check if the model is already trained
def check_trained_model(save_dir):
    """
    Checks if a fine-tuned model already exists in the save_dir.

    Args:
        save_dir (str): Directory where the model is saved.

    Returns:
        bool: True if the model exists, False otherwise.
    """
    if os.path.exists(save_dir):
        logger.info(f"Trained model found at: {save_dir}")
        return True
    else:
        logger.info("No trained model found. Proceeding with training...")
        return False


# Train the model
def train_model(
    captions_file,
    images_folder,
    epochs=1,
    batch_size=4,
    lr=5e-5,
    gradient_accumulation_steps=4,
    log_interval=100,
    save_dir="saved_models/fine_tuned_blip",
    hf_token=None,
    model_name="Salesforce/blip-image-captioning-base",
    max_samples=1000,
):
    """
    Trains the BLIP model on a subset of the dataset.

    Args:
        captions_file (str): Path to the cleaned captions file.
        images_folder (str): Path to the folder containing images.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        gradient_accumulation_steps (int): Number of steps for gradient accumulation.
        log_interval (int): Log progress after this many images.
        save_dir (str): Directory to save the fine-tuned model.
        hf_token (str): Hugging Face token for downloading the base model.
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    logger.info(f"Loading pre-trained BLIP processor and model: {model_name}...")
    processor = BlipProcessor.from_pretrained(model_name, token=hf_token)
    model = BlipForConditionalGeneration.from_pretrained(model_name, token=hf_token)

    # Load the dataset and limit to the first 6000 samples
    logger.info("Loading dataset...")
    full_dataset = FlickrDataset(captions_file, images_folder, processor)
    dataset = Subset(full_dataset, range(min(len(full_dataset), max_samples)))

    logger.info(f"Using {len(dataset)} samples for training.")

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=False
    )

    # Set up the optimizer and device
    # MPS (Apple Silicon) causes bus errors during backward pass with BLIP - use CPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler() if torch.cuda.is_available() else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        images_processed = 0

        logger.info(f"Starting epoch {epoch + 1}/{epochs}...")
        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images_processed += len(pixel_values)

            if scaler:
                with autocast(device_type="cuda"):
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
                    loss = outputs.loss
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()

            if images_processed % log_interval == 0:
                logger.info(f"Processed {images_processed}/{len(train_loader.dataset)} images.")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

    logger.info(f"Saving the fine-tuned model to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    logger.info(f"Model saved successfully at {save_dir}")
