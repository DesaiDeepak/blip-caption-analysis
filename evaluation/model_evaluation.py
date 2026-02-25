import os
import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from PIL import Image
from data.dataset import collate_fn


def preprocess_images_and_captions(images_folder, captions_file, processor):
    """
    Preprocess the images and captions directly without using a dataset class.

    Args:
        images_folder (str): Path to the folder containing images.
        captions_file (str): Path to the captions file.
        processor: The BLIP processor for image-caption processing.

    Returns:
        List[Dict]: A list of processed samples (image tensors and captions).
    """
    data = []
    with open(captions_file, "r") as file:
        for line in file:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                image_name, caption = parts
                image_path = os.path.join(images_folder, image_name)
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                    inputs = processor(images=image, text=caption, return_tensors="pt", padding=True)
                    data.append({
                        "pixel_values": inputs["pixel_values"].squeeze(0),
                        "input_ids": inputs["input_ids"].squeeze(0),
                        "attention_mask": inputs["attention_mask"].squeeze(0),
                    })
                else:
                    print(f"Warning: Image {image_name} not found in {images_folder}.")
    return data

def evaluate_model(model, processor, images_folder, captions_file, device, batch_size=4):
    """
    Evaluate the fine-tuned model on the test set and calculate BLEU scores.

    Args:
        model: The fine-tuned BLIP model.
        processor: The BLIP processor for image and caption processing.
        images_folder: Path to the folder containing images.
        captions_file: Path to the captions file.
        device: The device (CPU or GPU) for evaluation.
        batch_size: Number of samples per batch for evaluation.
    """
    print("\n--- Evaluating Model ---")

    # Preprocess the images and captions
    data = preprocess_images_and_captions(images_folder, captions_file, processor)
    test_loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    model.eval()
    total_loss = 0
    bleu_scores = []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Decode predictions and compute BLEU scores
            predicted_ids = model.generate(pixel_values)
            predicted_captions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            actual_captions = processor.batch_decode(input_ids, skip_special_tokens=True)

            for actual, predicted in zip(actual_captions, predicted_captions):
                bleu_score = sentence_bleu(
                    [actual.split()], predicted.split(), smoothing_function=SmoothingFunction().method1
                )
                bleu_scores.append(bleu_score)

    avg_loss = total_loss / len(test_loader)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    print(f"Test Set Loss: {avg_loss:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
