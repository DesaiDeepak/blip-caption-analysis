"""
Medical Image Captioning Dataset loader.

Drop-in replacement for FlickrDataset — same __getitem__ return format,
same constructor signature, same collate_fn compatibility.

Expected caption file format (identical to Flickr):
    image_filename,caption text here
"""

import os
import re
from torch.utils.data import Dataset
from PIL import Image
from transformers import BlipProcessor


def normalize_medical_caption(text: str) -> str:
    """Light normalisation for radiology reports."""
    text = text.lower()
    text = re.sub(r"\(fig\s*\d+[a-z]?\)", "", text)
    text = re.sub(r"xxxx", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class MedicalDataset(Dataset):
    """
    PyTorch Dataset for medical image-caption pairs.

    Same interface as FlickrDataset:
        __init__(captions_file, images_folder, processor)
        __getitem__  → {"pixel_values", "input_ids", "attention_mask"}
        __len__      → int

    Works with the existing collate_fn from data.dataset.
    """

    def __init__(self, captions_file, images_folder, processor=None):
        self.images_folder = images_folder
        self.processor = processor or BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.data = self._load_data(captions_file)

    def _load_data(self, captions_file):
        data = []
        with open(captions_file, "r") as fh:
            for line in fh:
                parts = line.strip().split(",", 1)
                if len(parts) != 2:
                    continue
                image_name, caption = parts
                caption = normalize_medical_caption(caption)
                if not caption:
                    continue
                image_path = os.path.join(self.images_folder, image_name)
                if os.path.exists(image_path):
                    data.append({"image": image_path, "caption": caption})
                else:
                    print(f"Warning: Image {image_name} not found in {self.images_folder}.")
        if not data:
            raise ValueError(
                "Medical dataset is empty. "
                "Ensure captions file and images folder are set up correctly."
            )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item["image"]).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {item['image']}: {e}")

        inputs = self.processor(
            images=image,
            text=item["caption"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if "pixel_values" not in inputs:
            raise KeyError("Missing 'pixel_values' in processed inputs.")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }
