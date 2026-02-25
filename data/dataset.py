import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from transformers import BlipProcessor


def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch.
    Shared by training and evaluation DataLoaders.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


class FlickrDataset(Dataset):
    """
    Custom PyTorch Dataset for handling Flickr-style image-caption datasets.

    Args:
        captions_file (str): Path to the cleaned captions file.
        images_folder (str): Path to the folder containing images.
        processor (BlipProcessor): Pre-trained BLIP processor for image-text preprocessing.
    """
    def __init__(self, captions_file, images_folder, processor=None):
        self.images_folder = images_folder
        self.processor = processor or BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.data = self._load_data(captions_file)

    def _load_data(self, captions_file):
        """
        Load image paths and captions from the captions file.

        Args:
            captions_file (str): Path to the cleaned captions file.

        Returns:
            list: A list of dictionaries containing image paths and captions.
        """
        data = []
        with open(captions_file, "r") as file:
            for line in file:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    image_name, caption = parts
                    image_path = os.path.join(self.images_folder, image_name)
                    if os.path.exists(image_path):
                        data.append({"image": image_path, "caption": caption})
                    else:
                        print(f"Warning: Image {image_name} not found in {self.images_folder}.")
        if not data:
            raise ValueError("Dataset is empty. Ensure captions file and images folder are set up correctly.")
        return data

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the processed image tensor and text tokens.
        """
        item = self.data[idx]
        try:
            # Ensure the image loads correctly
            image = Image.open(item["image"]).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {item['image']}: {e}")

        # Preprocess the image and text
        inputs = self.processor(
            images=image,             # Note the 'images' argument (plural, required for BLIP processor)
            text=item["caption"],     # Caption for the image
            return_tensors="pt",      # Return PyTorch tensors
            padding=True,             # Pad captions for consistency
            truncation=True           # Truncate captions if necessary
        )

        # Debugging: Print keys of the inputs to verify "pixel_values" exists
        if "pixel_values" not in inputs:
            raise KeyError("Missing 'pixel_values' in processed inputs. Check processor or image.")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # Image tensor
            "input_ids": inputs["input_ids"].squeeze(0),        # Caption input IDs
            "attention_mask": inputs["attention_mask"].squeeze(0)  # Attention mask
        }
