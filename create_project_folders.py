import os
import urllib.request
import zipfile
import shutil


def ensure_dataset(images_folder, captions_file):
    """
    Checks if the Flickr30k dataset (images + captions) is present.
    If not, downloads and extracts it automatically from the GitHub mirror.

    Args:
        images_folder (str): Path to the folder that should contain images (e.g. ./Images/Images).
        captions_file (str): Path to the captions file (e.g. ./Images/captions.txt).
    """
    images_exist = os.path.isdir(images_folder) and len(os.listdir(images_folder)) > 0
    captions_exist = os.path.isfile(captions_file)

    if images_exist and captions_exist:
        print("Dataset already present. Skipping download.")
        return

    print("\n--- Dataset not found. Downloading Flickr30k Dataset ---")
    dest_dir = os.path.dirname(images_folder)  # ./Images
    os.makedirs(images_folder, exist_ok=True)

    _download_and_extract(dest_dir)

    # Verify after download
    if not os.path.isdir(images_folder) or len(os.listdir(images_folder)) == 0:
        raise RuntimeError(
            f"Download completed but no images found in '{images_folder}'. "
            "Check the download URLs or extract the dataset manually."
        )
    if not os.path.isfile(captions_file):
        raise RuntimeError(
            f"Download completed but captions file not found at '{captions_file}'. "
            "Ensure the archive contains a captions.txt file."
        )

    print("Dataset is ready.")


def _download_and_extract(dest_dir):
    """
    Downloads the split Flickr30k archive parts from GitHub, combines and extracts them.

    Args:
        dest_dir (str): Directory to download and extract into (e.g. ./Images).
    """
    urls = [
        "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00",
        "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01",
        "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02",
    ]

    parts = []
    for url in urls:
        filename = os.path.join(dest_dir, os.path.basename(url))
        print(f"Downloading {os.path.basename(url)}...")
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            for p in parts:
                if os.path.exists(p):
                    os.remove(p)
            if os.path.exists(filename):
                os.remove(filename)
            raise RuntimeError(f"Failed to download {url}: {e}")
        parts.append(filename)

    combined_zip = os.path.join(dest_dir, "flickr30k.zip")
    print("Combining parts...")
    with open(combined_zip, "wb") as outfile:
        for part in parts:
            with open(part, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    print("Extracting archive...")
    with zipfile.ZipFile(combined_zip, "r") as zip_ref:
        zip_ref.extractall(dest_dir)

    for part in parts:
        os.remove(part)
    os.remove(combined_zip)
    print("Extraction complete.")


if __name__ == "__main__":
    ensure_dataset(
        images_folder="./Images/Images",
        captions_file="./Images/captions.txt",
    )
