import os
from utils.logger import get_logger


logger=get_logger("cleaning")

def clean_captions_file(
    captions_file=os.path.join("Images", "captions.txt"),
    cleaned_captions_file=os.path.join("Images", "cleaned_captions.txt"),
    images_folder=os.path.join("Images", "Images"),
):
    """
    Cleans the captions file by validating entries and saving only those with existing images.

    Args:
        captions_file (str): Path to the raw captions file.
        cleaned_captions_file (str): Path to write the cleaned captions file.
        images_folder (str): Path to the folder containing images.
    """

    valid_lines = []
    skipped_lines = []

    logger.info(f"Starting cleaning process...")
    logger.info(f"Reading from: {captions_file}")
    logger.info(f"Images folder: {images_folder}")

    # Check if the input file and images folder exist
    if not os.path.exists(captions_file):
        logger.error(f"Captions file '{captions_file}' not found.")
        return
    if not os.path.exists(images_folder):
        logger.error(f"Images folder '{images_folder}' not found.")
        return

    # Read and validate the captions file
    with open(captions_file, "r") as infile:
        lines = infile.readlines()
        for line_num, line in enumerate(lines[1:], start=2):  # Skip header
            line = line.strip()
            if not line:
                skipped_lines.append(f"Line {line_num}: Empty line")
                continue

            parts = line.split(",", 1)
            if len(parts) == 2:
                image_name, _ = parts
                image_path = os.path.join(images_folder, image_name)

                if os.path.exists(image_path):
                    valid_lines.append(line)
                else:
                    skipped_lines.append(f"Line {line_num}: Missing image '{image_name}'")
            else:
                skipped_lines.append(f"Line {line_num}: Invalid format - {line}")

    # Handle existing output file
    if os.path.exists(cleaned_captions_file):
        os.remove(cleaned_captions_file)
        logger.info(f"Old file '{cleaned_captions_file}' deleted.")

    # Write cleaned captions
    if valid_lines:
        with open(cleaned_captions_file, "w") as outfile:
            outfile.write("\n".join(valid_lines))
        logger.info(f"Cleaned captions saved to: {cleaned_captions_file}")
        logger.info(f"Total valid entries: {len(valid_lines)}")
    else:
        logger.warning("No valid entries found in the captions file.")

    if skipped_lines:
        logger.warning(f"Total skipped entries: {len(skipped_lines)}")
        for skipped in skipped_lines[:10]:
            logger.warning(skipped)
    else:
        logger.info("No anomalies found. File is clean.")

# Run directly
if __name__ == "__main__":
    clean_captions_file()
