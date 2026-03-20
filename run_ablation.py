"""
Ablation experiment runner.

Loops over combinations of learning rate, epochs, and dataset size,
calls the existing train_model + evaluate_model, and logs every
result to   ablation_results.csv .

Usage:
    python run_ablation.py
    python run_ablation.py --dataset medical
    python run_ablation.py --captions ./Images/cleaned_captions.txt --images ./Images/Images

No changes to training or evaluation code required.
"""

import os
import csv
import json
import shutil
import argparse
from datetime import datetime

import torch
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration

from training.model_training import train_model
from evaluation.model_evaluation import evaluate_model
from utils.logger import get_logger

load_dotenv()
logger = get_logger("ablation")

# ---- Hyperparameter grid (edit these lists to change experiments) -----------

LEARNING_RATES = [1e-5, 5e-5, 1e-4]
EPOCH_LIST = [3, 5, 10]
DATA_FRACTIONS = [0.25, 0.50, 1.00]       # fraction of max_samples to use

# Fixed defaults (can override via CLI)
BASE_MODEL = "Salesforce/blip-image-captioning-base"
MAX_SAMPLES = 6000                         # 100 % corresponds to this number
BATCH_SIZE = 2
GRAD_ACCUM = 4
EVAL_MAX = 500                             # samples used during evaluation

RESULTS_FILE = "ablation_results.csv"

# ---- helpers ---------------------------------------------------------------

CSV_COLUMNS = [
    "experiment", "dataset", "lr", "epochs", "data_fraction", "num_samples",
    "loss", "bleu", "meteor", "rouge_l", "cider", "timestamp",
]


def _append_row(path, row: dict):
    """Append one row to the CSV, creating the file + header if needed."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---- main ------------------------------------------------------------------

def run_ablation(
    captions_file: str,
    images_folder: str,
    dataset_name: str = "flickr8k",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = os.environ.get("HF_TOKEN")
    exp_id = 0

    total = len(LEARNING_RATES) * len(EPOCH_LIST) * len(DATA_FRACTIONS)
    logger.info("=" * 60)
    logger.info(f"Starting ablation: {total} experiments on '{dataset_name}'")
    logger.info("=" * 60)

    for lr in LEARNING_RATES:
        for epochs in EPOCH_LIST:
            for frac in DATA_FRACTIONS:
                exp_id += 1
                num_samples = max(1, int(MAX_SAMPLES * frac))
                save_dir = os.path.join(
                    "saved_models", "ablation",
                    f"lr{lr}_ep{epochs}_frac{frac}",
                )

                logger.info(
                    f"\n--- Experiment {exp_id}/{total}: "
                    f"lr={lr}, epochs={epochs}, data={frac*100:.0f}% "
                    f"({num_samples} samples) ---"
                )

                # ---- Train (uses existing train_model, untouched) ----------
                train_model(
                    captions_file=captions_file,
                    images_folder=images_folder,
                    epochs=epochs,
                    batch_size=BATCH_SIZE,
                    lr=lr,
                    gradient_accumulation_steps=GRAD_ACCUM,
                    log_interval=200,
                    save_dir=save_dir,
                    hf_token=hf_token,
                    model_name=BASE_MODEL,
                    max_samples=num_samples,
                    dataset_name=dataset_name,
                )

                # ---- Evaluate (uses existing evaluate_model, untouched) ----
                processor = BlipProcessor.from_pretrained(save_dir)
                model = BlipForConditionalGeneration.from_pretrained(save_dir)
                model.to(device)

                metrics = evaluate_model(
                    model=model,
                    processor=processor,
                    images_folder=images_folder,
                    captions_file=captions_file,
                    device=device,
                    batch_size=BATCH_SIZE,
                    max_samples=EVAL_MAX,
                )

                # ---- Log to CSV --------------------------------------------
                row = {
                    "experiment": exp_id,
                    "dataset": dataset_name,
                    "lr": lr,
                    "epochs": epochs,
                    "data_fraction": frac,
                    "num_samples": num_samples,
                    "loss": round(metrics["loss"], 4),
                    "bleu": round(metrics["bleu"], 4),
                    "meteor": round(metrics["meteor"], 4),
                    "rouge_l": round(metrics["rouge_l"], 4),
                    "cider": round(metrics["cider"], 4),
                    "timestamp": datetime.now().isoformat(),
                }
                _append_row(RESULTS_FILE, row)
                logger.info(f"  → metrics: {json.dumps(row, indent=2)}")

                # Optional: delete checkpoint to save disk (keep CSV)
                # shutil.rmtree(save_dir, ignore_errors=True)

    logger.info("=" * 60)
    logger.info(f"Ablation complete. Results saved to {RESULTS_FILE}")
    logger.info("=" * 60)


# ---- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--captions", default="./Images/cleaned_captions.txt")
    parser.add_argument("--images", default="./Images/Images")
    parser.add_argument("--dataset", default="flickr8k", choices=["flickr8k", "medical"])
    args = parser.parse_args()

    run_ablation(
        captions_file=args.captions,
        images_folder=args.images,
        dataset_name=args.dataset,
    )
