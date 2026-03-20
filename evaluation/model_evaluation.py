import os
import math
from collections import Counter
import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from PIL import Image
from data.dataset import collate_fn
from utils.logger import get_logger

logger=get_logger("evaluation")

def preprocess_images_and_captions(images_folder, captions_file, processor, max_samples=1000):
    """
    Preprocess the images and captions directly without using a dataset class.

    Args:
        images_folder (str): Path to the folder containing images.
        captions_file (str): Path to the captions file.
        processor: The BLIP processor for image-caption processing.
        max_samples (int): Maximum number of samples to load into memory.

    Returns:
        List[Dict]: A list of processed samples (image tensors and captions).
    """
    data = []
    with open(captions_file, "r") as file:
        for line in file:
            if len(data) >= max_samples:
                break
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
                    logger.warning(f"Image {image_name} not found in {images_folder}.")
    return data


# ---- CIDEr (lightweight, no pycocoevalcap dependency) ----------------------

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _compute_cider(references, hypotheses, n=4):
    """
    Simplified CIDEr-D score averaged over all samples.
    Each reference / hypothesis is a plain string.
    """
    if not references:
        return 0.0

    # Build document-frequency counts across all references
    doc_freq = Counter()
    for ref in references:
        seen = set()
        for k in range(1, n+1):
            for ng in _ngrams(ref.lower().split(), k):
                if ng not in seen:
                    doc_freq[ng] += 1
                    seen.add(ng)

    num_docs = len(references)
    scores = []

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        score_sum = 0.0
        weight_sum = 0

        for k in range(1, n+1):
            ref_ng = Counter(_ngrams(ref_tokens, k))
            hyp_ng = Counter(_ngrams(hyp_tokens, k))
            common = set(ref_ng) | set(hyp_ng)
            if not common:
                continue
            # TF-IDF vectors
            ref_vec, hyp_vec = [], []
            for ng in common:
                idf = math.log(max(1.0, num_docs) / (1.0 + doc_freq.get(ng, 0)))
                ref_vec.append(ref_ng.get(ng, 0) * idf)
                hyp_vec.append(hyp_ng.get(ng, 0) * idf)
            dot = sum(a*b for a, b in zip(ref_vec, hyp_vec))
            mag_r = math.sqrt(sum(v*v for v in ref_vec)) or 1.0
            mag_h = math.sqrt(sum(v*v for v in hyp_vec)) or 1.0
            score_sum += dot / (mag_r * mag_h)
            weight_sum += 1

        scores.append(score_sum / weight_sum if weight_sum else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


# ---- Evaluation ------------------------------------------------------------

def evaluate_model(model, processor, images_folder, captions_file, device, batch_size=4, max_samples=1000):
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
    logger.info("Starting model evaluation...")

    # Preprocess the images and captions
    data = preprocess_images_and_captions(images_folder, captions_file, processor, max_samples=max_samples)
    test_loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    model.eval()
    total_loss = 0
    bleu_scores = []
    all_actual = []
    all_predicted = []

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
                all_actual.append(actual)
                all_predicted.append(predicted)

    # ---- Aggregate metrics --------------------------------------------------
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    # METEOR (nltk)
    meteor_scores = [
        meteor_score([ref.split()], hyp.split())
        for ref, hyp in zip(all_actual, all_predicted)
    ]
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    # ROUGE-L (rouge-score library)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(all_actual, all_predicted)
    ]
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    # CIDEr (lightweight inline implementation)
    avg_cider = _compute_cider(all_actual, all_predicted)

    # ---- Log all metrics ----------------------------------------------------
    logger.info(f"Test Set Loss:        {avg_loss:.4f}")
    logger.info(f"Average BLEU Score:   {avg_bleu:.4f}")
    logger.info(f"Average METEOR Score: {avg_meteor:.4f}")
    logger.info(f"Average ROUGE-L Score:{avg_rouge_l:.4f}")
    logger.info(f"Average CIDEr Score:  {avg_cider:.4f}")

    return {
        "loss": avg_loss,
        "bleu": avg_bleu,
        "meteor": avg_meteor,
        "rouge_l": avg_rouge_l,
        "cider": avg_cider,
    }
