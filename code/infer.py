"""Evaluate one trained KDH-CAD checkpoint on a classification split.

The public test scripts use this module instead of re-running training.  By
default it writes only a compact ``metrics.json``; detailed per-sample
predictions remain opt-in with ``--save-predictions``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from tqdm import tqdm

from args import parse_args
from MultiviewDataset import create_dataloaders
from encoderModels.classifierNet import PrototypeShiftClassifier


def parse_inference_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    """Parse evaluation-only flags while forwarding model flags to ``args.py``."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True, help="Path to a model state dict.")
    parser.add_argument("--output-dir", default="inference_results", help="Directory for compact metrics.")
    parser.add_argument("--save-predictions", action="store_true", help="Also save per-sample probabilities.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most this many samples (0 means all).")
    inference_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        model_args = parse_args()
    finally:
        sys.argv = original_argv
    return inference_args, model_args


def load_class_names(train_json: str) -> list[str]:
    with open(train_json, "r", encoding="utf-8") as handle:
        train_data = json.load(handle)
    label_to_name = {item["label"]: item["label_name"] for item in train_data}
    return [label_to_name[index] for index in sorted(label_to_name)]


def run_inference(
    checkpoint_path: str,
    output_dir: str,
    args: argparse.Namespace,
    *,
    save_predictions: bool = False,
    limit: int = 0,
) -> dict[str, object]:
    """Load a checkpoint, evaluate its test split, and return classification metrics."""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.train_json)
    _, _, test_loader = create_dataloaders(args)
    test_dataset_list = test_loader.dataset.data

    print(f"[*] Initializing model: {args.model_type} / {args.classifier_mode}")
    model = PrototypeShiftClassifier(args, class_names)
    print(f"[*] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()

    labels_all: list[int] = []
    predictions_all: list[int] = []
    detailed_results: list[dict[str, object]] = []
    current_idx = 0

    with torch.inference_mode():
        for images_batch, labels in tqdm(test_loader, desc="Evaluating"):
            logits = model(images_batch)[0]
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            for batch_index in range(len(labels)):
                if limit and len(labels_all) >= limit:
                    break
                true_label = int(labels[batch_index].item())
                predicted_label = int(predictions[batch_index].item())
                labels_all.append(true_label)
                predictions_all.append(predicted_label)

                if save_predictions:
                    sample_info = test_dataset_list[current_idx]
                    distribution = probabilities[batch_index].detach().cpu().float().numpy()
                    detailed_results.append({
                        "filename": os.path.basename(sample_info.get("image_paths", ["unknown"])[0]),
                        "true_label": true_label,
                        "true_name": class_names[true_label],
                        "pred_label": predicted_label,
                        "pred_name": class_names[predicted_label],
                        "is_correct": predicted_label == true_label,
                        "probabilities": {
                            class_names[index]: round(float(distribution[index]), 6)
                            for index in range(len(class_names))
                        },
                    })
                current_idx += 1
            if limit and len(labels_all) >= limit:
                break

    if not labels_all:
        raise RuntimeError("No evaluable samples were found in the requested test split.")

    metrics: dict[str, object] = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "test_json": str(Path(args.test_json).resolve()),
        "num_samples": len(labels_all),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "balanced_accuracy": balanced_accuracy_score(labels_all, predictions_all),
        "macro_f1": f1_score(labels_all, predictions_all, average="macro", zero_division=0),
        "micro_f1": f1_score(labels_all, predictions_all, average="micro", zero_division=0),
    }
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    if save_predictions:
        with (output_path / "predictions.json").open("w", encoding="utf-8") as handle:
            json.dump(detailed_results, handle, indent=2, ensure_ascii=False)

    print("\nKDH-CAD classification results")
    print(f"  Samples:   {metrics['num_samples']}")
    print(f"  Accuracy:  {float(metrics['accuracy']) * 100:.2f}%")
    print(f"  BACC:      {float(metrics['balanced_accuracy']) * 100:.2f}%")
    print(f"  Macro-F1:  {float(metrics['macro_f1']):.4f}")
    print(f"  Micro-F1:  {float(metrics['micro_f1']):.4f}")
    print(classification_report(
        labels_all,
        predictions_all,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))
    print(f"[*] Compact metrics saved to: {output_path / 'metrics.json'}")
    return metrics


def main() -> None:
    inference_args, model_args = parse_inference_args()
    run_inference(
        inference_args.checkpoint,
        inference_args.output_dir,
        model_args,
        save_predictions=inference_args.save_predictions,
        limit=inference_args.limit,
    )


if __name__ == "__main__":
    main()
