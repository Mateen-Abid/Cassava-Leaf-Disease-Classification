import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


# ************************************* DATA MODELS ************************************* #
@dataclass
class Sample:
    image_id: str
    label: int


# ************************************* LOGGING ************************************* #
class RunLogger:
    def __init__(self) -> None:
        self._lines: List[str] = []

    def log(self, message: str = "") -> None:
        print(message)
        self._lines.append(message)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")


# ************************************* DATASET ************************************* #
class CassavaSplitDataset(Dataset):
    def __init__(
        self,
        split_csv_path: Path,
        image_dir: Path,
        transform: transforms.Compose,
    ) -> None:
        self.samples = self._read_split(split_csv_path)
        self.image_dir = image_dir
        self.transform = transform

    @staticmethod
    def _read_split(path: Path) -> List[Sample]:
        samples: List[Sample] = []
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                samples.append(Sample(image_id=row["image_id"], label=int(row["label"])))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        image_path = self.image_dir / sample.image_id
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
        image_tensor = self.transform(image_rgb)
        return image_tensor, sample.label, sample.image_id


# ************************************* MODEL FACTORIES ************************************* #
class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_transfer_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.20),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "transfer":
        return build_transfer_model(num_classes=num_classes)
    raise ValueError(f"Unsupported model name: {model_name}")


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]
    model = build_model(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


# ************************************* METRICS ************************************* #
def confusion_matrix_from_predictions(
    labels: Sequence[int],
    predictions: Sequence[int],
    num_classes: int,
) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, pred_label in zip(labels, predictions):
        matrix[true_label][pred_label] += 1
    return matrix


def per_class_metrics(confusion: List[List[int]]) -> Dict[int, Dict[str, float]]:
    num_classes = len(confusion)
    results: Dict[int, Dict[str, float]] = {}
    for class_id in range(num_classes):
        tp = confusion[class_id][class_id]
        fp = sum(confusion[row][class_id] for row in range(num_classes) if row != class_id)
        fn = sum(confusion[class_id][col] for col in range(num_classes) if col != class_id)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(confusion[class_id])
        results[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return results


def macro_f1(per_class: Dict[int, Dict[str, float]]) -> float:
    return sum(info["f1"] for info in per_class.values()) / max(len(per_class), 1)


def accuracy_score(labels: Sequence[int], predictions: Sequence[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(int(y_true == y_pred) for y_true, y_pred in zip(labels, predictions))
    return correct / len(labels)


def top_failure_pairs(confusion: List[List[int]], max_pairs: int = 5) -> List[Dict]:
    rows = []
    for true_label in range(len(confusion)):
        for pred_label in range(len(confusion)):
            if true_label == pred_label:
                continue
            rows.append(
                {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": confusion[true_label][pred_label],
                }
            )
    rows = [row for row in rows if row["count"] > 0]
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows[:max_pairs]


# ************************************* EVALUATION ************************************* #
def run_evaluation(
    model_name: str,
    checkpoint_path: Path,
    test_split: Path,
    image_dir: Path,
    label_map: Dict[int, str],
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    force_cpu: bool,
    max_test_samples: int,
    logger: RunLogger,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    model, checkpoint = load_checkpoint_model(checkpoint_path=checkpoint_path, device=device)

    image_size = int(checkpoint.get("image_size", 224))
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = CassavaSplitDataset(
        split_csv_path=test_split,
        image_dir=image_dir,
        transform=eval_transform,
    )
    dataset_for_run: Dataset = dataset
    if max_test_samples > 0 and max_test_samples < len(dataset):
        dataset_for_run = Subset(dataset, list(range(max_test_samples)))
        logger.log(f"Using limited test subset: {max_test_samples} samples")

    loader = DataLoader(
        dataset_for_run,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.log(" ")
    logger.log(f"==================== EVALUATION: {model_name.upper()} ====================")
    logger.log(f"Checkpoint: {checkpoint_path.as_posix()}")
    logger.log(f"Device: {device}")
    logger.log(f"Test samples: {len(dataset_for_run)}")

    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_confidences: List[float] = []
    all_image_ids: List[str] = []

    model.eval()
    with torch.no_grad():
        for images, labels, image_ids in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())
            all_image_ids.extend(list(image_ids))

    confusion = confusion_matrix_from_predictions(
        labels=all_labels,
        predictions=all_predictions,
        num_classes=len(label_map),
    )
    per_class = per_class_metrics(confusion)
    metric_accuracy = accuracy_score(all_labels, all_predictions)
    metric_macro_f1 = macro_f1(per_class)
    failures = top_failure_pairs(confusion, max_pairs=7)

    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Quantitative results.
    metrics_payload = {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path.as_posix(),
        "test_split": test_split.as_posix(),
        "num_test_samples": len(dataset_for_run),
        "accuracy": metric_accuracy,
        "macro_f1": metric_macro_f1,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "top_failure_pairs": failures,
    }
    (model_output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    # Save prediction table for later analysis/UI.
    prediction_rows = []
    for image_id, true_label, pred_label, conf in zip(
        all_image_ids,
        all_labels,
        all_predictions,
        all_confidences,
    ):
        prediction_rows.append(
            {
                "image_id": image_id,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": round(conf, 6),
                "is_correct": int(true_label == pred_label),
            }
        )
    (model_output_dir / "predictions.json").write_text(
        json.dumps(prediction_rows, indent=2),
        encoding="utf-8",
    )

    # Qualitative: misclassified and low-confidence samples.
    misclassified = [row for row in prediction_rows if row["is_correct"] == 0]
    low_confidence = sorted(prediction_rows, key=lambda row: row["confidence"])[:20]
    misclassified_sorted = sorted(misclassified, key=lambda row: row["confidence"])[:20]

    qualitative_dir = model_output_dir / "qualitative"
    misclassified_dir = qualitative_dir / "misclassified"
    low_conf_dir = qualitative_dir / "low_confidence"
    misclassified_dir.mkdir(parents=True, exist_ok=True)
    low_conf_dir.mkdir(parents=True, exist_ok=True)

    for row in misclassified_sorted:
        source = image_dir / row["image_id"]
        target_name = (
            f"true_{row['true_label']}_pred_{row['pred_label']}_"
            f"conf_{row['confidence']:.3f}_{row['image_id']}"
        )
        shutil.copy2(source, misclassified_dir / target_name)

    for row in low_confidence:
        source = image_dir / row["image_id"]
        target_name = (
            f"true_{row['true_label']}_pred_{row['pred_label']}_"
            f"conf_{row['confidence']:.3f}_{row['image_id']}"
        )
        shutil.copy2(source, low_conf_dir / target_name)

    logger.log(f"Accuracy: {metric_accuracy:.4f}")
    logger.log(f"Macro-F1: {metric_macro_f1:.4f}")
    logger.log(f"Saved evaluation bundle: {model_output_dir.as_posix()}")

    return metrics_payload


def write_markdown_summary(
    path: Path,
    label_map: Dict[int, str],
    baseline_metrics: Dict,
    transfer_metrics: Dict,
) -> None:
    lines = []
    lines.append("# Deliverable 3 - Evaluation and Error Analysis")
    lines.append("")
    lines.append("## Test Methodology")
    lines.append("- Fixed train/validation/test split generated in Deliverable 1")
    lines.append("- Metrics computed on held-out test set only")
    lines.append("- Both baseline and transfer models evaluated consistently")
    lines.append("")
    lines.append("## Model Comparison")
    lines.append(f"- Baseline accuracy: **{baseline_metrics['accuracy']:.4f}**")
    lines.append(f"- Baseline macro-F1: **{baseline_metrics['macro_f1']:.4f}**")
    lines.append(f"- Transfer accuracy: **{transfer_metrics['accuracy']:.4f}**")
    lines.append(f"- Transfer macro-F1: **{transfer_metrics['macro_f1']:.4f}**")
    lines.append("")
    lines.append("## Top Failure Modes (Transfer Model)")
    for pair in transfer_metrics["top_failure_pairs"]:
        true_name = label_map[pair["true_label"]]
        pred_name = label_map[pair["pred_label"]]
        lines.append(
            f"- True `{pair['true_label']} ({true_name})` predicted as "
            f"`{pair['pred_label']} ({pred_name})`: {pair['count']} samples"
        )
    lines.append("")
    lines.append("## Improvement Hypotheses")
    lines.append("- Add class-specific augmentations for confusion-prone disease pairs")
    lines.append("- Use label smoothing or focal loss to reduce overconfident mistakes")
    lines.append("- Curate hard examples from qualitative folders and perform targeted retraining")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ************************************* CLI ************************************* #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deliverable 3: evaluation and error analysis.")
    parser.add_argument("--test-split", type=Path, default=Path("data/processed/splits/test_split.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/raw/train_images"))
    parser.add_argument("--label-map", type=Path, default=Path("data/raw/label_num_to_disease_map.json"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("artifacts/training/models/baseline_best.pt"))
    parser.add_argument("--transfer-checkpoint", type=Path, default=Path("artifacts/training/models/transfer_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--run-log-path", type=Path, default=Path("artifacts/reports/deliverable3_terminal_output.txt"))
    parser.add_argument("--report-path", type=Path, default=Path("artifacts/reports/deliverable3_evaluation.md"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


# ************************************* MAIN ************************************* #
def main() -> None:
    args = parse_args()
    logger = RunLogger()

    if not args.baseline_checkpoint.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {args.baseline_checkpoint}")
    if not args.transfer_checkpoint.exists():
        raise FileNotFoundError(f"Missing transfer checkpoint: {args.transfer_checkpoint}")

    label_map_raw = json.loads(args.label_map.read_text(encoding="utf-8"))
    label_map = {int(key): value for key, value in label_map_raw.items()}

    logger.log("Deliverable 3 - Evaluation and Error Analysis")
    logger.log(f"Using test split: {args.test_split.as_posix()}")

    baseline_metrics = run_evaluation(
        model_name="baseline",
        checkpoint_path=args.baseline_checkpoint,
        test_split=args.test_split,
        image_dir=args.image_dir,
        label_map=label_map,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_cpu=args.force_cpu,
        max_test_samples=args.max_test_samples,
        logger=logger,
    )
    transfer_metrics = run_evaluation(
        model_name="transfer",
        checkpoint_path=args.transfer_checkpoint,
        test_split=args.test_split,
        image_dir=args.image_dir,
        label_map=label_map,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_cpu=args.force_cpu,
        max_test_samples=args.max_test_samples,
        logger=logger,
    )

    summary = {
        "baseline": {
            "accuracy": baseline_metrics["accuracy"],
            "macro_f1": baseline_metrics["macro_f1"],
        },
        "transfer": {
            "accuracy": transfer_metrics["accuracy"],
            "macro_f1": transfer_metrics["macro_f1"],
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(
        path=args.report_path,
        label_map=label_map,
        baseline_metrics=baseline_metrics,
        transfer_metrics=transfer_metrics,
    )

    logger.log(" ")
    logger.log("Evaluation complete.")
    logger.log(f"Summary saved: {(args.output_dir / 'summary.json').as_posix()}")
    logger.log(f"Report saved: {args.report_path.as_posix()}")
    logger.log(f"Terminal output saved to: {args.run_log_path.as_posix()}")
    logger.save(args.run_log_path)


if __name__ == "__main__":
    main()

