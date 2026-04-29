import argparse
import csv
import json
import math
import os
import random
import time
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


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float


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


def resolve_training_device(logger: RunLogger, device_choice: str, force_cpu: bool) -> torch.device:
    if force_cpu:
        logger.log("Using CPU (--force-cpu).")
        return torch.device("cpu")

    if device_choice == "cpu":
        logger.log("Using CPU (--device cpu).")
        return torch.device("cpu")

    if device_choice == "cuda":
        if torch.cuda.is_available():
            logger.log("Using NVIDIA CUDA GPU.")
            return torch.device("cuda")
        raise SystemExit(
            "--device cuda was requested, but CUDA is not available. "
            "Use --device cpu, --device auto (CPU/DirectML), or fix your NVIDIA CUDA install."
        )

    if device_choice == "directml":
        try:
            import torch_directml

            gpu_device = torch_directml.device()
            try:
                name = torch_directml.device_name(0)
            except Exception:
                name = "DirectML GPU"
            logger.log(f"Using AMD/Intel/other GPU via DirectML: {name}")
            return gpu_device
        except Exception as error:
            raise SystemExit(
                "--device directml was requested but DirectML is not usable.\n"
                f"Error: {error}\n"
                "Fix: pip install torch-directml  (AMD Radeon on Windows).\n"
                "See README.md (AMD Radeon / DirectML section) and Microsoft PyTorch + DirectML docs.\n"
                "Or drop back to CPU with --force-cpu instead of pretending you are on GPU."
            ) from error

    # auto
    if torch.cuda.is_available():
        logger.log("Auto device: NVIDIA CUDA.")
        return torch.device("cuda")
    try:
        import torch_directml

        gpu_device = torch_directml.device()
        try:
            name = torch_directml.device_name(0)
        except Exception:
            name = "DirectML GPU"
        logger.log(f"Auto device: GPU via DirectML ({name}).")
        logger.log(
            "AMD Radeon on Windows uses DirectML — install: pip install torch-directml "
            "(use Python 3.10–3.12 per microsoft docs)."
        )
        return gpu_device
    except Exception:
        logger.log("Auto device: CPU (no CUDA or DirectML detected).")
        return torch.device("cpu")


def device_accepts_cuda_pin_memory(device: torch.device) -> bool:
    return getattr(device, "type", None) == "cuda"


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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image_path = self.image_dir / sample.image_id
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
        image_tensor = self.transform(image_rgb)
        return image_tensor, sample.label


# ************************************* TRANSFORMS ************************************* #
def build_train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


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


def build_transfer_model(num_classes: int, logger: RunLogger) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        logger.log("Using pretrained ResNet18 weights for transfer learning.")
    except Exception as error:
        logger.log(f"Could not load pretrained weights ({error}). Falling back to random init.")
        model = models.resnet18(weights=None)

    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze the final residual block + classifier for lightweight fine-tuning.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.20),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_model(model_name: str, num_classes: int, logger: RunLogger) -> nn.Module:
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "transfer":
        return build_transfer_model(num_classes=num_classes, logger=logger)
    raise ValueError(f"Unsupported model name: {model_name}")


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


def macro_f1_from_confusion(confusion: List[List[int]]) -> float:
    f1_values: List[float] = []
    num_classes = len(confusion)
    for class_id in range(num_classes):
        tp = confusion[class_id][class_id]
        fp = sum(confusion[row][class_id] for row in range(num_classes) if row != class_id)
        fn = sum(confusion[class_id][col] for col in range(num_classes) if col != class_id)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
    return sum(f1_values) / len(f1_values)


def accuracy_score(labels: Sequence[int], predictions: Sequence[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(int(y_true == y_pred) for y_true, y_pred in zip(labels, predictions))
    return correct / len(labels)


# ************************************* TRAIN / EVAL ************************************* #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float, List[List[int]]]:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    all_labels: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    confusion = confusion_matrix_from_predictions(
        labels=all_labels,
        predictions=all_predictions,
        num_classes=num_classes,
    )
    val_loss = running_loss / max(sample_count, 1)
    val_accuracy = accuracy_score(all_labels, all_predictions)
    val_macro_f1 = macro_f1_from_confusion(confusion)
    return val_loss, val_accuracy, val_macro_f1, confusion


def build_weight_tensor(train_samples: Sequence[Sample], num_classes: int) -> torch.Tensor:
    class_counts = {class_id: 0 for class_id in range(num_classes)}
    for sample in train_samples:
        class_counts[sample.label] += 1

    total = sum(class_counts.values())
    weights = []
    for class_id in range(num_classes):
        count = class_counts[class_id]
        weight = total / (num_classes * count) if count > 0 else 1.0
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def maybe_limit_dataset(
    dataset: CassavaSplitDataset,
    max_samples: int,
    seed: int,
) -> Dataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected = indices[:max_samples]
    return Subset(dataset, selected)


# ************************************* RUNNER ************************************* #
def apply_fast_training_preset(args: argparse.Namespace) -> None:
    """Use smaller inputs and lighter tuning for ~15-40 min runs on GPU vs hours on CPU."""
    args.image_size = 160
    args.epochs = 5
    args.tune = True
    args.max_tune_trials = 2
    args.tune_epochs = 1
    if getattr(args, "num_workers", 0) == 0:
        args.num_workers = min(4, (os.cpu_count() or 4))


def adjust_batch_size_for_fast_and_device(args: argparse.Namespace, device: torch.device) -> None:
    if not getattr(args, "fast", False):
        return
    cpu_like = getattr(device, "type", None) == "cpu"
    if cpu_like or args.force_cpu:
        args.batch_size = max(14, min(int(args.batch_size), 28))
    else:
        args.batch_size = max(48, int(args.batch_size))


def run_training(
    model_name: str,
    args: argparse.Namespace,
    logger: RunLogger,
    training_device: torch.device,
) -> Dict:
    start_time = time.time()
    num_classes = args.num_classes
    device = training_device

    logger.log(" ")
    logger.log(f"==================== MODEL: {model_name.upper()} ====================")
    logger.log(f"Device: {device}")
    logger.log(f"Image size: {args.image_size}")
    logger.log(f"Batch size: {args.batch_size}")
    logger.log(f"Epochs: {args.epochs}")

    train_dataset = CassavaSplitDataset(
        split_csv_path=args.train_split,
        image_dir=args.image_dir,
        transform=build_train_transforms(args.image_size),
    )
    val_dataset = CassavaSplitDataset(
        split_csv_path=args.val_split,
        image_dir=args.image_dir,
        transform=build_eval_transforms(args.image_size),
    )

    train_dataset_for_run = maybe_limit_dataset(
        dataset=train_dataset,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_dataset_for_run = maybe_limit_dataset(
        dataset=val_dataset,
        max_samples=args.max_val_samples,
        seed=args.seed + 7,
    )

    train_loader = DataLoader(
        train_dataset_for_run,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device_accepts_cuda_pin_memory(device),
    )
    val_loader = DataLoader(
        val_dataset_for_run,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device_accepts_cuda_pin_memory(device),
    )

    logger.log(f"Train samples used: {len(train_dataset_for_run)}")
    logger.log(f"Validation samples used: {len(val_dataset_for_run)}")

    tune_candidates = [(args.learning_rate, args.weight_decay)]
    if args.tune:
        tune_candidates = [
            (1e-3, 1e-4),
            (3e-4, 1e-4),
            (1e-3, 1e-3),
            (3e-4, 1e-3),
        ][: args.max_tune_trials]
        logger.log("Running basic hyperparameter tuning on validation macro-F1.")

    best_lr, best_wd = tune_candidates[0]
    best_tune_score = -math.inf

    for idx, (lr, wd) in enumerate(tune_candidates, start=1):
        model = build_model(model_name=model_name, num_classes=num_classes, logger=logger).to(device)
        class_weights = build_weight_tensor(train_dataset.samples, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights if args.use_class_weights else None)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=wd,
        )

        tune_epochs = args.tune_epochs if args.tune else min(1, args.epochs)
        logger.log(f"Tune trial {idx}/{len(tune_candidates)} -> lr={lr}, wd={wd}, epochs={tune_epochs}")

        for _ in range(tune_epochs):
            _ = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

        _, _, macro_f1, _ = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )
        logger.log(f"  Validation macro-F1: {macro_f1:.4f}")

        if macro_f1 > best_tune_score:
            best_tune_score = macro_f1
            best_lr = lr
            best_wd = wd

    logger.log(f"Selected hyperparameters -> lr={best_lr}, wd={best_wd}")

    model = build_model(model_name=model_name, num_classes=num_classes, logger=logger).to(device)
    class_weights = build_weight_tensor(train_dataset.samples, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights if args.use_class_weights else None)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=best_lr,
        weight_decay=best_wd,
    )

    best_val_f1 = -math.inf
    best_checkpoint = None
    history: List[Dict] = []
    best_confusion: List[List[int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy, val_macro_f1, confusion = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        history_entry = EpochMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            val_macro_f1=val_macro_f1,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": history_entry.train_loss,
                "val_loss": history_entry.val_loss,
                "val_accuracy": history_entry.val_accuracy,
                "val_macro_f1": history_entry.val_macro_f1,
            }
        )

        logger.log(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_confusion = confusion
            best_checkpoint = {
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "image_size": args.image_size,
                "best_val_macro_f1": best_val_f1,
                "selected_lr": best_lr,
                "selected_weight_decay": best_wd,
            }

    elapsed = time.time() - start_time

    model_dir = args.artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"{model_name}_best.pt"
    if best_checkpoint is not None:
        torch.save(best_checkpoint, checkpoint_path)

    metrics_dir = args.artifact_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "model_name": model_name,
        "device": str(device),
        "epochs": args.epochs,
        "selected_hyperparameters": {
            "learning_rate": best_lr,
            "weight_decay": best_wd,
            "used_tuning": args.tune,
            "best_tuning_macro_f1": best_tune_score if args.tune else None,
        },
        "best_val_macro_f1": best_val_f1,
        "best_confusion_matrix": best_confusion,
        "history": history,
        "runtime_seconds": round(elapsed, 2),
    }
    metrics_path = metrics_dir / f"{model_name}_training_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    logger.log(f"Saved best checkpoint: {checkpoint_path.as_posix()}")
    logger.log(f"Saved metrics: {metrics_path.as_posix()}")
    logger.log(f"Runtime: {elapsed / 60:.2f} minutes")

    return metrics_payload


# ************************************* CLI ************************************* #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deliverable 2: model training pipeline.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "transfer", "both"],
        default="both",
        help="Which model to train for Deliverable 2.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "directml"],
        default="auto",
        help="Device: auto picks CUDA then DirectML (AMD Radeon Windows) else CPU.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Preset for ~15-40 min on GPU: 160px inputs, lighter tuning (2x1 epoch), 5 epochs, larger batch on GPU.",
    )
    parser.add_argument("--train-split", type=Path, default=Path("data/processed/splits/train_split.csv"))
    parser.add_argument("--val-split", type=Path, default=Path("data/processed/splits/val_split.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/raw/train_images"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--run-log-path", type=Path, default=Path("artifacts/reports/deliverable2_terminal_output.txt"))
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--tune", action="store_true", help="Run basic hyperparameter tuning.")
    parser.add_argument("--tune-epochs", type=int, default=2)
    parser.add_argument("--max-tune-trials", type=int, default=4)
    parser.add_argument("--use-class-weights", action="store_true", default=True)
    parser.add_argument("--disable-class-weights", dest="use_class_weights", action="store_false")
    return parser.parse_args()


# ************************************* MAIN ************************************* #
def main() -> None:
    args = parse_args()
    logger = RunLogger()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if getattr(args, "fast", False):
        apply_fast_training_preset(args)

    logger.log("Deliverable 2 - Model Selection and Training")
    logger.log("Compute-aware setup: baseline + transfer model, optional basic tuning.")
    logger.log(f"Class-weighted loss enabled: {args.use_class_weights}")
    if getattr(args, "fast", False):
        logger.log(
            "Fast preset enabled (--fast): tuned for GPU (DirectML/CUDA). "
            "AMD Radeon: pip install torch-directml then run with --device auto."
        )

    training_device = resolve_training_device(logger, args.device, args.force_cpu)
    adjust_batch_size_for_fast_and_device(args, training_device)

    model_plan = ["baseline", "transfer"] if args.model == "both" else [args.model]
    results = {}
    for model_name in model_plan:
        metrics = run_training(model_name=model_name, args=args, logger=logger, training_device=training_device)
        results[model_name] = {
            "best_val_macro_f1": metrics["best_val_macro_f1"],
            "selected_hyperparameters": metrics["selected_hyperparameters"],
            "runtime_seconds": metrics["runtime_seconds"],
        }

    summary_path = args.artifact_dir / "metrics" / "deliverable2_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.log(" ")
    logger.log("Final Summary")
    for model_name in model_plan:
        model_result = results[model_name]
        logger.log(
            f"- {model_name}: best_val_macro_f1={model_result['best_val_macro_f1']:.4f}, "
            f"runtime={model_result['runtime_seconds']:.1f}s"
        )
    logger.log(f"Saved summary: {summary_path.as_posix()}")

    logger.log(f"Terminal output saved to: {args.run_log_path.as_posix()}")
    logger.save(args.run_log_path)


if __name__ == "__main__":
    main()

