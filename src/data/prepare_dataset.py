import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageEnhance, ImageOps


# ************************************* DATA MODELS ************************************* #
@dataclass
class Sample:
    """Single labeled image entry from train.csv."""

    image_id: str
    label: int


# ************************************* TERMINAL LOGGER ************************************* #
class RunLogger:
    """Prints to terminal and stores the same lines in a log file."""

    def __init__(self) -> None:
        self._lines: List[str] = []

    def log(self, message: str = "") -> None:
        print(message)
        self._lines.append(message)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")


# ************************************* DATA LOADING ************************************* #
def read_label_map(path: Path) -> Dict[int, str]:
    """Reads class-id to class-name mapping from JSON."""

    with path.open("r", encoding="utf-8") as file:
        raw_map = json.load(file)
    return {int(key): value for key, value in raw_map.items()}


def read_samples(path: Path) -> List[Sample]:
    """Reads train.csv into a typed list of Sample entries."""

    samples: List[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            samples.append(Sample(image_id=row["image_id"], label=int(row["label"])))
    return samples


# ************************************* DATA QUALITY CHECKS ************************************* #
def verify_image_files(samples: List[Sample], image_dir: Path) -> Tuple[int, int]:
    """Checks missing files and unreadable/corrupt images."""

    missing_files = 0
    unreadable_images = 0

    for sample in samples:
        image_path = image_dir / sample.image_id
        if not image_path.exists():
            missing_files += 1
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            unreadable_images += 1

    return missing_files, unreadable_images


def image_resolution_stats(samples: List[Sample], image_dir: Path) -> Counter:
    """Collects image resolution counts to understand raw data consistency."""

    resolution_counter: Counter = Counter()
    for sample in samples:
        image_path = image_dir / sample.image_id
        with Image.open(image_path) as image:
            resolution_counter[image.size] += 1
    return resolution_counter


# ************************************* SPLIT + CLASS STATS ************************************* #
def stratified_split(
    samples: List[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Creates train/val/test split while preserving per-class ratios."""

    rng = random.Random(seed)
    by_label: Dict[int, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    test_samples: List[Sample] = []

    for _, label_samples in by_label.items():
        rng.shuffle(label_samples)

        total_count = len(label_samples)
        test_count = max(1, round(total_count * test_ratio))
        val_count = max(1, round(total_count * val_ratio))

        # Guard to ensure we always have at least one train example per class.
        if test_count + val_count >= total_count:
            val_count = max(1, total_count - test_count - 1)

        test_part = label_samples[:test_count]
        val_part = label_samples[test_count : test_count + val_count]
        train_part = label_samples[test_count + val_count :]

        test_samples.extend(test_part)
        val_samples.extend(val_part)
        train_samples.extend(train_part)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_samples, val_samples, test_samples


def write_samples(path: Path, samples: List[Sample]) -> None:
    """Writes split CSV with schema: image_id,label."""

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "label"])
        for sample in samples:
            writer.writerow([sample.image_id, sample.label])


def class_distribution(samples: List[Sample]) -> Dict[int, int]:
    """Returns class->count dictionary sorted by class id."""

    counts: Counter = Counter(sample.label for sample in samples)
    return dict(sorted(counts.items()))


def class_weights_from_counts(counts: Dict[int, int]) -> Dict[int, float]:
    """Builds inverse-frequency class weights for imbalanced training."""

    total = sum(counts.values())
    num_classes = len(counts)
    # Common inverse-frequency style weighting for imbalanced training.
    raw_weights = {
        class_id: total / (num_classes * count)
        for class_id, count in counts.items()
    }
    return dict(sorted(raw_weights.items()))


def find_conflicting_duplicates(samples: List[Sample], image_dir: Path) -> List[Dict]:
    """Detects exact duplicate images assigned to different labels."""

    hash_to_labels: Dict[str, set] = defaultdict(set)
    hash_to_images: Dict[str, List[str]] = defaultdict(list)

    for sample in samples:
        image_path = image_dir / sample.image_id
        with image_path.open("rb") as file:
            image_hash = hashlib.md5(file.read()).hexdigest()
        hash_to_labels[image_hash].add(sample.label)
        hash_to_images[image_hash].append(sample.image_id)

    conflicts: List[Dict] = []
    for image_hash, labels in hash_to_labels.items():
        if len(labels) > 1:
            conflicts.append(
                {
                    "hash": image_hash,
                    "labels": sorted(list(labels)),
                    "image_ids": sorted(hash_to_images[image_hash]),
                }
            )
    return conflicts


# ************************************* PREPROCESSING + AUGMENTATION ************************************* #
def build_preprocessing_config(image_size: int) -> Dict:
    """Returns preprocessing plan to be used later in training/inference."""

    return {
        "resize": [image_size, image_size],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }


def build_augmentation_config() -> Dict:
    """Returns train-time augmentation plan used in this assessment."""

    return {
        "horizontal_flip_probability": 0.50,
        "vertical_flip_probability": 0.20,
        "rotation_degrees": 15,
        "brightness_factor_range": [0.90, 1.10],
        "contrast_factor_range": [0.90, 1.10],
        "color_factor_range": [0.90, 1.10],
    }


def apply_augmentation_preview(
    image: Image.Image,
    rng: random.Random,
    augmentation_config: Dict,
) -> Image.Image:
    """Applies deterministic preview augmentation (for report visuals only)."""

    output = image.copy()

    if rng.random() < augmentation_config["horizontal_flip_probability"]:
        output = ImageOps.mirror(output)
    if rng.random() < augmentation_config["vertical_flip_probability"]:
        output = ImageOps.flip(output)

    angle = rng.uniform(
        -augmentation_config["rotation_degrees"],
        augmentation_config["rotation_degrees"],
    )
    output = output.rotate(angle, resample=Image.Resampling.BILINEAR)

    bright_factor = rng.uniform(*augmentation_config["brightness_factor_range"])
    contrast_factor = rng.uniform(*augmentation_config["contrast_factor_range"])
    color_factor = rng.uniform(*augmentation_config["color_factor_range"])

    output = ImageEnhance.Brightness(output).enhance(bright_factor)
    output = ImageEnhance.Contrast(output).enhance(contrast_factor)
    output = ImageEnhance.Color(output).enhance(color_factor)
    return output


def save_augmentation_preview(
    samples: List[Sample],
    image_dir: Path,
    output_dir: Path,
    image_size: int,
    seed: int,
    max_examples: int,
) -> int:
    """Saves original+augmented sample pairs for quick qualitative inspection."""

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_rng = random.Random(seed + 99)
    augmentation_config = build_augmentation_config()

    by_label: Dict[int, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)

    saved_count = 0
    for label in sorted(by_label.keys()):
        if saved_count >= max_examples:
            break
        candidate = by_label[label][0]
        image_path = image_dir / candidate.image_id

        with Image.open(image_path) as image:
            base = image.convert("RGB").resize((image_size, image_size))
            augmented = apply_augmentation_preview(base, preview_rng, augmentation_config)

            base.save(output_dir / f"class_{label}_original_{candidate.image_id}", quality=95)
            augmented.save(
                output_dir / f"class_{label}_augmented_{candidate.image_id}",
                quality=95,
            )
            saved_count += 1

    return saved_count


# ************************************* REPORTING ************************************* #
def write_markdown_report(
    report_path: Path,
    dataset_size: int,
    label_map: Dict[int, str],
    full_distribution: Dict[int, int],
    train_distribution: Dict[int, int],
    val_distribution: Dict[int, int],
    test_distribution: Dict[int, int],
    top_resolution: Tuple[int, int],
    imbalance_ratio: float,
    missing_files: int,
    unreadable_images: int,
    conflicting_duplicates_count: int,
) -> None:
    """Writes a readable markdown summary for evaluator review."""

    lines = []
    lines.append("# Data Understanding and Preparation Report")
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append(f"- Total labeled training images: **{dataset_size}**")
    lines.append("- Label format: integer class IDs in `train.csv` (`image_id`, `label`)")
    lines.append(f"- Main image resolution observed: **{top_resolution[0]}x{top_resolution[1]}**")
    lines.append("")
    lines.append("## Label Map")
    for class_id, class_name in label_map.items():
        lines.append(f"- {class_id}: {class_name}")
    lines.append("")
    lines.append("## Class Distribution (Original)")
    for class_id, count in full_distribution.items():
        pct = (count / dataset_size) * 100
        lines.append(f"- Class {class_id}: {count} ({pct:.2f}%)")
    lines.append("")
    lines.append("## Split Summary")
    lines.append("- Stratified split used to preserve class ratios across subsets.")
    lines.append(f"- Train: {sum(train_distribution.values())}")
    lines.append(f"- Validation: {sum(val_distribution.values())}")
    lines.append(f"- Test: {sum(test_distribution.values())}")
    lines.append("")
    lines.append("## Data Quality Checks")
    lines.append(f"- Missing image files referenced in CSV: **{missing_files}**")
    lines.append(f"- Unreadable/corrupted images: **{unreadable_images}**")
    lines.append(
        f"- Exact-duplicate images with conflicting labels: **{conflicting_duplicates_count}**"
    )
    lines.append("")
    lines.append("## Handling Class Imbalance")
    lines.append(
        f"- Imbalance ratio (largest class / smallest class): **{imbalance_ratio:.2f}x**"
    )
    lines.append("- Planned mitigation strategy:")
    lines.append("  - Use class-weighted cross-entropy loss")
    lines.append("  - Track macro-F1 to avoid majority-class bias")
    lines.append("  - Consider weighted sampling if baseline underperforms on minority classes")
    lines.append("")
    lines.append("## Preprocessing and Augmentation Plan")
    lines.append("- Preprocessing (all splits): resize to 224x224 and normalize with ImageNet mean/std")
    lines.append("- Augmentation (train only): random horizontal/vertical flips, mild rotation, color jitter")
    lines.append("")
    lines.append("## Dataset Limitations")
    lines.append("- Strong class imbalance toward class 3 may bias models if unaddressed")
    lines.append("- Potential label noise cannot be fully eliminated without manual review")
    lines.append("- Field images can contain lighting/background variability that affects predictions")
    lines.append("")
    lines.append("## Improvement Hypotheses")
    lines.append("- Hard-example mining for minority classes")
    lines.append("- Targeted augmentations for confusion-prone classes")
    lines.append("- Label audit on persistent failure cases after first training run")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ************************************* CLI ************************************* #
def parse_args() -> argparse.Namespace:
    """Parses runtime arguments for the Deliverable 1 pipeline."""

    parser = argparse.ArgumentParser(description="Prepare and profile cassava dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-path", type=Path, default=Path("artifacts/reports/data_understanding.md"))
    parser.add_argument(
        "--run-log-path",
        type=Path,
        default=Path("artifacts/reports/deliverable1_terminal_output.txt"),
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("artifacts/reports/augmentation_preview"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check-noise",
        action="store_true",
        help="Run exact-duplicate conflict check (slower, but useful for data quality reporting).",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip markdown report generation and print results to terminal only.",
    )
    return parser.parse_args()


# ************************************* MAIN PIPELINE ************************************* #
def main() -> None:
    """Runs Deliverable 1 end-to-end in one command."""

    args = parse_args()
    logger = RunLogger()

    train_csv = args.raw_dir / "train.csv"
    label_map_path = args.raw_dir / "label_num_to_disease_map.json"
    train_image_dir = args.raw_dir / "train_images"

    label_map = read_label_map(label_map_path)
    samples = read_samples(train_csv)

    # 1) Inspect dataset integrity and basic image characteristics.
    missing_files, unreadable_images = verify_image_files(samples, train_image_dir)
    resolution_counter = image_resolution_stats(samples, train_image_dir)
    top_resolution = resolution_counter.most_common(1)[0][0]

    # 2) Create reproducible stratified splits for fair evaluation later.
    full_distribution = class_distribution(samples)
    train_samples, val_samples, test_samples = stratified_split(
        samples=samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_distribution = class_distribution(train_samples)
    val_distribution = class_distribution(val_samples)
    test_distribution = class_distribution(test_samples)

    class_weights = class_weights_from_counts(train_distribution)
    preprocessing_config = build_preprocessing_config(args.image_size)
    augmentation_config = build_augmentation_config()

    # 3) Persist split files (single source of truth for next deliverables).
    split_dir = args.processed_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    write_samples(split_dir / "train_split.csv", train_samples)
    write_samples(split_dir / "val_split.csv", val_samples)
    write_samples(split_dir / "test_split.csv", test_samples)

    max_count = max(full_distribution.values())
    min_count = min(full_distribution.values())
    imbalance_ratio = max_count / min_count

    # 4) Optional noisy-label heuristic: same file content with different labels.
    conflicts = []
    if args.check_noise:
        conflicts = find_conflicting_duplicates(samples, train_image_dir)

    # 5) Save visual augmentation examples to support qualitative analysis.
    preview_saved_count = save_augmentation_preview(
        samples=train_samples,
        image_dir=train_image_dir,
        output_dir=args.preview_dir,
        image_size=args.image_size,
        seed=args.seed,
        max_examples=args.preview_count,
    )

    summary = {
        "dataset_size": len(samples),
        "label_map": label_map,
        "full_distribution": full_distribution,
        "split_distribution": {
            "train": train_distribution,
            "val": val_distribution,
            "test": test_distribution,
        },
        "class_weights": class_weights,
        "image_resolution": {
            "most_common": {"width": top_resolution[0], "height": top_resolution[1]},
            "unique_resolution_count": len(resolution_counter),
        },
        "quality_checks": {
            "missing_files": missing_files,
            "unreadable_images": unreadable_images,
            "conflicting_exact_duplicates": len(conflicts),
        },
        "preprocessing": preprocessing_config,
        "augmentation": augmentation_config,
        "split_config": {
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        },
        "preview_examples_saved": preview_saved_count,
    }

    # 6) Persist machine-readable artifacts used by downstream scripts.
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    (args.processed_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if args.check_noise:
        (args.processed_dir / "noise_conflicts.json").write_text(
            json.dumps(conflicts[:200], indent=2),
            encoding="utf-8",
        )

    # 7) Optional human-readable markdown report.
    if not args.skip_report:
        write_markdown_report(
            report_path=args.report_path,
            dataset_size=len(samples),
            label_map=label_map,
            full_distribution=full_distribution,
            train_distribution=train_distribution,
            val_distribution=val_distribution,
            test_distribution=test_distribution,
            top_resolution=top_resolution,
            imbalance_ratio=imbalance_ratio,
            missing_files=missing_files,
            unreadable_images=unreadable_images,
            conflicting_duplicates_count=len(conflicts),
        )

    # 8) Print terminal output (also saved to run log file).
    logger.log("Data preparation complete.")
    logger.log("")
    logger.log("Dataset Overview")
    logger.log(f"- Total labeled images: {len(samples)}")
    logger.log("- Label format: train.csv columns = image_id, label")
    logger.log(f"- Primary resolution: {top_resolution[0]}x{top_resolution[1]}")
    logger.log("")
    logger.log("Class Distribution")
    for class_id, count in full_distribution.items():
        pct = (count / len(samples)) * 100
        logger.log(f"- Class {class_id}: {count} ({pct:.2f}%)")
    logger.log("")
    logger.log("Split Sizes")
    logger.log(f"- Train: {len(train_samples)}")
    logger.log(f"- Validation: {len(val_samples)}")
    logger.log(f"- Test: {len(test_samples)}")
    logger.log("")
    logger.log("Quality Checks")
    logger.log(f"- Missing files: {missing_files}")
    logger.log(f"- Unreadable images: {unreadable_images}")
    logger.log(f"- Conflicting exact duplicates: {len(conflicts)}")
    logger.log("")
    logger.log(f"Imbalance ratio (max/min class count): {imbalance_ratio:.2f}x")
    logger.log("")
    logger.log("Preprocessing Config")
    logger.log(f"- Resize: {preprocessing_config['resize'][0]}x{preprocessing_config['resize'][1]}")
    logger.log(f"- Normalize mean: {preprocessing_config['normalize_mean']}")
    logger.log(f"- Normalize std: {preprocessing_config['normalize_std']}")
    logger.log("")
    logger.log("Augmentation Config")
    logger.log(
        f"- Horizontal flip probability: {augmentation_config['horizontal_flip_probability']}"
    )
    logger.log(
        f"- Vertical flip probability: {augmentation_config['vertical_flip_probability']}"
    )
    logger.log(f"- Rotation degrees: +/-{augmentation_config['rotation_degrees']}")
    logger.log(
        f"- Brightness factor range: {augmentation_config['brightness_factor_range']}"
    )
    logger.log(f"- Contrast factor range: {augmentation_config['contrast_factor_range']}")
    logger.log(f"- Color factor range: {augmentation_config['color_factor_range']}")
    logger.log("")
    logger.log("Class Weights (for weighted loss)")
    for class_id, weight in class_weights.items():
        logger.log(f"- Class {class_id}: {weight:.4f}")
    logger.log("")
    logger.log(f"Summary written to: {(args.processed_dir / 'data_summary.json').as_posix()}")
    logger.log(f"Splits written to: {split_dir.as_posix()}")
    logger.log(f"Augmentation preview saved to: {args.preview_dir.as_posix()}")
    if args.skip_report:
        logger.log("Report generation skipped (--skip-report enabled)")
    else:
        logger.log(f"Report written to: {args.report_path.as_posix()}")

    logger.log(f"Terminal output saved to: {args.run_log_path.as_posix()}")
    logger.save(args.run_log_path)


if __name__ == "__main__":
    main()

