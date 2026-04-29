import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms


# ************************************* MODEL DEFINITIONS ************************************* #
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
        return self.classifier(self.features(x))


def build_transfer_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(p=0.20),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "transfer":
        return build_transfer_model(num_classes=num_classes)
    raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")


# ************************************* INFERENCE CORE ************************************* #
def load_inference_bundle(
    checkpoint_path: Path,
    label_map_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[int, str], transforms.Compose]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]
    image_size = int(checkpoint.get("image_size", 224))

    model = build_model(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    label_map_raw = json.loads(label_map_path.read_text(encoding="utf-8"))
    label_map = {int(k): v for k, v in label_map_raw.items()}

    preprocessor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, label_map, preprocessor


def predict_single_image(
    image_path: Path,
    model: nn.Module,
    preprocessor: transforms.Compose,
    label_map: Dict[int, str],
    device: torch.device,
) -> Dict:
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")

    image_tensor = preprocessor(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        confidence, pred_idx = torch.max(probs, dim=0)

    predicted_class = int(pred_idx.item())
    result = {
        "image_path": image_path.as_posix(),
        "predicted_class_id": predicted_class,
        "predicted_class_name": label_map.get(predicted_class, "Unknown"),
        "confidence": float(confidence.item()),
        "top3": [],
    }

    top_values, top_indices = torch.topk(probs, k=min(3, len(probs)))
    for prob, idx in zip(top_values.tolist(), top_indices.tolist()):
        result["top3"].append(
            {
                "class_id": int(idx),
                "class_name": label_map.get(int(idx), "Unknown"),
                "probability": float(prob),
            }
        )
    return result


# ************************************* CLI ************************************* #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deliverable 4: single-image inference script.")
    parser.add_argument("--image", type=Path, required=True, help="Path to image file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/training/models/transfer_best.pt"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("data/raw/label_num_to_disease_map.json"),
        help="Class id to label map path.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/inference/last_prediction.json"))
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image does not exist: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    model, label_map, preprocessor = load_inference_bundle(
        checkpoint_path=args.checkpoint,
        label_map_path=args.label_map,
        device=device,
    )
    prediction = predict_single_image(
        image_path=args.image,
        model=model,
        preprocessor=preprocessor,
        label_map=label_map,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prediction, indent=2), encoding="utf-8")

    print("Inference complete.")
    print(json.dumps(prediction, indent=2))
    print(f"Saved output: {args.output.as_posix()}")


if __name__ == "__main__":
    main()

