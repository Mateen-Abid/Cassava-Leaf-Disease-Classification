import io
import json
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms


# ************************************* CONFIG ************************************* #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "inference_config.json"


def load_config() -> Dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


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
    raise ValueError(f"Unsupported model_name: {model_name}")


# ************************************* APP STARTUP ************************************* #
app = FastAPI(title="Cassava Inference API", version="1.0.0")

INFER_STATE = {
    "model": None,
    "label_map": {},
    "preprocessor": None,
    "device": torch.device("cpu"),
    "checkpoint_path": "",
}


@app.on_event("startup")
def startup_load_model() -> None:
    config = load_config()
    checkpoint_path = PROJECT_ROOT / config["checkpoint_path"]
    label_map_path = PROJECT_ROOT / config["label_map_path"]
    force_cpu = bool(config.get("force_cpu", True))
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]
    image_size = int(checkpoint.get("image_size", config.get("image_size", 224)))

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

    INFER_STATE["model"] = model
    INFER_STATE["label_map"] = label_map
    INFER_STATE["preprocessor"] = preprocessor
    INFER_STATE["device"] = device
    INFER_STATE["checkpoint_path"] = checkpoint_path.as_posix()


# ************************************* API ROUTES ************************************* #
@app.get("/health")
def health() -> Dict:
    return {
        "status": "ok",
        "device": str(INFER_STATE["device"]),
        "checkpoint_path": INFER_STATE["checkpoint_path"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    if INFER_STATE["model"] is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {error}") from error

    tensor = INFER_STATE["preprocessor"](image).unsqueeze(0).to(INFER_STATE["device"])
    with torch.no_grad():
        logits = INFER_STATE["model"](tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        confidence, pred_idx = torch.max(probs, dim=0)

    predicted_id = int(pred_idx.item())
    top_values, top_indices = torch.topk(probs, k=min(3, len(probs)))
    top3 = []
    for p, idx in zip(top_values.tolist(), top_indices.tolist()):
        top3.append(
            {
                "class_id": int(idx),
                "class_name": INFER_STATE["label_map"].get(int(idx), "Unknown"),
                "probability": float(p),
            }
        )

    return {
        "predicted_class_id": predicted_id,
        "predicted_class_name": INFER_STATE["label_map"].get(predicted_id, "Unknown"),
        "confidence": float(confidence.item()),
        "top3": top3,
    }

