# Technical Report: Cassava Leaf Disease Classification

**Project type:** Job interview technical assessment (Computer Vision & ML Engineering)

**Repository:** https://github.com/Mateen-Abid/Cassava-Leaf-Disease-Classification

**Date:** April 2026

---

## 1. Executive Summary

This report describes an end‑to‑end solution for cassava leaf image classification using a public cassava dataset. The work follows the assessment outline: prepare and understand data, train two models (a simple baseline and a stronger pretrained model), evaluate on a held‑out test set with error analysis, and ship a clean inference path (command‑line script, REST API, and Docker for the API).

Hardware used for training: **AMD Radeon RX 6550M** with **PyTorch DirectML** (Windows does not use NVIDIA CUDA for this GPU).

Main outcome: the transfer model (fine‑tuned **ResNet‑18**) clearly beats the custom baseline on validation macro‑F1 and on held‑out test macro‑F1 and accuracy. Inference was demonstrated on a sample image with believable probabilities for the top classes.

---

## 2. Problem and Dataset

### 2.1 Task

**Task:** Multi‑class image classification of cassava leaves into five classes (four disease types plus healthy).

**Data source:** Kaggle competition dataset *Cassava Leaf Disease Classification* (images plus `train.csv` with `image_id` and integer `label`).

### 2.2 Basic facts (Deliverable 1)

| Item | Value |
|------|--------|
| Number of labeled training images | 21,397 |
| Number of classes | 5 |
| Image resolution (as stored) | 800 × 600 |
| Class imbalance (max class count ÷ min) | 12.10× |

**Class distribution (original training split):**

| Class ID | Short name | Count | Share |
|----------|------------|-------|-------|
| 0 | Cassava Bacterial Blight (CBB) | 1,087 | 5.08% |
| 1 | Cassava Brown Streak Disease (CBSD) | 2,189 | 10.23% |
| 2 | Cassava Green Mottle (CGM) | 2,386 | 11.15% |
| 3 | Cassava Mosaic Disease (CMD) | 13,158 | 61.49% |
| 4 | Healthy | 2,577 | 12.04% |

Class **3** (CMD) is the majority class. Any accuracy‑only story would hide poor performance on rare classes, so the project reports **macro‑F1** alongside accuracy.

### 2.3 Data quality checks

- Missing files listed in the CSV: **0**
- Unreadable or corrupt images: **0**
- Exact duplicate file content with **conflicting** labels: **0**

### 2.4 Train / validation / test split

Splits were created with a **stratified** method so each subset keeps similar class ratios.

| Subset | Row count |
|--------|-----------|
| Train | 14,977 |
| Validation | 3,210 |
| Test | 3,210 |

### 2.5 Preprocessing and augmentation (concept)

- **Preprocessing (all splits):** resize to model input size, **ImageNet** mean and standard deviation for normalization (aligned with ResNet pretraining).
- **Augmentation (training only):** flips, light rotation, light color jitter to mimic field variation without destroying leaf shape.
- **Imbalance handling in training:** class‑weighted loss and attention to macro‑F1; optional weighted sampling if a future model still fails on small classes.

### 2.6 Limitations (honest list)

- Strong imbalance toward CMD can still bias the model if metrics are not read carefully.
- Some field images are hard even for humans; labels may not be perfect for every image.
- Lighting and background vary; the model can confuse visually similar classes (see failure analysis in Section 5).

---

## 3. Models and Training (Deliverable 2)

### 3.1 What we trained and why

**Model A — Baseline CNN (`BaselineCNN`)**  
A small convolutional network trained **from scratch**. Purpose: meet the “simple baseline” requirement and give a fair lower bound without pretrained weights.

**Model B — ResNet‑18 with transfer learning (`transfer`)**  
**ResNet‑18** with **ImageNet** pretrained weights. Only the last residual block and the classifier head were tuned in this project to save time and GPU memory while still getting strong features. This is a standard choice for limited data and laptop‑class hardware.

**Why transfer learning here:** The dataset is imbalanced and limited compared to training a huge network from zero. Pretrained filters help on texture and shape under typical compute limits.

### 3.2 Training setup used for the reported run

- **Device:** GPU via **DirectML** (PyTorch shows this as `privateuseone:0`; the machine reports **AMD Radeon RX 6550M**).
- **Fast preset (`--fast`):** 160×160 inputs, batch size 48, 5 epochs after a short hyperparameter search (2 learning‑rate trials × 1 epoch each on validation).
- **Loss:** Cross‑entropy with **class weights** enabled.
- **Approximate wall time:** about **11 minutes** per model on this GPU (baseline ~11.1 min, transfer ~10.9 min).

### 3.3 Validation results (best checkpoint per model)

| Model | Best validation macro‑F1 | Notes |
|--------|----------------------------|--------|
| Baseline CNN | **0.4284** | Best epoch macro‑F1 across 5 epochs was at epoch 3 (0.4284 logged as best saved). |
| ResNet‑18 transfer | **0.5611** | Stronger representation; macro‑F1 **+0.1326** over baseline. |

**Takeaway:** Transfer learning delivers a solid gain within the same training budget.

---

## 4. Test Evaluation and Error Analysis (Deliverable 3)

Evaluation used the **held‑out test CSV** produced in Deliverable 1 (not validation). Both models ran on the **same** test loader for fairness. For this run the evaluation script used **CPU** (`--force-cpu`), which is fine for measuring accuracy; training time is unchanged.

### 4.1 Headline test metrics

| Model | Accuracy | Macro‑F1 |
|--------|-----------|----------|
| Baseline | 0.5813 | **0.4123** |
| Transfer | **0.6165** | **0.5050** |

The transfer model improves **test accuracy by +3.52 points** (absolute) and **macro‑F1 by +0.0927** versus the baseline. Macro‑F1 matters here because minority classes stay visible in the score.

### 4.2 Where the transfer model fails most often

From the confusion matrix on the transfer model (top swapped label pairs):

1. **True CMD (3) predicted as Healthy (4)** — 285 images  
2. **True CMD (3) predicted as CGM (2)** — 230  
3. **True CGM (2) predicted as Healthy (4)** — 115  
4. **True CMD (3) predicted as CBSD (1)** — 105  
5. **True CBSD (1) predicted as Healthy (4)** — 84  

These patterns suggest **misleading green leaf appearance**, **overlap between mosaic and mottle patterns**, and **healthy vs mild disease** confusion. That matches real agricultural imaging challenges.

### 4.3 Qualitative review

Misclassified examples were copied into an `artifacts/.../qualitative/misclassified` folder for manual review. Many show **low model confidence** (example filenames include confidences in the 0.27–0.53 range), which supports the idea of hard examples rather than a single silly bug.

### 4.4 Suggested next steps (not implemented in full)

- Targeted augmentations or small expert rules for the worst pairs (CMD vs healthy, CGM vs healthy).
- Focal loss or label smoothing if overconfidence shows up in calibration plots.
- Review a small set of persistent errors with a domain expert if labels are uncertain.

---

## 5. Inference and Deployment (Deliverable 4)

### 5.1 Command‑line inference

The script loads the saved transfer checkpoint, applies the same preprocessing as training, and outputs class id, name, confidence, and **top‑3** probabilities.

**Example run (dashboard upload):**

- **Predicted class:** 1 — Cassava Brown Streak Disease (CBSD)  
- **Confidence:** ~0.608  
- **Runner‑up classes:** 0 (CBB) ~0.355, 4 (Healthy) ~0.019  

Output is also written to `artifacts/inference/last_prediction.json`.

*Note:* A PyTorch `FutureWarning` about `torch.load` and `weights_only` may appear; it does not change the prediction. Production code can move to `weights_only=True` once checkpoint format is adjusted.

### 5.2 REST API

A **FastAPI** app (`src/infer/api.py`) exposes `POST /predict` (multipart file upload). Local run:

```text
uvicorn src.infer.api:app --host 0.0.0.0 --port 8000
```

Configuration (checkpoint path, label map) lives in `configs/inference_config.json`.

### 5.3 Docker

`Dockerfile` builds a small image that runs the API with `uvicorn`. Build and run:

```text
docker build -t cassava-inference .
docker run --rm -p 8000:8000 cassava-inference
```

The default image uses **CPU** PyTorch wheels for portability. A cloud GPU deployment would switch base image and install a CUDA build of PyTorch; that is outside this assessment scope.

---

## 6. Engineering Quality (Deliverable 5)

The code is split into clear modules instead of one giant notebook:

| Area | Path |
|------|------|
| Data prep | `src/data/prepare_dataset.py` |
| Training | `src/train/train_models.py` |
| Evaluation | `src/eval/evaluate_models.py` |
| Inference CLI | `src/infer/predict_image.py` |
| Inference API | `src/infer/api.py` |
| Optional pipeline | `src/run_deliverables.py` |
| Dashboard | `ui/dashboard.py` |

Parameters such as paths and inference settings are configurable (especially `configs/inference_config.json`). Generated metrics, logs, and checkpoints sit under `artifacts/` for traceability.

Full setup and commands are in the repository **`README.md`**.

---

## 7. Environment and Reproducibility

- **Python 3.12** was used on the author machine (see paths in logs).
- **Dependencies:** see `requirements.txt` in the repo root.
- **AMD GPU on Windows:** install `pip install torch-directml` after base requirements when you want GPU training; see **`README.md`** for details and verification commands.

Anyone cloning the repo should:

1. Create a virtual environment.  
2. `pip install -r requirements.txt`  
3. Add the Kaggle data under `data/raw/` as described in README.  
4. Run Deliverable 1, then training, then evaluation, following README.

---

## 8. Repository and Submission

- **GitHub URL:** https://github.com/Mateen-Abid/Cassava-Leaf-Disease-Classification  
- **Branch / tag (optional):** [e.g. `main`, `submission`]

Interview follow‑up can reference this document, README, and saved metrics under `artifacts/training/` and `artifacts/evaluation/`.

---

## 9. Closing remarks

This project meets the assessment goals: disciplined data work, two defensible models with a clear winner, honest test metrics with error analysis, and a deployment‑shaped inference layer. The strongest honest message for an interview is **not** raw accuracy alone but the **tradeoffs** (imbalance, transfer learning, macro‑F1, failure modes, and CPU vs GPU paths on real hardware).

---

*End of report.*
