# Step 1: Problem Framing (Computer Vision Assessment)

## Objective
Build a practical computer vision classification system that demonstrates:
- Data understanding and preprocessing discipline
- Defensible model choices under limited compute
- Clean evaluation and error analysis
- Deployment-ready inference logic

## Chosen Task
Image classification.

Reason:
- Lowest implementation risk for the 72-hour cap while still meeting all required deliverables
- Enables clear side-by-side comparison of baseline vs stronger transfer-learning model
- Leaves enough time for proper evaluation, error analysis, API, and Docker

## Chosen Dataset
Cassava Leaf Disease Classification (public dataset; real-world agriculture images, multi-class labels).

Why this dataset:
- Non-trivial and domain-relevant (not MNIST/CIFAR-10)
- Realistic image variability (lighting, leaf orientation, background clutter)
- Supports practical discussions on class imbalance and confusing classes
- Suitable for laptop/free-tier compute with transfer learning

## Scope and Constraints
- No high-end GPU assumptions
- No benchmark chasing; prioritize methodology and engineering quality
- Deliver a prototype that is reproducible and production-aware, not production-scale

## Planned Modeling Strategy
Two-model requirement will be satisfied by:
1. Baseline: small CNN trained from scratch
2. Stronger model: fine-tuned pretrained backbone (for example, ResNet18 or EfficientNet-B0)

Compute-aware choices:
- Moderate image size (for example, 224x224)
- Early stopping / best-checkpoint saving
- Limited but meaningful hyperparameter tuning (learning rate, weight decay, augmentation strength)

## Evaluation Plan
- Proper train/validation/test split
- Primary metrics: accuracy, macro-F1, per-class precision/recall/F1
- Confusion matrix and failure case visualization
- Error analysis notes with concrete improvement hypotheses

## Inference and Deployment Plan
- Inference script with explicit model loading and preprocessing
- FastAPI endpoint for prediction
- Dockerized inference service (optional requirement included)

## Engineering Plan
- Modular project structure (data, training, eval, inference separated)
- Config-driven runs (paths, model selection, hyperparameters)
- Logging and basic error handling
- README with decisions, results, tradeoffs, and "what to improve with more compute"

## Step 1 Exit Criteria
Step 1 is complete when:
- Task and dataset are finalized
- Decision rationale is documented
- Evaluation and deployment direction is locked

Status: Complete.
