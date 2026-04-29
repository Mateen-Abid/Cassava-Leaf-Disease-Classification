from pathlib import Path
import json
import re
import subprocess
import sys
import tempfile

import pandas as pd
import streamlit as st


# ************************************* PATHS ************************************* #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "data_summary.json"
RUN_LOG_D1 = PROJECT_ROOT / "artifacts" / "reports" / "deliverable1_terminal_output.txt"
RUN_LOG_D2 = PROJECT_ROOT / "artifacts" / "reports" / "deliverable2_terminal_output.txt"
RUN_LOG_D3 = PROJECT_ROOT / "artifacts" / "reports" / "deliverable3_terminal_output.txt"

REPORT_D1 = PROJECT_ROOT / "artifacts" / "reports" / "data_understanding.md"
REPORT_D3 = PROJECT_ROOT / "artifacts" / "reports" / "deliverable3_evaluation.md"

PREVIEW_DIR = PROJECT_ROOT / "artifacts" / "reports" / "augmentation_preview"
TRAIN_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "train_split.csv"
VAL_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "val_split.csv"
TEST_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "test_split.csv"

TRAINING_SUMMARY = PROJECT_ROOT / "artifacts" / "training" / "metrics" / "deliverable2_summary.json"
BASELINE_TRAINING_METRICS = PROJECT_ROOT / "artifacts" / "training" / "metrics" / "baseline_training_metrics.json"
TRANSFER_TRAINING_METRICS = PROJECT_ROOT / "artifacts" / "training" / "metrics" / "transfer_training_metrics.json"

EVAL_SUMMARY = PROJECT_ROOT / "artifacts" / "evaluation" / "summary.json"
BASELINE_EVAL_METRICS = PROJECT_ROOT / "artifacts" / "evaluation" / "baseline" / "metrics.json"
TRANSFER_EVAL_METRICS = PROJECT_ROOT / "artifacts" / "evaluation" / "transfer" / "metrics.json"
BASELINE_QUAL_MIS = PROJECT_ROOT / "artifacts" / "evaluation" / "baseline" / "qualitative" / "misclassified"
TRANSFER_QUAL_MIS = PROJECT_ROOT / "artifacts" / "evaluation" / "transfer" / "qualitative" / "misclassified"

INFER_LAST_OUTPUT = PROJECT_ROOT / "artifacts" / "inference" / "last_prediction.json"
INFER_CHECKPOINT = PROJECT_ROOT / "artifacts" / "training" / "models" / "transfer_best.pt"
INFER_LABEL_MAP = PROJECT_ROOT / "data" / "raw" / "label_num_to_disease_map.json"


# ************************************* HELPERS ************************************* #
def read_text_file(path: Path, fallback: str) -> str:
    if not path.exists():
        return fallback
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dict_to_dataframe(input_dict: dict, value_name: str) -> pd.DataFrame:
    rows = [{"class_id": int(key), value_name: value} for key, value in input_dict.items()]
    return pd.DataFrame(rows).sort_values("class_id")


def parse_preview_images() -> dict:
    grouped = {}
    if not PREVIEW_DIR.exists():
        return grouped
    pattern = re.compile(r"class_(\d+)_(original|augmented)_(.+)\.jpg$")
    for image_path in sorted(PREVIEW_DIR.glob("*.jpg")):
        match = pattern.match(image_path.name)
        if not match:
            continue
        class_id = int(match.group(1))
        kind = match.group(2)
        grouped.setdefault(class_id, {})[kind] = image_path
    return grouped


def run_python_command(command: list[str], title: str) -> tuple[int, str]:
    process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (
        f"{title}\n\n"
        f"Command: {' '.join(command)}\n\n"
        f"Exit code: {process.returncode}\n\n"
        f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}"
    )
    return process.returncode, output


def render_run_block(title: str, command: list[str], key: str) -> None:
    st.markdown(f"**{title}**")
    st.code(" ".join(command), language="bash")
    if st.button(f"Run {title}", key=f"run_{key}"):
        with st.spinner(f"Running {title}..."):
            code, output = run_python_command(command, title)
        if code == 0:
            st.success(f"{title} completed successfully.")
        else:
            st.error(f"{title} failed with exit code {code}.")
        st.code(output, language="text")


# ************************************* PAGE: DELIVERABLE 1 ************************************* #
def render_deliverable_1_page() -> None:
    st.title("Deliverable 1 - Data Understanding and Preparation")
    st.caption("Inspection, preprocessing plan, augmentation strategy, and imbalance analysis")

    command = [
        sys.executable,
        "src/data/prepare_dataset.py",
        "--check-noise",
    ]
    render_run_block("Deliverable 1 Pipeline", command, "d1")

    if not SUMMARY_PATH.exists():
        st.warning("Run Deliverable 1 first to generate summary artifacts.")
        return

    summary = read_json(SUMMARY_PATH)
    dataset_size = summary["dataset_size"]
    full_distribution = summary["full_distribution"]
    quality_checks = summary["quality_checks"]
    image_resolution = summary["image_resolution"]["most_common"]
    split_distribution = summary["split_distribution"]
    imbalance_ratio = max(full_distribution.values()) / min(full_distribution.values())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset Size", f"{dataset_size:,}")
    m2.metric("Classes", len(summary["label_map"]))
    m3.metric("Resolution", f"{image_resolution['width']}x{image_resolution['height']}")
    m4.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}x")

    st.subheader("Class Distribution")
    distribution_df = dict_to_dataframe(full_distribution, "count")
    distribution_df["percentage"] = (distribution_df["count"] / dataset_size * 100).round(2)
    st.dataframe(distribution_df, hide_index=True, use_container_width=True)
    st.bar_chart(distribution_df.set_index("class_id")["count"], use_container_width=True)

    st.subheader("Split Overview")
    split_rows = []
    for split_name in ("train", "val", "test"):
        split_counts = split_distribution[split_name]
        split_total = sum(split_counts.values())
        for class_id_str, count in split_counts.items():
            split_rows.append(
                {
                    "split": split_name,
                    "class_id": int(class_id_str),
                    "count": count,
                    "percentage": round(count / split_total * 100, 2),
                }
            )
    st.dataframe(pd.DataFrame(split_rows).sort_values(["split", "class_id"]), hide_index=True, use_container_width=True)

    st.subheader("Quality Checks")
    st.write(f"- Missing files: **{quality_checks['missing_files']}**")
    st.write(f"- Unreadable images: **{quality_checks['unreadable_images']}**")
    st.write(f"- Conflicting exact duplicates: **{quality_checks['conflicting_exact_duplicates']}**")

    st.subheader("Augmentation Preview")
    preview_map = parse_preview_images()
    if preview_map:
        for class_id in sorted(preview_map.keys()):
            st.markdown(f"**Class {class_id}**")
            left, right = st.columns(2)
            if "original" in preview_map[class_id]:
                left.image(str(preview_map[class_id]["original"]), caption="Original", use_container_width=True)
            if "augmented" in preview_map[class_id]:
                right.image(str(preview_map[class_id]["augmented"]), caption="Augmented", use_container_width=True)

    tabs = st.tabs(["Report", "Terminal Output"])
    with tabs[0]:
        st.markdown(read_text_file(REPORT_D1, "Report not found."))
    with tabs[1]:
        st.code(read_text_file(RUN_LOG_D1, "Terminal output not found."), language="text")


# ************************************* PAGE: DELIVERABLE 2 ************************************* #
def render_deliverable_2_page() -> None:
    st.title("Deliverable 2 - Model Selection and Training")
    st.caption("Baseline CNN + Fine-tuned ResNet18 — use **Fast GPU** for ~15–40 min with AMD DirectML.")

    st.info(
        "**AMD Radeon (no NVIDIA CUDA):** install DirectML GPU support:\n\n"
        "`pip install torch-directml`\n\n"
        "(Use a clean venv; see **README.md → Environment setup**.) Then run **Fast GPU training** below."
    )

    cmd_fast_gpu = [
        sys.executable,
        "src/train/train_models.py",
        "--model",
        "both",
        "--fast",
        "--device",
        "directml",
    ]
    cmd_fast_auto = [
        sys.executable,
        "src/train/train_models.py",
        "--model",
        "both",
        "--fast",
        "--device",
        "auto",
    ]
    cmd_fast_cpu = [
        sys.executable,
        "src/train/train_models.py",
        "--model",
        "both",
        "--fast",
        "--force-cpu",
    ]
    cmd_quality_cpu = [
        sys.executable,
        "src/train/train_models.py",
        "--model",
        "both",
        "--epochs",
        "8",
        "--batch-size",
        "32",
        "--tune",
        "--tune-epochs",
        "2",
        "--max-tune-trials",
        "3",
        "--force-cpu",
    ]

    render_run_block("Fast GPU (DirectML — Radeon)", cmd_fast_gpu, "d2_dml")
    render_run_block("Fast GPU (auto device: CUDA then DirectML)", cmd_fast_auto, "d2_auto")
    render_run_block("Fast CPU (much slower than GPU)", cmd_fast_cpu, "d2_cpu_fast")
    with st.expander("Long-run CPU (hours) — older default"):
        render_run_block("Quality / long CPU training", cmd_quality_cpu, "d2_long")

    if not TRAINING_SUMMARY.exists():
        st.warning("Training summary not found yet. Run Deliverable 2 training.")
        return

    summary = read_json(TRAINING_SUMMARY)
    baseline = summary.get("baseline", {})
    transfer = summary.get("transfer", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline best macro-F1", f"{baseline.get('best_val_macro_f1', 0.0):.4f}")
    c2.metric("Transfer best macro-F1", f"{transfer.get('best_val_macro_f1', 0.0):.4f}")
    c3.metric("Transfer - Baseline", f"{transfer.get('best_val_macro_f1', 0.0) - baseline.get('best_val_macro_f1', 0.0):+.4f}")

    comparison_rows = []
    for model_name in ("baseline", "transfer"):
        row = summary.get(model_name, {})
        hp = row.get("selected_hyperparameters", {})
        comparison_rows.append(
            {
                "model": model_name,
                "best_val_macro_f1": round(row.get("best_val_macro_f1", 0.0), 4),
                "learning_rate": hp.get("learning_rate"),
                "weight_decay": hp.get("weight_decay"),
                "used_tuning": hp.get("used_tuning"),
                "runtime_seconds": row.get("runtime_seconds"),
            }
        )
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)

    curve_rows = []
    for model_name, metrics_path in (
        ("baseline", BASELINE_TRAINING_METRICS),
        ("transfer", TRANSFER_TRAINING_METRICS),
    ):
        if not metrics_path.exists():
            continue
        payload = read_json(metrics_path)
        for item in payload.get("history", []):
            curve_rows.append(
                {
                    "model": model_name,
                    "epoch": item["epoch"],
                    "val_macro_f1": item["val_macro_f1"],
                    "val_accuracy": item["val_accuracy"],
                }
            )
    if curve_rows:
        curve_df = pd.DataFrame(curve_rows)
        st.line_chart(curve_df.pivot(index="epoch", columns="model", values="val_macro_f1"), use_container_width=True)
        st.line_chart(curve_df.pivot(index="epoch", columns="model", values="val_accuracy"), use_container_width=True)

    st.subheader("Terminal Output")
    st.code(read_text_file(RUN_LOG_D2, "Deliverable 2 terminal output not found."), language="text")


# ************************************* PAGE: DELIVERABLE 3 ************************************* #
def render_deliverable_3_page() -> None:
    st.title("Deliverable 3 - Evaluation and Error Analysis")
    st.caption("Held-out test evaluation with quantitative + qualitative analysis")

    cmd_eval = [sys.executable, "src/eval/evaluate_models.py", "--force-cpu"]
    render_run_block("Deliverable 3 Evaluation", cmd_eval, "d3")

    if not EVAL_SUMMARY.exists():
        st.warning("Evaluation summary not found. Run Deliverable 3 first.")
        return

    eval_summary = read_json(EVAL_SUMMARY)
    b = eval_summary.get("baseline", {})
    t = eval_summary.get("transfer", {})
    e1, e2, e3 = st.columns(3)
    e1.metric("Baseline Test Macro-F1", f"{b.get('macro_f1', 0.0):.4f}")
    e2.metric("Transfer Test Macro-F1", f"{t.get('macro_f1', 0.0):.4f}")
    e3.metric("Transfer Test Accuracy", f"{t.get('accuracy', 0.0):.4f}")

    if TRANSFER_EVAL_METRICS.exists():
        transfer_metrics = read_json(TRANSFER_EVAL_METRICS)
        st.subheader("Transfer Model - Per Class Metrics")
        per_class = transfer_metrics.get("per_class_metrics", {})
        rows = []
        for k, v in per_class.items():
            rows.append(
                {
                    "class_id": int(k),
                    "precision": round(v["precision"], 4),
                    "recall": round(v["recall"], 4),
                    "f1": round(v["f1"], 4),
                    "support": int(v["support"]),
                }
            )
        st.dataframe(pd.DataFrame(rows).sort_values("class_id"), hide_index=True, use_container_width=True)

        st.subheader("Top Failure Pairs")
        st.dataframe(pd.DataFrame(transfer_metrics.get("top_failure_pairs", [])), hide_index=True, use_container_width=True)

        st.subheader("Qualitative Misclassified Samples (Transfer)")
        if TRANSFER_QUAL_MIS.exists():
            image_paths = sorted(TRANSFER_QUAL_MIS.glob("*.jpg"))[:12]
            for i in range(0, len(image_paths), 3):
                cols = st.columns(3)
                for j, image_path in enumerate(image_paths[i : i + 3]):
                    cols[j].image(str(image_path), caption=image_path.name, use_container_width=True)

    tabs = st.tabs(["Evaluation Report", "Terminal Output"])
    with tabs[0]:
        st.markdown(read_text_file(REPORT_D3, "Evaluation report not found."))
    with tabs[1]:
        st.code(read_text_file(RUN_LOG_D3, "Deliverable 3 terminal output not found."), language="text")


# ************************************* PAGE: DELIVERABLE 4 ************************************* #
def render_deliverable_4_page() -> None:
    st.title("Deliverable 4 - Inference and Deployment Readiness")
    st.caption("Single-image inference script + FastAPI + Docker setup")

    st.subheader("Single Image Inference (Run from Dashboard)")
    uploaded = st.file_uploader("Upload image for prediction", type=["jpg", "jpeg", "png"], key="infer_upload")
    if uploaded is not None:
        st.image(uploaded, caption="Uploaded image", width=320)
        if st.button("Run Prediction", key="run_prediction"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded.getvalue())
                tmp_path = Path(tmp_file.name)

            command = [
                sys.executable,
                "src/infer/predict_image.py",
                "--image",
                str(tmp_path),
                "--checkpoint",
                str(INFER_CHECKPOINT),
                "--label-map",
                str(INFER_LABEL_MAP),
                "--force-cpu",
            ]
            with st.spinner("Running inference..."):
                code, output = run_python_command(command, "Inference Script")
            if code == 0 and INFER_LAST_OUTPUT.exists():
                st.success("Inference completed successfully.")
                st.json(read_json(INFER_LAST_OUTPUT))
            else:
                st.error("Inference failed.")
            st.code(output, language="text")

    st.subheader("FastAPI Service Commands")
    st.code("uvicorn src.infer.api:app --host 0.0.0.0 --port 8000", language="bash")
    st.code('curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"', language="bash")

    st.subheader("Docker Commands")
    st.code("docker build -t cassava-inference .", language="bash")
    st.code("docker run --rm -p 8000:8000 cassava-inference", language="bash")

    st.subheader("Current Inference Output")
    st.code(read_text_file(INFER_LAST_OUTPUT, "No inference output yet."), language="json")


# ************************************* PAGE: DELIVERABLE 5 ************************************* #
def render_deliverable_5_page() -> None:
    st.title("Deliverable 5 - Engineering Quality")
    st.caption("Project structure, separation of concerns, reproducibility, and deployment maturity")

    st.markdown("### Architecture Checklist")
    checklist = [
        "Data preparation separated in `src/data/prepare_dataset.py`",
        "Model training separated in `src/train/train_models.py`",
        "Evaluation/error analysis separated in `src/eval/evaluate_models.py`",
        "Inference script separated in `src/infer/predict_image.py`",
        "REST API separated in `src/infer/api.py`",
        "End-to-end runner in `src/run_deliverables.py`",
        "Config file for inference in `configs/inference_config.json`",
        "Dockerized API deployment using `Dockerfile`",
        "Artifacts stored under `artifacts/` for reproducibility",
    ]
    for item in checklist:
        st.write(f"- {item}")

    st.markdown("### Run End-to-End from Dashboard")
    command = [sys.executable, "src/run_deliverables.py", "--force-cpu"]
    render_run_block("Deliverables 1-3 End-to-End", command, "end2end")

    st.markdown("### Split File Counts")
    if TRAIN_SPLIT_PATH.exists():
        counts_df = pd.DataFrame(
            {
                "split": ["train", "val", "test"],
                "count": [
                    len(pd.read_csv(TRAIN_SPLIT_PATH)),
                    len(pd.read_csv(VAL_SPLIT_PATH)),
                    len(pd.read_csv(TEST_SPLIT_PATH)),
                ],
            }
        )
        st.dataframe(counts_df, hide_index=True, use_container_width=True)
        st.bar_chart(counts_df.set_index("split")["count"], use_container_width=True)


# ************************************* UI ROOT ************************************* #
st.set_page_config(page_title="Assessment Dashboard", layout="wide")
selected_page = st.sidebar.radio(
    "Choose Page",
    [
        "Deliverable 1",
        "Deliverable 2",
        "Deliverable 3",
        "Deliverable 4",
        "Deliverable 5",
    ],
    index=0,
)

if selected_page == "Deliverable 1":
    render_deliverable_1_page()
elif selected_page == "Deliverable 2":
    render_deliverable_2_page()
elif selected_page == "Deliverable 3":
    render_deliverable_3_page()
elif selected_page == "Deliverable 4":
    render_deliverable_4_page()
else:
    render_deliverable_5_page()

