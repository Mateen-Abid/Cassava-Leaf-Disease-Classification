import argparse
import subprocess
import sys
from pathlib import Path


# ************************************* HELPERS ************************************* #
def run_command(command: list[str], title: str) -> None:
    print("", flush=True)
    print("=" * 88, flush=True)
    print(title, flush=True)
    print("Command:", " ".join(command), flush=True)
    print("=" * 88, flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{title} failed with exit code {result.returncode}")


# ************************************* CLI ************************************* #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Deliverable 1, 2, and 3 in one command."
    )
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--tune-epochs", type=int, default=2)
    parser.add_argument("--max-tune-trials", type=int, default=4)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick sanity mode for fast local verification.",
    )
    return parser.parse_args()


# ************************************* MAIN ************************************* #
def main() -> None:
    args = parse_args()
    python_exe = sys.executable
    root = Path(__file__).resolve().parents[1]

    deliverable_1_cmd = [
        python_exe,
        str(root / "src" / "data" / "prepare_dataset.py"),
        "--check-noise",
    ]

    deliverable_2_cmd = [
        python_exe,
        str(root / "src" / "train" / "train_models.py"),
        "--model",
        "both",
    ]

    if args.quick:
        print("NOTE: --quick mode is for sanity checks only (not final quality metrics).", flush=True)
        deliverable_2_cmd.extend(
            [
                "--epochs",
                "1",
                "--batch-size",
                str(args.batch_size),
                "--tune",
                "--tune-epochs",
                "1",
                "--max-tune-trials",
                "2",
                "--max-train-samples",
                "256",
                "--max-val-samples",
                "128",
                "--force-cpu",
            ]
        )
    else:
        deliverable_2_cmd.extend(
            [
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--tune",
                "--tune-epochs",
                str(args.tune_epochs),
                "--max-tune-trials",
                str(args.max_tune_trials),
            ]
        )
        if args.force_cpu:
            deliverable_2_cmd.append("--force-cpu")

    step_2_label = "STEP 2/2 - DELIVERABLE 2 (MODEL TRAINING)" if args.skip_evaluation else "STEP 2/3 - DELIVERABLE 2 (MODEL TRAINING)"
    step_1_label = "STEP 1/2 - DELIVERABLE 1 (DATA PREPARATION)" if args.skip_evaluation else "STEP 1/3 - DELIVERABLE 1 (DATA PREPARATION)"
    run_command(deliverable_1_cmd, step_1_label)
    run_command(deliverable_2_cmd, step_2_label)
    if not args.skip_evaluation:
        deliverable_3_cmd = [
            python_exe,
            str(root / "src" / "eval" / "evaluate_models.py"),
        ]
        if args.force_cpu or args.quick:
            deliverable_3_cmd.append("--force-cpu")
        run_command(
            deliverable_3_cmd,
            "STEP 3/3 - DELIVERABLE 3 (EVALUATION + ERROR ANALYSIS)",
        )

    print("")
    print("All requested deliverables finished successfully.")
    print("You can now open the UI:")
    print("streamlit run ui/dashboard.py")


if __name__ == "__main__":
    main()

