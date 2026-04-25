import argparse
import subprocess
import sys
from pathlib import Path


TASKS = {
    "super_resolution": {
        "task_config": "task_configurations/dcdp_paper_super_resolution_config.yaml",
        "purification_config": "purification_configurations/purification_config_super_resolution.yaml",
        "source": "paper",
    },
    "inpainting_box": {
        "task_config": "task_configurations/dcdp_user_inpainting_box_config.yaml",
        "purification_config": "purification_configurations/purification_config_inpainting.yaml",
        "source": "user-measurement + paper-hyperparameters",
    },
    "gaussian_blur": {
        "task_config": "task_configurations/dcdp_paper_gaussian_blur_config.yaml",
        "purification_config": "purification_configurations/purification_config_gaussian_deblur.yaml",
        "source": "paper",
    },
    "motion_blur": {
        "task_config": "task_configurations/dcdp_paper_motion_blur_config.yaml",
        "purification_config": "purification_configurations/dcdp_paper_motion_deblur.yaml",
        "source": "paper",
    },
    "inpainting_random": {
        "task_config": "task_configurations/dcdp_repo_inpainting_random_config.yaml",
        "purification_config": "purification_configurations/purification_config_inpainting.yaml",
        "source": "repo",
    },
    "phase_retrieval": {
        "task_config": "task_configurations/dcdp_repo_phase_retrieval_config.yaml",
        "purification_config": "purification_configurations/purification_config_phase_retrieval.yaml",
        "source": "repo",
    },
}


def parse_task_names(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(TASKS))
    if unknown:
        known = ", ".join(TASKS)
        raise argparse.ArgumentTypeError(f"Unknown task(s): {', '.join(unknown)}. Known tasks: {known}")
    return names


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run DCDP inverse-problem presets from the DCDP paper/repository defaults."
    )
    parser.add_argument("--tasks", type=parse_task_names, default=list(TASKS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=repo_root / "purification_results" / "dcdp_defaults")
    parser.add_argument("--model-config", type=Path, default=repo_root / "model_configurations" / "model_config_ffhq.yaml")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root or (repo_root / "data" / "ffhq")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    for task_name in args.tasks:
        task = TASKS[task_name]
        cmd = [
            str(args.python),
            str(repo_root / "dcdp.py"),
            "--model_config",
            str(args.model_config),
            "--task_config",
            str(repo_root / task["task_config"]),
            "--purification_config",
            str(repo_root / task["purification_config"]),
            "--gpu",
            str(args.gpu),
            "--save_dir",
            str(args.save_dir),
            "--dataset_root",
            str(dataset_root),
            "--max_images",
            str(args.max_images),
            "--seed",
            str(args.seed),
        ]
        print(f"\n=== {task_name} ({task['source']} preset) ===")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=repo_root, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
