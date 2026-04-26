import argparse
import subprocess
import sys
from pathlib import Path


TASKS = {
    "super_resolution": {
        "task_config": "task_configurations/dcdp_paper_super_resolution_config.yaml",
        "purification_config": "purification_configurations/purification_config_super_resolution.yaml",
        "output_name": "dcdp_paper_super_resolution",
        "source": "paper",
    },
    "inpainting_box": {
        "task_config": "task_configurations/dcdp_user_inpainting_box_config.yaml",
        "purification_config": "purification_configurations/purification_config_inpainting.yaml",
        "output_name": "dcdp_user_inpainting_box",
        "source": "user-measurement + paper-hyperparameters",
    },
    "gaussian_blur": {
        "task_config": "task_configurations/dcdp_paper_gaussian_blur_config.yaml",
        "purification_config": "purification_configurations/purification_config_gaussian_deblur.yaml",
        "output_name": "dcdp_paper_gaussian_blur",
        "source": "paper",
    },
    "motion_blur": {
        "task_config": "task_configurations/dcdp_paper_motion_blur_config.yaml",
        "purification_config": "purification_configurations/dcdp_paper_motion_deblur.yaml",
        "output_name": "dcdp_paper_motion_blur",
        "source": "paper",
    },
    "inpainting_random": {
        "task_config": "task_configurations/dcdp_repo_inpainting_random_config.yaml",
        "purification_config": "purification_configurations/purification_config_inpainting.yaml",
        "output_name": "dcdp_repo_inpainting_random",
        "source": "repo",
    },
    "phase_retrieval": {
        "task_config": "task_configurations/dcdp_repo_phase_retrieval_config.yaml",
        "purification_config": "purification_configurations/purification_config_phase_retrieval.yaml",
        "output_name": "dcdp_repo_phase_retrieval",
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
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=repo_root / "purification_results" / "dcdp_defaults")
    parser.add_argument("--model-config", type=Path, default=repo_root / "model_configurations" / "model_config_ffhq.yaml")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mode", choices=["ddim", "tweedie"], default="ddim")
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--save-measurements", action="store_true")
    parser.add_argument("--save-progress-figures", action="store_true")
    parser.add_argument("--save-recon-history", action="store_true")
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
            "--start_idx",
            str(args.start_idx),
            "--batch_size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
        ]
        if args.mode == "tweedie":
            cmd.extend(["--full_ddim_override", "false"])
        elif args.mode == "ddim":
            cmd.extend(["--full_ddim_override", "true"])
        if args.ddim_steps is not None:
            cmd.extend(["--ddim_num_iterations_override", str(args.ddim_steps)])
        if args.skip_metrics:
            cmd.append("--skip_metrics")
        if args.save_measurements:
            cmd.append("--save_measurements")
        if args.save_progress_figures:
            cmd.append("--save_progress_figures")
        if args.save_recon_history:
            cmd.append("--save_recon_history")
        output_name = task["output_name"]
        print(f"\n=== {task_name} ({task['source']} preset) ===")
        print(f"Progress JSON: {args.save_dir / output_name / 'progress.json'}")
        print(f"History JSON: {args.save_dir / output_name / 'history.json'}")
        print(f"Generated images zip: {args.save_dir / output_name / 'generated_images.zip'}")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=repo_root, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
