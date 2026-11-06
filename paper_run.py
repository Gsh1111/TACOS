import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PREVIEW_FIELDS = (
    "model_name",
    "time_steps",
    "duration",
    "mlp_ratios",
    "drop_sequence",
    "drop_ratio",
    "drop_windows",
    "drop_comp_window",
    "drop_comp_cap",
    "seed_list",
    "lr",
    "epochs",
    "batch_size",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=["dvs", "speech"])
    parser.add_argument("model")
    parser.add_argument("mode", choices=["full", "half", "tacos"])
    parser.add_argument("--dataset")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def locate_preset(domain, dataset, model):
    if domain == "dvs":
        pipeline = REPO_ROOT / "code_DVS"
        config_dir = pipeline / "configs" / dataset / "16T"
    else:
        pipeline = REPO_ROOT / "code_speech"
        config_dir = pipeline / "configs" / "T=100" / dataset.upper()
    preset = config_dir / f"{dataset}_{model}.yaml"
    if not preset.is_file():
        raise FileNotFoundError(preset)
    return pipeline, preset


def main():
    args = parse_args()
    dataset = args.dataset
    if not dataset:
        if args.domain == "dvs":
            dataset = "cifar10dvs"
        else:
            dataset = "shd"
    pipeline, preset = locate_preset(args.domain, dataset, args.model)
    data_dir = args.data.resolve()
    output_dir = args.output.resolve()

    sys.path.insert(0, str(pipeline))
    from config_loader import Config

    overrides = argparse.Namespace(
        tacos_mode=args.mode,
        tacos_profile="paper",
        datasets_path=str(data_dir),
        seed_list=args.seeds,
    )
    config = Config(str(preset), overrides)
    command = [
        sys.executable,
        str(pipeline / "train.py"),
        "--config",
        str(preset),
        "--tacos-mode",
        args.mode,
        "--tacos-profile",
        "paper",
        "--datasets-path",
        str(data_dir),
        "--seed-list",
        *map(str, args.seeds),
        "--gpu",
        str(args.gpu),
        "--run-tag",
        f"paper-r1-{dataset}-{args.model}-{args.mode}",
    ]
    preview = {
        "command": command,
        "cwd": str(output_dir),
        "resolved": {field: getattr(config, field, None) for field in PREVIEW_FIELDS},
        "execute": args.execute,
    }
    print(json.dumps(preview, indent=2), flush=True)
    if not args.execute:
        return 0
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    return subprocess.call(command, cwd=output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
