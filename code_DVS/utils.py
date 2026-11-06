import re
import os
import sys
import yaml
import torch
import random
import logging
import numpy as np
import torch.nn as nn

from torch import optim
from datetime import datetime
from logging import getLogger
from typing import Dict, Optional, List


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert_config_dict(config_dict):
    """This function convert the str parameters to their original type."""
    for key in config_dict:
        param = config_dict[key]
        if not isinstance(param, str):
            continue
        try:
            value = eval(param)
            if not isinstance(value, (str, int, float, list, tuple, dict, bool)):
                value = param
        except (NameError, SyntaxError, TypeError):
            if isinstance(param, str):
                if param.lower() == "true":
                    value = True
                elif param.lower() == "false":
                    value = False
                else:
                    value = param
            else:
                value = param
        config_dict[key] = value
    return config_dict


def read_configuration(config_file):
    # read configuration from yaml file
    yaml_loader = yaml.FullLoader
    yaml_loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    with open(config_file, "r") as f:
        yaml_config_dict = yaml.load(f.read(), Loader=yaml_loader)

    # read configuration from cmd line
    cmd_config_dict = dict()
    unrecognized_args = []
    if "ipykernel_launcher" not in sys.argv[0]:
        for arg in sys.argv[1:]:
            if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                unrecognized_args.append(arg)
                continue
            cmd_arg_name, cmd_arg_value = arg[2:].split("=")
            if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                raise SyntaxError(
                    "There are duplicate commend arg '%s' with different value." % arg
                )
            else:
                cmd_config_dict[cmd_arg_name] = cmd_arg_value
    if len(unrecognized_args) > 0:
        logger = getLogger()
        # logger.warning('command line args [{}] will not be used in TextBox'.format(' '.join(unrecognized_args)))

    cmd_config_dict = convert_config_dict(cmd_config_dict)

    final_config_dict = dict()
    final_config_dict.update(yaml_config_dict)
    final_config_dict.update(cmd_config_dict)

    return final_config_dict


def check_versions():
    python_version = sys.version.split(" ")[0]
    print("============== Checking Packages versions ================")
    print(f"python {python_version}")
    print(f"numpy {np.__version__}")
    print(f"pytorch {torch.__version__}")


def set_seed(seed):
    print(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_local_time():
    cur = datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")
    return cur


def init_logger(config, run_type: str, filename: str = "train.log"):
    """Log to the console and the run-specific log file without duplicating handlers."""
    log_dir = getattr(config, "log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    logfilepath = os.path.join(log_dir, filename)

    logger = getLogger("training")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    # formatter
    filefmt = "%(asctime)s %(levelname)s - %(message)s"
    filedatefmt = "%Y-%m-%d %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)s %(levelname)s - %(message)s"
    sdatefmt = "%m-%d %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)

    # handlers
    fh = logging.FileHandler(logfilepath, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(sformatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    seed = getattr(config, "seed", "na")
    logger.info(f"[LOGGER] file={logfilepath} | run_type={run_type} | seed={seed}")

    return logger


def build_optimizer(config, model):
    if config.optimizer_w == "adam":
        # optimizer = optim.Adam(model.parameters(), lr = config.lr_w, betas=(0.9,0.999),weight_decay=config.weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=config.lr_w, betas=(0.9, 0.999))
    elif config.optimizer_w == "adamw":
        # optimizer = optim.AdamW(model.parameters(), lr=config.lr_w, betas=(0.9, 0.999),weight_decay=config.weight_decay)
        optimizer = optim.AdamW(
            model.parameters(), lr=config.lr_w, betas=(0.9, 0.999), weight_decay=config.weight_decay
        )  # weight_decay default 1e-2
    return optimizer


def calc_loss(config, output, y):
    """Calculate loss based on model type"""

    if config.loss == "mean":
        m = torch.mean(output, 0)
    elif config.loss == "max":
        m, _ = torch.max(output, 0)
    elif config.loss == "spike_count":
        m = torch.sum(output, 0)
    elif config.loss == "sum":
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)

    if config.loss_fn == "CEloss":
        return nn.CrossEntropyLoss()(m, y)


def calc_metric(config, output, y):
    """Calculate accuracy metric"""

    if config.loss == "mean":
        m = torch.mean(output, 0)
    elif config.loss == "max":
        m, _ = torch.max(output, 0)
    elif config.loss == "spike_count":
        m = torch.sum(output, 0)
    elif config.loss == "sum":
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)
    elif config.loss == "none":
        m = output

    if y.ndim > 1:
        target_idx = y.argmax(dim=-1)
    else:
        target_idx = y

    pred_idx = m.argmax(dim=-1)  # (B,)
    acc = (pred_idx == target_idx).float().mean().item()
    return acc


def _safe(s: str) -> str:
    """Make a string safe for filesystem paths."""
    s = str(s)
    return s.replace(" ", "_").replace("/", "-").replace("\\", "-").replace(":", "-")


def _fmt_ratio(r: float, ndigits: int = 2) -> str:
    # r0.50 / r1.00 style
    return f"{round(float(r), ndigits):.{ndigits}f}"


def _layer_tag(layer: int) -> str:
    # -1 => emb, else L{idx}
    if int(layer) == -1:
        return "emb"
    else:
        return f"L{int(layer)}"


def make_output_dirs(config, base_tag: str, seed: int, base_dir: str = "runs") -> Dict[str, str]:
    """Create per-run output directories grouped by dataset, model, configuration and seed."""
    dataset = _safe(str(getattr(config, "dataset", "data")).lower())
    model = _safe(str(getattr(config, "model_name", "model")))

    drop_on = bool(getattr(config, "drop_sequence", False))

    seed_folder = f"seed={int(seed)}"
    time_steps_folder = f"time_steps={int(config.time_steps)}"

    if "__" in base_tag:
        timestamp = base_tag.split("__")[-1]
    else:
        timestamp = base_tag
    ts_folder = _safe(timestamp)

    if not drop_on:
        parts = [
            base_dir,
            dataset,
            model,
            time_steps_folder,
            "no_drop",
            seed_folder,
            ts_folder,
        ]
    else:
        target = _safe(str(getattr(config, "drop_target", "feature")).lower())
        selector = _safe(str(getattr(config, "drop_selector", "unknown")).lower())
        score = _safe(str(getattr(config, "drop_score", "unknown")).lower())
        ratio = _safe(_fmt_ratio(getattr(config, "drop_ratio", 0.0)))

        if target == "feature":
            after = _safe(_layer_tag(getattr(config, "drop_after_layer", -1)))
        else:
            after = "NAN"

        use_windows = bool(getattr(config, "drop_use_windows", False))
        if use_windows:
            wins: Optional[List[int]] = getattr(config, "drop_windows", None)
            if wins and len(wins) > 0:
                wins_sorted = sorted(int(x) for x in wins)
                win_folder = "win=" + "-".join(str(x) for x in wins_sorted)
            else:
                win_folder = "win=empty"
        else:
            win_folder = "win=NAN"

        parts = [
            base_dir,
            dataset,
            model,
            time_steps_folder,
            "drop",
            f"target={target}",
            f"after={after}",
            f"selector={selector}",
            f"score={score}",
            f"ratio={ratio}",
            _safe(win_folder),
            seed_folder,
            ts_folder,
        ]

    base = os.path.join(*parts)

    dirs = {
        "base": base,
        "logs": os.path.join(base, "logs"),
        "plots": os.path.join(base, "plots"),
        "checkpoints": os.path.join(base, "checkpoints"),
        "structures": os.path.join(base, "model_structure"),
        "configs": os.path.join(base, "configs"),
    }

    for p in dirs.values():
        os.makedirs(p, exist_ok=True)

    return dirs


def log_drop_config(logger, config):
    """Log the temporal pruning configuration."""
    drop_on = bool(getattr(config, "drop_sequence", False))

    if not drop_on:
        logger.info("[DROP] Disabled (no_drop)")
        return

    target = str(getattr(config, "drop_target", "feature")).lower()
    selector = str(getattr(config, "drop_selector", "unknown")).lower()
    ratio = float(getattr(config, "drop_ratio", 0.0))
    score = str(getattr(config, "drop_score", "unknown")).lower()

    if target == "feature":
        layer = int(getattr(config, "drop_after_layer", -1))
        if layer == -1:
            after = "emb"
        else:
            after = f"L{layer}"
    else:
        after = "NAN"

    use_windows = bool(getattr(config, "drop_use_windows", False))
    if use_windows:
        wins: Optional[List[int]] = getattr(config, "drop_windows", None)
        if wins and len(wins) > 0:
            win_tag = "win=" + "-".join(str(int(x)) for x in wins)
        else:
            win_tag = "win=empty"
    else:
        win_tag = "win=NAN"

    logger.info("[DROP] Enabled")
    logger.info(f"  target={target}")
    logger.info(f"  after={after}")
    logger.info(f"  selector={selector}")
    logger.info(f"  ratio={ratio:.3f}")
    logger.info(f"  score={score}")
    logger.info(f"  {win_tag}")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print(target[None])
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
