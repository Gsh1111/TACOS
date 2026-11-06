import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import torch
import utils
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from torch.optim import lr_scheduler
from config_loader import Config, get_argparser
from spikingjelly.activation_based import functional
from spikingjelly.datasets import padded_sequence_mask
from datasets import SHD_dataloaders, SSC_dataloaders, NTIDIGITS_dataloaders
from models import SpikCommander, SpikeSCR, SpikFormer, SDT, QKFormer
from utils import (
    init_logger,
    build_optimizer,
    calc_loss,
    calc_metric,
    make_output_dirs,
    log_drop_config,
)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@torch.no_grad()
def eval_model(config, model, loader, device, phase="Eval", use_amp=False):
    """Evaluate the model using the configured temporal pruning settings."""
    model.eval()
    loss_list, metric_list, batch_weights = [], [], []

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for x, y, x_len in tqdm(loader, desc=phase, leave=False, miniters=10):
        x = x.float().to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_len = x_len.to(device, non_blocking=True)

        attention_mask = padded_sequence_mask(x_len).transpose(0, 1)

        y_oh = F.one_hot(y, config.n_outputs).float()

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(x, attention_mask)
                loss = calc_loss(config, output, y_oh)
        else:
            output = model(x, attention_mask)
            loss = calc_loss(config, output, y_oh)

        metric = calc_metric(config, output, y_oh)

        loss_list.append(loss.item())
        metric_list.append(metric)
        batch_weights.append(len(y))

        functional.reset_net(model)

    return float(np.average(loss_list, weights=batch_weights)), float(
        np.average(metric_list, weights=batch_weights)
    )


def train_model(
    config,
    train_loader,
    valid_loader,
    test_loader,
    device,
    model,
    optimizer,
    scheduler,
    num_epochs,
    logger,
    run_tag,
    output_dirs,
):
    """Train and evaluate the model across epochs."""
    loss_epochs = {"train": [], "valid": [], "test": []}
    metric_epochs = {"train": [], "valid": [], "test": []}
    best_metric_val = 0.0
    best_metric_test = 0.0

    use_amp = getattr(config, "use_amp", True)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if use_amp:
        logger.info("[AMP] Mixed precision training enabled")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        loss_batch, metric_batch = [], []
        num_batches = len(train_loader)

        for i, (x, y, x_len) in enumerate(train_loader, start=1):
            if i == 1:
                logger.info(f"X shape: {x.shape}")

            x = x.float().to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_len = x_len.to(device, non_blocking=True)

            # # Attention mask
            attention_mask = padded_sequence_mask(x_len).transpose(0, 1)
            y_oh = F.one_hot(y, config.n_outputs).float()

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(x, attention_mask)
                    loss = calc_loss(config, output, y_oh)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(x, attention_mask)
                loss = calc_loss(config, output, y_oh)
                loss.backward()
                optimizer.step()

            metric = calc_metric(config, output, y_oh)
            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)

            if i % 10 == 0 or i == num_batches:
                progress = 100.0 * i / num_batches

                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"{progress:.1f}% "
                    f"({i}/{num_batches}) "
                    f"Loss={loss.item():.4f} "
                    f"Acc={metric * 100:.2f}%"
                )

        scheduler.step(epoch + 1)

        loss_epochs["train"].append(float(np.mean(loss_batch)))
        metric_epochs["train"].append(float(np.mean(metric_batch)))

        if valid_loader is not None:
            loss_valid, metric_valid = eval_model(
                config, model, valid_loader, device, "Valid", use_amp
            )
        else:
            loss_valid, metric_valid = 100.0, 0.0

        loss_epochs["valid"].append(loss_valid)
        metric_epochs["valid"].append(metric_valid)

        if test_loader is not None:
            loss_test, metric_test = eval_model(config, model, test_loader, device, "Test", use_amp)
        else:
            loss_test, metric_test = 100.0, 0.0

        loss_epochs["test"].append(loss_test)
        metric_epochs["test"].append(metric_test)

        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024
            mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024
            gpu_info = f"GPU_Mem={mem_alloc:.1f}/{mem_reserved:.1f}MB"
        else:
            gpu_info = "GPU_N/A"

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss={loss_epochs['train'][-1]:.4f} Acc={100 * metric_epochs['train'][-1]:.2f}% | "
            f"Valid Loss={loss_valid:.4f} Acc={100 * metric_valid:.2f}% | "
            f"Test Loss={loss_test:.4f} Acc={100 * metric_test:.2f}% | "
            f"LR={optimizer.param_groups[0]['lr']:.6f} | "
            f"Time={time.time() - epoch_start:.1f}s | "
            f"{gpu_info}"
        )

        checkpoint_dir = output_dirs["checkpoints"]

        if valid_loader and metric_valid > best_metric_val:
            best_metric_val = metric_valid
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"Best_ACC_VAL.pth"))
            logger.info(f"  → Saved best ACC VAL model ({100 * best_metric_val:.2f}%)")

        if test_loader and metric_test > best_metric_test:
            best_metric_test = metric_test
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"Best_ACC_TEST.pth"))
            logger.info(f"  → Saved best ACC TEST model ({100 * best_metric_test:.2f}%)")

        logger.info(
            f"Best Valid Acc: {100 * best_metric_val:.2f}% | Best Test Acc: {100 * best_metric_test:.2f}%"
        )

        if epoch % 50 == 0:
            model.print_drop_stats()
            model.reset_drop_stats()

    if getattr(config, "make_plot", True):
        make_accuracy_plot(metric_epochs, output_dirs["plots"], run_tag)

    return loss_epochs, metric_epochs


def make_accuracy_plot(metric_epochs, out_dir, run_tag):
    """Plot the training curves."""
    train_acc = [x * 100 for x in metric_epochs["train"]]
    valid_acc = [x * 100 for x in metric_epochs["valid"]]
    test_acc = [x * 100 for x in metric_epochs["test"]]
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, "-o", label="Train", alpha=0.8, markersize=3)
    plt.plot(epochs, valid_acc, "-^", label="Valid", alpha=0.8, markersize=3)
    plt.plot(epochs, test_acc, "-s", label="Test", alpha=0.8, markersize=3)
    plt.title(f"Accuracy - {run_tag}", fontsize=10)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"Accuracy__{run_tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()


def setup_device(config, logger):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device {device}")
    return device


import hashlib


def get_tensor_hash(tensor: torch.Tensor) -> str:
    """Hash the tensor for exact comparisons."""

    data = tensor.detach().cpu().numpy().tobytes()
    return hashlib.md5(data).hexdigest()


def setup_dataloaders(config, dataset_seed=312):
    if config.dataset == "shd":
        train_loader, valid_loader = SHD_dataloaders(config, dataset_seed=dataset_seed)
        test_loader = None
    elif config.dataset == "ssc":
        train_loader, valid_loader, test_loader = SSC_dataloaders(config, dataset_seed=dataset_seed)
    elif config.dataset == "ntidigits":
        train_loader, valid_loader = NTIDIGITS_dataloaders(config, dataset_seed=dataset_seed)
        test_loader = None
    else:
        raise ValueError(f"Dataset {config.dataset} not implemented")

    x, y, x_len = next(iter(train_loader))
    print(f"[DEBUG] Dataset seed verification:")
    print(f"  - Label hash:      {get_tensor_hash(y)}")
    print(f"  - Input hash:      {get_tensor_hash(x)}")
    print(f"  - Length hash:     {get_tensor_hash(x_len)}")
    print(f"  - First sample:    label={y[0].item()}, len={x_len[0].item()}")

    return train_loader, valid_loader, test_loader


def create_model(config, device):
    backbones = {
        "spikcommander": SpikCommander,
        "spikescr": SpikeSCR,
        "spikformer": SpikFormer,
        "sdt": SDT,
        "qkformer": QKFormer,
    }
    name = config.model_name.lower()
    if name not in backbones:
        raise ValueError(f"Unknown model: {config.model_name}")
    return backbones[name](config).to(device)


def summarize_accuracies(values):
    if not values:
        return 0.0, 0.0, 0.0, 0.0

    std = 0.0
    if len(values) > 1:
        std = float(np.std(values, ddof=1))
    return float(np.mean(values)), std, float(np.max(values)), float(np.min(values))


def main():
    parser = get_argparser()
    args = parser.parse_args()
    config = Config(config_path=args.config, args=args)
    config.print_config()

    base_tag = args.run_tag
    if not base_tag:
        raise ValueError("--run-tag is required! Please provide a base tag from launcher.")

    results_valid = []
    results_test = []

    for seed in config.seed_list:
        run_start = time.time()

        config.update(seed=seed)

        run_tag = f"{base_tag}__seed={seed}"

        output_dirs = make_output_dirs(config, base_tag=base_tag, seed=seed)

        config.log_dir = output_dirs["logs"]
        logger = init_logger(config, "training")

        logger.info("=" * 80)
        logger.info(f"Base tag: {base_tag}")
        logger.info(f"Run tag : {run_tag}")
        logger.info("=" * 80)

        device = setup_device(config, logger)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        logger.info(f"[TIME] Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"The GPU is {config.gpu} | Device = {device}")
        logger.info(f"Seed: {seed} | Epochs: {config.epochs} | Batch: {config.batch_size}")
        logger.info(f"Model: {config.model_name} | Dataset: {config.dataset}")

        log_drop_config(logger, config)

        train_loader, valid_loader, test_loader = setup_dataloaders(config, dataset_seed=312)
        for split, loader in (
            ("train", train_loader),
            ("valid", valid_loader),
            ("test", test_loader),
        ):
            if loader is not None:
                logger.info(f"{split}: {loader.num_workers} workers, {len(loader)} batches")

        # Dataset-specific logs
        logger.info(f"Duration: {config.duration}")
        logger.info(f"N bins: {config.n_bins}")

        if config.use_aug:
            logger.info(f"Augmentation: {config.use_aug}")
            logger.info(f"- TN_mask_aug_proba: {config.TN_mask_aug_proba}")
            logger.info(f"- time_mask_proportion: {config.time_mask_proportion}")
            logger.info(f"- neuron_mask_size: {config.neuron_mask_size}")
        else:
            logger.info("Augmentations: None")

        model = create_model(config, device)
        logger.info(f"Model parameters: {utils.count_parameters(model):,}")
        logger.info(str(model))

        optimizer = build_optimizer(config, model)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.t_max)

        config.save(os.path.join(output_dirs["configs"], f"config_{run_tag}.yaml"))

        loss_epochs, metric_epochs = train_model(
            config,
            train_loader,
            valid_loader,
            test_loader,
            device,
            model,
            optimizer,
            scheduler,
            config.epochs,
            logger,
            run_tag,
            output_dirs,
        )

        best_valid = max(metric_epochs["valid"], default=0.0)
        results_valid.append(best_valid)

        best_test = max(metric_epochs["test"], default=0.0)
        results_test.append(best_test)

        run_time = time.time() - run_start
        logger.info("=" * 80)
        logger.info(f"Run completed in {run_time / 60:.1f} minutes")
        logger.info("=" * 80)

    mean_valid, std_valid, max_valid, min_valid = summarize_accuracies(results_valid)
    mean_test, std_test, max_test, min_test = summarize_accuracies(results_test)

    logger.info(f"\n{'=' * 80}")
    logger.info(
        f"[VALID] n={len(results_valid)} "
        f"Mean={100 * mean_valid:.2f}% Std={100 * std_valid:.2f}% "
        f"Max={100 * max_valid:.2f}% Min={100 * min_valid:.2f}%"
    )
    logger.info(
        f"[TEST ] n={len(results_test)} "
        f"Mean={100 * mean_test:.2f}% Std={100 * std_test:.2f}% "
        f"Max={100 * max_test:.2f}% Min={100 * min_test:.2f}%"
    )
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
