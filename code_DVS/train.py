import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import math
import yaml
import torch
import random
import autoaugment
import numpy as np
import torch.nn as nn

from datetime import datetime
from torchvision import transforms

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from config_loader import Config, get_argparser
from models import QKFormer, SpikFormer, SpikingFormer, SpikeDrivenTransformer
from utils import init_logger, make_output_dirs, count_parameters, log_drop_config, accuracy

from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import SoftTargetCrossEntropy

from spikingjelly.activation_based import functional
from spikingjelly.datasets import dvs128_gesture, cifar10_dvs, n_caltech101


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config, device):
    """Build the configured model."""
    model_name = config.model_name.lower()

    if model_name in ("spike_driven_transformer", "sdt"):
        return SpikeDrivenTransformer(
            args=config,
            num_classes=config.n_outputs,
            num_heads=config.num_heads,
            mlp_ratios=config.mlp_ratios,
        ).to(device)
    if model_name == "spikformer":
        return SpikFormer(
            args=config,
            num_classes=config.n_outputs,
            num_heads=config.num_heads,
            mlp_ratios=config.mlp_ratios,
        ).to(device)
    if model_name == "spikingformer":
        return SpikingFormer(
            args=config,
            num_classes=config.n_outputs,
            num_heads=config.num_heads,
            mlp_ratios=config.mlp_ratios,
        ).to(device)
    if model_name == "qkformer":
        return QKFormer(
            args=config,
            patch_size=16,
            embed_dims=256,
            num_heads=config.num_heads,
            mlp_ratios=config.mlp_ratios,
            in_channels=2,
            num_classes=config.n_outputs,
            qkv_bias=False,
            norm_layer=nn.LayerNorm,
            depths=4,
            sr_ratios=1,
        ).to(device)

    raise ValueError(f"Unknown model: {config.model_name}")


def split_to_train_test_set(
    train_ratio: float, origin_dataset, num_classes: int, random_split: bool = False
):

    if hasattr(origin_dataset, "targets"):
        labels = np.array(origin_dataset.targets)
    elif hasattr(origin_dataset, "_labels"):
        labels = np.array(origin_dataset._labels)
    elif hasattr(origin_dataset, "labels"):
        labels = np.array(origin_dataset.labels)
    else:
        labels = np.array([origin_dataset[i][1] for i in range(len(origin_dataset))])

    if labels.ndim > 1:
        labels = labels.squeeze()

    label_idx = [np.where(labels == i)[0].tolist() for i in range(num_classes)]

    train_idx, test_idx = [], []
    for i in range(num_classes):
        if random_split:
            np.random.shuffle(label_idx[i])
        pos = math.ceil(len(label_idx[i]) * train_ratio)
        train_idx.extend(label_idx[i][:pos])
        test_idx.extend(label_idx[i][pos:])

    return Subset(origin_dataset, train_idx), Subset(origin_dataset, test_idx)


import hashlib


def get_tensor_hash(tensor: torch.Tensor) -> str:
    """Hash the tensor for exact comparisons."""

    data = tensor.detach().cpu().numpy().tobytes()
    return hashlib.md5(data).hexdigest()


def load_data(config, dataset_dir, T, dataset_seed=312):

    original_seed = config.seed
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    print("Loading data...")

    np.random.seed(dataset_seed)
    torch.manual_seed(dataset_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dataset_seed)
    config.seed = dataset_seed

    st = time.time()

    if config.dataset.lower() == "dvs128":
        dataset_train = dvs128_gesture.DVS128Gesture(
            root=dataset_dir, train=True, data_type="frame", frames_number=T, split_by="number"
        )
        dataset_val = dvs128_gesture.DVS128Gesture(
            root=dataset_dir, train=False, data_type="frame", frames_number=T, split_by="number"
        )

    elif config.dataset.lower() == "cifar10dvs":
        origin_set = cifar10_dvs.CIFAR10DVS(
            root=dataset_dir, data_type="frame", frames_number=T, split_by="number"
        )
        dataset_train, dataset_val = split_to_train_test_set(0.9, origin_set, 10)

    elif config.dataset.lower() == "ncaltech101":
        origin_set = n_caltech101.NCaltech101(
            root=dataset_dir, data_type="frame", frames_number=T, split_by="number"
        )
        print(f"Data set finished, took {time.time() - st:.2f}s")
        dataset_train, dataset_val = split_to_train_test_set(0.9, origin_set, 101)

    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    print(f"Data split finished, took {time.time() - st:.2f}s")

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, "workers", 4),
        drop_last=True,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, "workers", 2),
        drop_last=False,
        pin_memory=True,
    )

    print(f"Data loader finished, took {time.time() - st:.2f}s")

    x, y = next(iter(train_loader))
    print(f"[DEBUG] Dataset seed verification:")
    print(f"  - Label hash:      {get_tensor_hash(y)}")
    print(f"  - Input hash:      {get_tensor_hash(x)}")
    print(f"  - First sample:    label={y[0].item()}")
    print(f"  - Workers:         {getattr(config, 'workers', 4)}")

    config.seed = original_seed
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)

    return train_loader, valid_loader


@torch.no_grad()
def eval_model(config, model, loader, device, phase="Eval"):
    model.eval()
    loss_list, metric_list, batch_weights = [], [], []

    use_amp = getattr(config, "amp", True)

    for image, target in loader:
        image = image.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True)

        if use_amp:
            with autocast():
                output = model(image)
                loss = nn.CrossEntropyLoss()(output, target)
        else:
            output = model(image)
            loss = nn.CrossEntropyLoss()(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_list.append(loss.detach().cpu().item())
        metric_list.append(acc1.item() / 100.0)
        batch_weights.append(len(target))

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
    aug,
    trival_aug,
    mixup_fn,
    scaler,
):
    """Train and evaluate the model across epochs."""
    loss_epochs = {"train": [], "valid": [], "test": []}
    metric_epochs = {"train": [], "valid": [], "test": []}
    best_metric_val = 0.0
    best_metric_test = 0.0

    if config.use_aug:
        criterion_train = SoftTargetCrossEntropy().to(device)
    else:
        criterion_train = nn.CrossEntropyLoss().to(device)

    use_amp = getattr(config, "amp", True)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        loss_batch, metric_batch = [], []

        if mixup_fn is not None and hasattr(mixup_fn, "mixup_enabled"):
            mixup_off_epoch = getattr(config, "mixup_off_epoch", 75)
            if epoch >= mixup_off_epoch:
                mixup_fn.mixup_enabled = False

        num_batches = len(train_loader)
        for i, (image, target) in enumerate(train_loader, start=1):
            image = image.to(device).float()
            target = target.to(device)

            # [N, T, C, H, W]
            N, T, C, H, W = image.shape

            if aug is not None:
                image = torch.stack([aug(image[j]) for j in range(N)])

            if trival_aug is not None:
                image = torch.stack([trival_aug(image[j]) for j in range(N)])

            if mixup_fn is not None:
                image, target = mixup_fn(image, target)
                target_for_acc = target.argmax(dim=-1)
            else:
                target_for_acc = target

            if use_amp:
                with autocast():
                    output = model(image)
                    loss = criterion_train(output, target)
            else:
                output = model(image)
                loss = criterion_train(output, target)

            optimizer.zero_grad()

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            acc1, acc5 = accuracy(output, target_for_acc, topk=(1, 5))
            metric = acc1.item() / 100.0

            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)

            if i % 100 == 0 or i == num_batches:
                progress = 100.0 * i / num_batches

                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"{progress:.1f}% "
                    f"({i}/{num_batches}) "
                    f"Loss={loss.item():.4f} "
                    f"Acc={metric * 100:.2f}%"
                )

        model.print_drop_stats()
        model.reset_drop_stats()

        scheduler.step(epoch + 1)

        loss_epochs["train"].append(float(np.mean(loss_batch)))
        metric_epochs["train"].append(float(np.mean(metric_batch)))

        loss_valid, metric_valid = eval_model(config, model, valid_loader, device, "Valid")
        loss_epochs["valid"].append(loss_valid)
        metric_epochs["valid"].append(metric_valid)

        if test_loader is not None:
            loss_test, metric_test = eval_model(config, model, test_loader, device, "Test")
        else:
            loss_test, metric_test = 100.0, 0.0

        loss_epochs["test"].append(loss_test)
        metric_epochs["test"].append(metric_test)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss={loss_epochs['train'][-1]:.4f} Acc={100 * metric_epochs['train'][-1]:.2f}% | "
            f"Valid Loss={loss_valid:.4f} Acc={100 * metric_valid:.2f}% | "
            f"Test Loss={loss_test:.4f} Acc={100 * metric_test:.2f}% | "
            f"LR={optimizer.param_groups[0]['lr']:.6f} | "
            f"Time={time.time() - epoch_start:.1f}s"
        )

        checkpoint_dir = output_dirs["checkpoints"]

        if metric_valid > best_metric_val:
            best_metric_val = metric_valid
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "Best_ACC_VAL.pth"))
            logger.info(f"  -> Saved best ACC VAL model ({100 * best_metric_val:.2f}%)")

        if test_loader is not None and metric_test > best_metric_test:
            best_metric_test = metric_test
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "Best_ACC_TEST.pth"))
            logger.info(f"  -> Saved best ACC TEST model ({100 * best_metric_test:.2f}%)")

        logger.info(
            f"Best Valid Acc: {100 * best_metric_val:.2f}% | Best Test Acc: {100 * best_metric_test:.2f}%"
        )

    return loss_epochs, metric_epochs


def setup_device(config, logger):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device {device}")
    return device


def build_mixup(config, logger):
    if not config.use_aug:
        return None

    if isinstance(config.cutmix_minmax, str) and config.cutmix_minmax.lower() == "none":
        config.cutmix_minmax = None

    logger.info("Using augmentation")
    logger.info(
        f"Mixup: {config.mixup} | Cutmix: {config.cutmix} | cutmix_minmax: {config.cutmix_minmax}"
    )
    logger.info(f"Mixup prob: {config.mixup_prob} | Mixup switch prob: {config.mixup_switch_prob}")
    logger.info(f"Mixup mode: {config.mixup_mode}")
    logger.info(f"Smoothing: {config.smoothing}")

    mixup_args = dict(
        mixup_alpha=config.mixup,
        cutmix_alpha=config.cutmix,
        cutmix_minmax=config.cutmix_minmax,
        prob=config.mixup_prob,
        switch_prob=config.mixup_switch_prob,
        mode=config.mixup_mode,
        label_smoothing=config.smoothing,
        num_classes=config.n_outputs,
    )
    return Mixup(**mixup_args)


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

    # Feed resolved values to timm, not nested YAML dictionaries or None.
    args.epochs = config.epochs
    args.lr = config.lr
    args.weight_decay = config.weight_decay
    args.warmup_epochs = config.warmup_epochs

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

        seed_everything(seed)

        logger.info(f"[TIME] Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"The GPU is {config.gpu} | Device = {device}")
        logger.info(f"Seed: {seed} | Epochs: {config.epochs} | Batch: {config.batch_size}")
        logger.info(f"Model: {config.model_name} | Dataset: {config.dataset}")

        log_drop_config(logger, config)

        train_snn_aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
        trival_aug = autoaugment.SNNAugmentWide()

        mixup_fn = build_mixup(config, logger)

        model = create_model(config, device)
        logger.info(f"Model parameters: {count_parameters(model):,}")
        logger.info(str(model))

        optimizer = create_optimizer(args, model)

        print("\n===== OPTIMIZER CONFIG =====")
        print(type(optimizer))
        print("defaults:", optimizer.defaults)
        for i, group in enumerate(optimizer.param_groups):
            print(f"\nParam group {i}")
            for k, v in group.items():
                if k != "params":
                    print(k, ":", v)
            print("num params:", len(group["params"]))

        if args.amp:
            scaler = GradScaler()
        else:
            scaler = None
        scheduler, num_epochs = create_scheduler(args, optimizer)

        print("\n===== SCHEDULER CONFIG =====")
        print(type(scheduler))
        print("num_epochs:", num_epochs)
        print("state_dict:", scheduler.state_dict())

        print("\n===== LR SCHEDULE =====")
        for epoch in range(num_epochs):
            scheduler.step(epoch)
            print(epoch, optimizer.param_groups[0]["lr"])

        config.save(os.path.join(output_dirs["configs"], f"config_{run_tag}.yaml"))

        data_path = config.datasets_path
        train_loader, valid_loader = load_data(config, data_path, T=config.time_steps)
        test_loader = None

        logger.info(f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")
        logger.info(f"The time_steps is: {config.time_steps}")

        loss_epochs, metric_epochs = train_model(
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
            train_snn_aug,
            trival_aug,
            mixup_fn,
            scaler,
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
