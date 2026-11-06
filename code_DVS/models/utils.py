import torch

    
def log_drop_event(log_state, context_name, x_len_before, x_len_after, T_before=None, T_after=None):
    """Print the header once and log each pruning event."""
    if not log_state["enabled"]:
        return

    if not log_state["started"]:
        print("\n====== Drop Strategy Verification (First forward only) ======")
        log_state["started"] = True

    buniq = torch.unique(x_len_before).detach().cpu().tolist()
    auniq = torch.unique(x_len_after).detach().cpu().tolist()

    extra = ""
    if T_before is not None and T_after is not None:
        extra = f" | T: {T_before} -> {T_after}"

    print(f"[{context_name}]{extra}")
    print(f"  x_len BEFORE (unique): {buniq}")
    print(f"  x_len AFTER  (unique): {auniq}")


def print_tensor_stats_before_drop(x, printed_flag=False):
    """Report feature statistics before temporal pruning."""
    if printed_flag:
        return

    T_before = x.shape[0]

    print(f"\n{'=' * 60}")
    print(
        f"[Drop] BEFORE  shape={tuple(x.shape)}  "
        f"mean={x.float().mean():.4f}  "
        f"std={x.float().std():.4f}  "
        f"sparsity={((x == 0).float().mean() * 100):.1f}%"
    )

    
    step_mean = x.float().abs().mean(dim=(1, 2))  # [T]
    max_val = step_mean.max().item() + 1e-8

    for t in range(T_before):
        bar = "█" * int(step_mean[t].item() * 40 / max_val)
        print(f"  t={t:2d}  mean_act={step_mean[t]:.4f}  {bar}")


def print_tensor_stats_after_drop(x, T_before, printed_flag=False):
    """Report feature statistics after temporal pruning."""
    if printed_flag:
        return

    T_after = x.shape[0]
    print(
        f"\n[Drop] AFTER   shape={tuple(x.shape)}  "
        f"mean={x.float().mean():.4f}  "
        f"std={x.float().std():.4f}  "
        f"sparsity={((x == 0).float().mean() * 100):.1f}%"
    )
    print(
        f"[Drop] {T_before} → {T_after} steps  "
        f"(dropped {T_before - T_after} / {T_before})"
    )
    print('=' * 60 + '\n')


def ensure_drop_stats_initialized(module, T_before):
    """Initialize the per-step drop counters."""
    if not hasattr(module, '_drop_count'):
        module._drop_count = torch.zeros(T_before, dtype=torch.long)
        module._drop_total = 0
        module._T_max = T_before
        module._batch_count = 0


def update_drop_stats(module, kept_indices, batch_size):
    """Update per-step drop counts from the retained indices."""
    all_indices = set(range(module._T_max))

    module._drop_total += batch_size
    module._batch_count += 1

    if isinstance(kept_indices, torch.Tensor):
        if kept_indices.dim() == 1:
            kept_set = set(kept_indices.cpu().tolist())
            dropped = sorted(all_indices - kept_set)
            for t in dropped:
                module._drop_count[t] += batch_size

        elif kept_indices.dim() == 2:
            for b in range(kept_indices.shape[0]):
                kept_set = set(kept_indices[b].cpu().tolist())
                dropped = all_indices - kept_set
                for t in dropped:
                    module._drop_count[t] += 1


def print_accumulated_drop_stats(module):
    """Report the cumulative drop probability for each time step."""
    if not hasattr(module, '_drop_count'):
        print("[Drop Stats] No observations")
        return

    total = max(module._drop_total, 1)
    T = module._T_max

    print(f"\n{'─' * 60}")
    print(f"[Drop Stats] Observed {total} samples  T={T}")
    print(f"{'─' * 60}")
    print(f"  {'Index':>6}  {'Drop count':>8}  {'Drop rate':>9}  {'Rate':}")
    print(f"  {'─' * 6}  {'─' * 8}  {'─' * 9}  {'─' * 30}")

    for t in range(T):
        cnt = module._drop_count[t].item()
        prob = cnt / total
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        flag = "  high" if prob > 0.7 else ("  low" if prob < 0.1 else "")
        print(f"  {t:>6}  {cnt:>8}  {prob:>8.1%}  {bar}{flag}")

    dropped_any = (module._drop_count > 0).sum().item()
    print(f"{'─' * 60}")
    print(f"  Steps dropped at least once: {dropped_any}/{T}")
    print(f"  Mean drop rate:     {module._drop_count.float().mean().item() / total:.1%}")


def reset_drop_stats(module):
    """Reset the accumulated drop statistics."""
    if hasattr(module, '_drop_count'):
        module._drop_count.zero_()
        module._drop_total = 0
        module._batch_count = 0
    module.has_printed_drop_info = False