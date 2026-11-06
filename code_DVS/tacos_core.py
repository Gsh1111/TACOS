import torch
import torch.nn.functional as F


def _to_btn(x, x_len):
    if x.ndim != 3 or x.shape[1] != len(x_len):
        raise ValueError("Expected T,B,N")
    return x.transpose(0, 1).contiguous(), True


def _parse_windows(ws):
    if isinstance(ws, str):
        sizes = ws.replace(",", " ").split()
    else:
        sizes = ws
    return [int(size) for size in sizes]


def _dice_sim_with_prev(active, eps=1e-8):
    consistency = torch.ones(active.shape[:2], device=active.device, dtype=torch.float32)
    current = active[:, 1:]
    previous = active[:, :-1]
    intersection = (current & previous).sum(dim=-1).float()
    active_count = current.sum(dim=-1).float() + previous.sum(dim=-1).float()
    consistency[:, 1:] = 2 * intersection / (active_count + eps)
    return consistency


def _compute_base_score(x_btn, score_name="v_x_spike", thr=0.0, v_seq_btn=None):
    if score_name != "v_x_spike" or v_seq_btn is None:
        raise ValueError("MEAC requires binary spikes and pre-spike membrane potentials")
    membrane_energy = v_seq_btn.float().abs().mean(dim=-1)
    spike_consistency = _dice_sim_with_prev(x_btn > 0)
    return 0.3 * membrane_energy + 0.7 * spike_consistency


def scores_multiscale_cover_sum_1d(sig_bt, ws=(2, 4, 8), pad_mode="replicate", normalize="w2"):
    if pad_mode != "replicate" or normalize != "w2":
        raise ValueError("MEAC uses replicate padding and squared window normalization")

    signal = sig_bt.float().unsqueeze(1)
    refined_scores = torch.zeros_like(sig_bt, dtype=torch.float32)
    for size in ws:
        width = int(size)
        if width < 1:
            raise ValueError("Window must be positive")
        left_pad = (width - 1) // 2
        padding = (left_pad, width - 1 - left_pad)
        kernel = signal.new_ones(1, 1, width)
        neighborhood = F.conv1d(F.pad(signal, padding, mode=pad_mode), kernel)
        coverage = F.conv1d(F.pad(neighborhood, padding, mode=pad_mode), kernel)
        refined_scores += coverage.squeeze(1) / (width * width)
    return refined_scores


def _gather_steps(sequence, indices):
    feature_count = sequence.shape[-1]
    return sequence.gather(1, indices.unsqueeze(-1).expand(-1, -1, feature_count))


def apply_compensation(x_orig, kept_idx, x_drop, x_len, cap=None, window=8):
    if cap is not None:
        raise ValueError("CQA uses a bounded time window without an amplitude cap")

    steps = torch.arange(x_orig.shape[1], device=x_orig.device)
    valid_steps = steps.unsqueeze(0) < x_len.unsqueeze(1)
    retained_steps = torch.zeros_like(valid_steps).scatter_(1, kept_idx, True)
    discarded = x_orig.masked_fill((~valid_steps | retained_steps).unsqueeze(-1), 0)
    prefix_sum = F.pad(discarded.float().cumsum(1), (0, 0, 1, 0))

    history_start = (kept_idx - int(window)).clamp(min=0)
    history_start[:, 1:] = torch.maximum(history_start[:, 1:], kept_idx[:, :-1])
    history = _gather_steps(prefix_sum, kept_idx) - _gather_steps(prefix_sum, history_start)
    compensated = x_drop + history.to(x_drop.dtype)
    valid_retained = kept_idx < x_len.unsqueeze(1)
    return compensated.masked_fill(~valid_retained.unsqueeze(-1), 0)


def _select_steps(raw_scores, lengths, windows, keep_count):
    scores = torch.full_like(raw_scores, -torch.inf)
    for sample, length in enumerate(lengths):
        end = int(length)
        valid_prefix = raw_scores[sample : sample + 1, :end]
        scores[sample, :end] = scores_multiscale_cover_sum_1d(valid_prefix, windows)[0]

    selected = scores.topk(keep_count, dim=1).indices
    return selected.sort(dim=1).values


def apply_drop_feature(config, x, x_len, v_seq=None, return_indices=False, spike_seq=None):
    use_compensation = bool(config.drop_compensation)
    if not config.drop_sequence or config.drop_ratio <= 0:
        if use_compensation:
            result = (x, x, x_len)
        else:
            result = (x, x_len)
        if return_indices:
            return (*result, None)
        else:
            return result

    if x.ndim != 3 or x.shape[1] != len(x_len):
        raise ValueError("Expected T,B,N")
    if v_seq is None or v_seq.shape != x.shape:
        raise ValueError("Membrane/spike shapes must match")
    if not 0 <= config.drop_ratio < 1:
        raise ValueError("Invalid pruning ratio")

    timesteps = x.shape[0]
    if bool(((x_len < 1) | (x_len > timesteps)).any()):
        raise ValueError("Invalid input lengths")
    if spike_seq is None:
        scoring_spikes = x
    else:
        scoring_spikes = spike_seq
    if scoring_spikes.shape != x.shape:
        raise ValueError("Scoring spike shape mismatch")
    if not bool(((scoring_spikes == 0) | (scoring_spikes == 1)).all()):
        raise ValueError("MEAC consistency requires binary spikes")

    raw_scores = _compute_base_score(
        scoring_spikes.transpose(0, 1), v_seq_btn=v_seq.transpose(0, 1)
    )
    keep_count = max(1, round(timesteps * (1 - config.drop_ratio)))
    indices = _select_steps(raw_scores, x_len, config.drop_windows, keep_count)
    valid = indices < x_len.unsqueeze(1)
    features = x.transpose(0, 1)
    retained = _gather_steps(features, indices).masked_fill(~valid.unsqueeze(-1), 0)
    lengths = valid.sum(dim=1).clamp(min=1)
    output = retained.transpose(0, 1).contiguous()

    if use_compensation:
        query = apply_compensation(
            features, indices, retained, x_len, window=config.drop_comp_window
        )
        result = (output, query.transpose(0, 1).contiguous(), lengths)
    else:
        result = (output, lengths)
    if return_indices:
        return (*result, indices)
    else:
        return result
