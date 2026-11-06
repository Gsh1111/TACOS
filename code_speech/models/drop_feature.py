from tacos_core import (
    _compute_base_score,
    _dice_sim_with_prev,
    _parse_windows,
    _to_btn,
    apply_compensation,
    apply_drop_feature,
    scores_multiscale_cover_sum_1d,
)
from .utils import (
    ensure_drop_stats_initialized,
    log_drop_event,
    print_tensor_stats_after_drop,
    print_tensor_stats_before_drop,
    update_drop_stats,
)


def compress_embedding(model, features, lengths, membrane, spikes):
    time_steps, batch_size, *feature_shape = features.shape
    flat_features = features.reshape(time_steps, batch_size, -1)
    flat_membrane = membrane.reshape_as(flat_features)
    flat_spikes = spikes.reshape_as(flat_features)
    already_logged = model.has_printed_drop_info
    print_tensor_stats_before_drop(flat_features, already_logged)
    ensure_drop_stats_initialized(model, time_steps)

    result = apply_drop_feature(
        model.config,
        flat_features,
        lengths,
        flat_membrane,
        return_indices=True,
        spike_seq=flat_spikes,
    )
    if model.config.drop_compensation:
        retained, query, retained_lengths, indices = result
    else:
        retained, retained_lengths, indices = result
        query = retained

    kept_steps = retained.shape[0]
    log_drop_event(
        {"enabled": not already_logged, "started": False},
        "after_embedding",
        lengths,
        retained_lengths,
        time_steps,
        kept_steps,
    )
    update_drop_stats(model, indices, batch_size)
    print_tensor_stats_after_drop(retained, time_steps, already_logged)
    model.has_printed_drop_info = True

    output_shape = (kept_steps, batch_size, *feature_shape)
    return retained.reshape(output_shape), query.reshape(output_shape), retained_lengths
