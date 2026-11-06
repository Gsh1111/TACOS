import os
from pathlib import Path


def apply_profile(config, domain):
    data_root = os.environ.get("TACOS_DATA_ROOT")
    if data_root:
        config.datasets_path = str(Path(data_root) / Path(config.datasets_path).name)

    mode = config.tacos_mode
    if not mode:
        if config.drop_sequence:
            mode = "tacos"
        else:
            mode = "full"
    config.tacos_mode = mode
    if config.tacos_profile != "paper":
        raise ValueError("Only the TACOS paper profile is supported")

    compress = mode == "tacos"
    config.drop_sequence = compress
    config.drop_target = "feature"
    if compress:
        config.drop_after_layers = [-1]
    else:
        config.drop_after_layers = []
    config.drop_selector = "topk"
    config.drop_score = "v_x_spike"
    config.drop_ratio = 0.5
    config.drop_use_windows = True
    config.drop_compensation = compress
    config.drop_comp_cap = None

    if domain == "dvs":
        config.drop_windows = [2, 4]
        config.drop_comp_window = 4
        if mode == "half":
            config.time_steps = 8
        else:
            config.time_steps = 16
        if config.model_name == "qkformer":
            config.mlp_ratios = 1
        else:
            config.mlp_ratios = 4
        if config.model_name in ("sdt", "spike_driven_transformer"):
            config.pooling_stat = "0111"
    else:
        config.drop_windows = [2, 4, 8]
        config.drop_comp_window = 8
        if config.dataset in ("shd", "ssc", "ntidigits"):
            if mode == "half":
                config.duration = 20
            else:
                config.duration = 10
