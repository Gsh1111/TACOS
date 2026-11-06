"""Explicit paper defaults; historical experiments are not rewritten."""
import os
from pathlib import Path

def apply_profile(c,domain):
    root=os.environ.get('TACOS_DATA_ROOT')
    if root:c.datasets_path=str(Path(root)/Path(c.datasets_path).name)
    mode=c.tacos_mode or ('tacos' if c.drop_sequence else 'full')
    c.tacos_mode=mode
    if c.tacos_profile!='paper':raise ValueError('Only the TACOS paper profile is supported')
    c.drop_sequence=mode=='tacos';c.drop_target='feature';c.drop_after_layers=[-1] if c.drop_sequence else []
    c.drop_selector='topk';c.drop_score='v_x_spike';c.drop_ratio=.5
    c.drop_use_windows=True;c.drop_windows=[2,4] if domain=='dvs' else [2,4,8]
    c.drop_compensation=mode=='tacos';c.drop_comp_window=4 if domain=='dvs' else 8;c.drop_comp_cap=None
    if domain=='dvs':
        c.time_steps=8 if mode=='half' else 16
        # Inferred from the paper's 2.57M parameter count and baseline default.
        # QKFormer is 1.50M and retains ratio=1.
        c.mlp_ratios=1 if c.model_name=='qkformer' else 4
        if c.model_name in ['sdt','spike_driven_transformer']:c.pooling_stat='0111'
    elif c.dataset in ['shd','ssc','ntidigits']:
        c.duration=20 if mode=='half' else 10
