import yaml
import argparse
import datetime
from spikingjelly.activation_based import surrogate


def _bool_optional(parser: argparse.ArgumentParser, name: str, **kwargs):
    """
    Create a --foo / --no-foo flag pair (Python>=3.9 has BooleanOptionalAction).
    If your Python is older, we fallback to store_true (cannot disable from CLI).
    """
    try:
        action = argparse.BooleanOptionalAction  # py>=3.9
        parser.add_argument(name, action=action, default=None, **kwargs)
    except AttributeError:
        dest = kwargs.pop('dest', name[2:].replace('-', '_'))
        parser.add_argument(name, dest=dest, action='store_true', default=None, **kwargs)
        parser.add_argument('--no-' + name[2:], dest=dest, action='store_false')


class Config:
    """Configuration class that can be initialized from YAML and command-line arguments"""

    def __init__(self, config_path=None, args=None):
        self._set_defaults()

        if config_path:
            self._load_from_yaml(config_path)

        if args:
            self._load_from_args(args)

        from tacos_protocol import apply_profile
        apply_profile(self, 'speech')
        self._compute_derived()

    def _set_defaults(self):
        """Set default configuration values"""
        # General
        self.dataset = 'ssc'
        self.datasets_path = 'Datasets/SSC'
        self.log_dir = './logs/logging/ssc/5seed'
        self.gpu = 0
        self.seed = 42
        self.seed_list = [42, 43, 44, 45, 46]
        self.tacos_mode = None
        self.tacos_profile = "paper"
        self.num_workers = 0
        self.allow_download = False
        self.backend = 'cupy'
        self.make_plot = True

        # Model selection
        self.embed_projection = True
        self.model_name = 'spikformer'

        # Input dimension hyperparams (SSC/SHD)
        self.duration = 10
        self.n_bins = 5

        # Model shape hyperparams
        self.depths = 2
        self.attention_window = 20
        self.n_hidden_neurons = 256

        self.num_heads = 16
        self.mlp_ratio = 4
        self.dropout_l = 0.1
        self.dropout_p = 0.1
        self.kernel_size = 31
        self.n_outputs = 35
        self.bias = True

        # Spiking neuron hyperparams
        self.init_tau = 2.0
        self.v_threshold = 1.0
        self.v_reset = 0.5
        self.surrogate_function_type = 'atan'
        self.alpha = 5.0
        self.detach_reset = True

        # Module flags
        self.use_dw_bias = False

        # Augmentation arguments
        self.use_aug = True

        # SHD/SSC TimeNeurons_mask_aug
        self.TN_mask_aug_proba = 0.5
        self.time_mask_proportion = 0.1
        self.neuron_mask_size = 10

        # Training arguments
        self.epochs = 300
        self.batch_size = 256
        self.keep_ratio = 1.0
        self.loss = 'sum'
        self.loss_fn = 'CEloss'

        # Optimization
        self.optimizer_w = 'adamw'
        self.lr = 0.005
        self.weight_decay = 0.01
        self.t_max = 40

        # ================= Unified Drop =================
        # One unified drop config controlling both input drop and feature drop
        self.drop_sequence = False
        self.drop_target = 'feature'
        self.drop_after_layers = None       # List[int] | None
        
        self.drop_selector = 'topk'
        self.drop_ratio = 0.5

        # Score names
        self.drop_score = 'v_x_spike'

        self.drop_use_windows = False
        self.drop_windows = None         # only when drop_use_windows=True
        
        self.drop_compensation = False
        self.drop_comp_cap = None       
        self.drop_comp_window = 8   
        
        self.run_tag = None


    def _load_from_yaml(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}

        if not isinstance(cfg, dict):
            raise ValueError("YAML must be a mapping/dict.")

        for section, values in cfg.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
            else:
                if hasattr(self, section):
                    setattr(self, section, values)

    def _load_from_args(self, args):
        """Load configuration from argparse arguments"""
        d = vars(args)

        list_keys = {'seed_list', 'drop_windows', 'drop_after_layers'}
        for k, v in d.items():
            if v is None or not hasattr(self, k):
                continue
            # If someone passes comma string (rare with nargs='+', but keep safe)
            if k in list_keys and isinstance(v, (str, list)):
                if isinstance(v, str):
                    v = [float(x.strip()) if '.' in x else int(x.strip()) for x in v.split(',') if x.strip()]
                # If already list from argparse nargs, keep it but ensure types if needed
                setattr(self, k, v)
            else:
                setattr(self, k, v)

    def _compute_derived(self):
        """Compute derived attributes"""
        print("DEBUG dataset =", repr(self.dataset))
        if self.drop_target != 'feature' or not self.drop_sequence:
            self.drop_after_layers = []
        else:
            if self.drop_after_layers is None:
                raise ValueError(f"drop_after_layers is None!")
            else:
                self.drop_after_layers = sorted(set(self.drop_after_layers))

        # Compute n_inputs depends on dataset
        if self.dataset in ('shd', 'ssc'):
            self.n_inputs = 700 // int(self.n_bins)
        elif self.dataset == 'ntidigits':
            self.n_inputs = 64
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        # Compute hidden_dims
        self.hidden_dims = int(self.mlp_ratio * self.n_hidden_neurons)

        # Set surrogate function
        sft = str(self.surrogate_function_type).lower()
        if sft == 'sigmoid':
            self.surrogate_function = surrogate.Sigmoid(alpha=float(self.alpha))
        elif sft == 'atan':
            self.surrogate_function = surrogate.ATan(alpha=float(self.alpha))
        else:
            raise ValueError(f"Unknown surrogate_function_type: {self.surrogate_function_type}")

        # Set n_outputs based on dataset
        if self.dataset == 'shd':
            self.n_outputs = 20
        elif self.dataset == 'ssc':
            self.n_outputs = 35
        elif self.dataset == 'ntidigits':
            self.n_outputs = 11
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        # Generate run information
        self.current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_name = str(self.model_name)
        
        if self.drop_sequence and self.drop_use_windows:
            if self.drop_windows is None or len(self.drop_windows) == 0:
                raise ValueError("drop_use_windows=True but drop_windows is None/empty. "
                                "Please set --drop-windows (e.g., 2 4 8) or in YAML.")

        # Unified drop info in run tag
        drop_info = ""
        if self.drop_sequence:
            win = ""
            if self.drop_use_windows:
                win = f"_win{','.join(str(x) for x in self.drop_windows)}"
            layers = ""
            if self.drop_target == 'feature':
                layers = ",".join(str(x) for x in self.drop_after_layers)
            drop_info = (
                f"||drop={self.drop_target}"
                f"_{self.drop_selector}"
                f"_{self.drop_score}"
                f"@{self.drop_ratio:.2f}"
                f"_com" if self.drop_compensation else ""
                f"_{layers}{win}"
            )

        self.run_info = (
            f'||{self.dataset}'
            f'||{self.depths}depths'
            f'||{self.duration}ms'
            f'||bins={self.n_bins}'
            f'||lr={self.lr}'
            f'{drop_info}'
        )
        self.wandb_run_name = self.run_name + f'||seed={self.seed}' + self.run_info
        self.save_model_path = f'{self.wandb_run_name}_REPL_{self.current_time}.pt'

    def update(self, **kwargs):
        """Update configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._compute_derived()

    def print_config(self):
        """Print key configuration info"""
        print("=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        print(f"Dataset: {self.dataset}")
        print(f"Model: {self.model_name}")
        print(f"Depths: {self.depths} | Heads: {self.num_heads}")
        print(f"Hidden neurons: {self.n_hidden_neurons}")
        print("-" * 60)
        print(f"Drop enabled: {self.drop_sequence}")
        if self.drop_sequence:
            print(f"Drop target: {self.drop_target}")
            if self.drop_target == 'feature':
                print(f"Drop after layers: {self.drop_after_layers}")
            print(f"Drop selector: {self.drop_selector}")
            print(f"Drop ratio: {self.drop_ratio}")
            print(f"Drop score: {self.drop_score}")
            print(f"Drop use windows: {self.drop_use_windows}")
            print(f"Drop windows: {self.drop_windows}")
            print(f"Drop compensation: {self.drop_compensation}")
            print(f"Drop comp_cap:    {self.drop_comp_cap}")     
            print(f"Drop comp_window: {self.drop_comp_window}")  
            
        print("=" * 60)

    def save(self, path):
        """Save configuration to YAML file"""

        general_keys = ['dataset', 'datasets_path', 'log_dir', 'gpu', 'seed',
                        'seed_list', 'backend', 'make_plot', 'tacos_mode', 'tacos_profile', 'num_workers', 'allow_download']
        model_keys = ['model_name', 'embed_projection', 'duration', 'n_bins',
                      'n_inputs', 'depths', 'attention_window', 'n_hidden_neurons',
                      'num_heads', 'mlp_ratio', 'dropout_l',
                      'dropout_p', 'kernel_size', 'n_outputs', 'bias',
                      'init_tau', 'v_threshold', 'v_reset', 'surrogate_function_type',
                      'alpha', 'detach_reset']
        module_keys = ['use_dw_bias']
        aug_keys = ['use_aug', 'TN_mask_aug_proba', 'time_mask_proportion',
                    'neuron_mask_size']
        training_keys = ['epochs', 'batch_size', 'keep_ratio', 'loss', 'loss_fn']
        opt_keys = ['optimizer_w', 'lr', 'weight_decay', 't_max']

        drop_keys = [
            'drop_sequence', 'drop_target', 'drop_after_layers',
            'drop_selector', 'drop_ratio', 'drop_score',
            'drop_use_windows', 'drop_windows',
            'drop_compensation', 'drop_comp_cap', 'drop_comp_window',
        ]

        out = {
            'general': {k: getattr(self, k) for k in general_keys if hasattr(self, k)},
            'model': {k: getattr(self, k) for k in model_keys if hasattr(self, k)},
            'modules': {k: getattr(self, k) for k in module_keys if hasattr(self, k)},
            'augmentation': {k: getattr(self, k) for k in aug_keys if hasattr(self, k)},
            'training': {k: getattr(self, k) for k in training_keys if hasattr(self, k)},
            'optimization': {k: getattr(self, k) for k in opt_keys if hasattr(self, k)},
            'drop': {k: getattr(self, k) for k in drop_keys if hasattr(self, k)},
        }

        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)


def get_argparser():
    """Create argument parser with all configuration options"""
    parser = argparse.ArgumentParser(description='Spike-driven Transformer Training')

    # YAML
    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')

    # General
    parser.add_argument('--dataset', type=str, choices=['shd', 'ssc', 'ntidigits'], help='Dataset to use')
    parser.add_argument('--datasets-path', dest='datasets_path', type=str, help='Path to datasets directory')
    parser.add_argument('--log-dir', dest='log_dir', type=str, help='Directory for logging')
    parser.add_argument('--gpu', type=int, help='GPU device ID')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--seed-list', dest='seed_list', nargs='+', type=int, help='Seed list for outer-loop runs')
    parser.add_argument('--backend', type=str, choices=['cupy', 'torch'], help='Backend for computation')
    _bool_optional(parser, '--make-plot', help='Generate accuracy plots')

    # Model selection
    parser.add_argument('--model-name', dest='model_name', type=str,
                        choices=['spikcommander', 'spikescr', 'spikformer_tim', 'spikformer', 'qkformer', 'sdt'],
                        help='Model architecture to use')

    # Core hyperparams
    parser.add_argument('--duration', dest='duration', type=int, help='Duration in ms per step')
    parser.add_argument('--n-bins', dest='n_bins', type=int, help='Number of bins for temporal encoding')

    # Model shape hyperparams
    parser.add_argument('--depths', type=int, help='Number of blocks/layers')
    parser.add_argument('--attention-window', dest='attention_window', type=int, help='Sliding attention window size')
    parser.add_argument('--n-hidden-neurons', dest='n_hidden_neurons', type=int, help='Hidden dimension (single run)')
    parser.add_argument('--num-heads', dest='num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--mlp-ratio', dest='mlp_ratio', type=int, help='MLP expansion ratio')
    parser.add_argument('--dropout-l', dest='dropout_l', type=float, help='Dropout rate for first layer')
    parser.add_argument('--dropout-p', dest='dropout_p', type=float, help='Dropout rate for attention')
    parser.add_argument('--kernel-size', dest='kernel_size', type=int, help='DW conv kernel size')
    _bool_optional(parser, '--bias', help='Use bias in conv/linear')

    # Spiking neuron hyperparams
    parser.add_argument('--init-tau', dest='init_tau', type=float, help='Neuron initialized tau')
    parser.add_argument('--v-threshold', dest='v_threshold', type=float, help='Neuron threshold')
    parser.add_argument('--v-reset', dest='v_reset', type=float, help='Neuron reset voltage')
    parser.add_argument('--surrogate-function-type', dest='surrogate_function_type', choices=['atan', 'sigmoid'], help='Surrogate gradient type')
    parser.add_argument('--alpha', type=float, help='Alpha for surrogate gradient')
    _bool_optional(parser, '--detach-reset', help='Detach reset in spiking neuron')

    # Module flags
    _bool_optional(parser, '--use-dw-bias', help='Use bias in depthwise conv')

    # Augmentation arguments
    _bool_optional(parser, '--use-aug', dest='use_aug', help='Enable data augmentation')

    # SHD/SSC TimeNeurons mask aug
    parser.add_argument('--TN-mask-aug-proba', dest='TN_mask_aug_proba', type=float, help='Time-Neuron mask augmentation probability')
    parser.add_argument('--time-mask-proportion', dest='time_mask_proportion', type=float, help='Proportion of time steps to mask')
    parser.add_argument('--neuron-mask-size', dest='neuron_mask_size', type=int, help='Neuron mask size')

    # Training arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Batch size')
    parser.add_argument('--keep-ratio', dest='keep_ratio', type=float, help='Subsample ratio (1.0 = no subsample)')
    parser.add_argument('--loss', type=str, choices=['mean', 'max', 'spike_count', 'sum'], help='Loss aggregation')
    parser.add_argument('--loss-fn', dest='loss_fn', type=str, help='Loss function type')

    # Optimization
    parser.add_argument('--optimizer-w', dest='optimizer_w', type=str, help='Optimizer for weights')
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, help='Weight decay')
    parser.add_argument('--t-max', dest='t_max', type=int, help='T_max for cosine scheduler')

    parser.add_argument('--run-tag', type=str, default=None, help='Optional custom run identifier')
    parser.add_argument('--tacos-mode', choices=['full','half','tacos'], default=None)
    parser.add_argument('--tacos-profile', choices=['paper'], default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    _bool_optional(parser, '--allow-download', dest='allow_download')
    return parser
