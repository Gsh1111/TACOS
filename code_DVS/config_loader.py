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
        apply_profile(self, 'dvs')
        self._compute_derived()

    def _set_defaults(self):
        """Set default configuration values"""
        # General
        self.dataset = 'dvs128'
        self.datasets_path = 'Datasets/DVS128'
        self.log_dir = './logs'
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
        self.model_name = 'spikformer'

        # Input dimension hyperparams (DVS128)
        self.time_steps = 16
        self.n_bins = 5

        # Model shape hyperparams
        self.depths = 2
        self.num_heads = 16
        self.n_outputs = 11
        self.mlp_ratios = 4

        # Spiking neuron hyperparams
        self.init_tau = 2.0
        self.v_threshold = 1.0
        self.v_reset = 0.5
        self.surrogate_function_type = 'atan'
        self.alpha = 5.0
        self.detach_reset = True

        # Module flags
        self.use_dw_bias = False
        self.pooling_stat = "0111"

        # Augmentation arguments
        self.use_aug = True
        self.mixup_off_epoch = 75
        self.mixup = 0.5
        self.cutmix = 0.
        self.cutmix_minmax = None
        self.mixup_prob = 0.5
        self.mixup_switch_prob = 0.5
        self.mixup_mode = "batch"
        self.smoothing = 0.2
        
        # Training arguments
        self.epochs = 300
        self.warmup_epochs = 10
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
        self.drop_comp_window = 4    
        
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
                    if k == "lr_w": k = "lr"
                    if hasattr(self, k):
                        setattr(self, k, v)
            else:
                if hasattr(self, section):
                    setattr(self, section, values)

    def _load_from_args(self, args):
        """Load configuration from argparse arguments"""
        d = vars(args)

        # Added cutmix_minmax to list_keys
        list_keys = {'seed_list', 'drop_windows', 'drop_after_layers', 'cutmix_minmax'}
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

        if self.drop_target != 'feature' or not self.drop_sequence:
            self.drop_after_layers = []
        else:
            if self.drop_after_layers is None:
                raise ValueError(f"drop_after_layers is None!")
            else:
                self.drop_after_layers = sorted(set(self.drop_after_layers))

        # Set surrogate function
        sft = str(self.surrogate_function_type).lower()
        if sft == 'sigmoid':
            self.surrogate_function = surrogate.Sigmoid(alpha=float(self.alpha))
        elif sft == 'atan':
            self.surrogate_function = surrogate.ATan(alpha=float(self.alpha))
        else:
            raise ValueError(f"Unknown surrogate_function_type: {self.surrogate_function_type}")

        # Set n_outputs based on dataset
        if self.dataset == 'dvs128':
            self.n_outputs = 11
        elif self.dataset == 'cifar10dvs':
            self.n_outputs = 10
        elif self.dataset == 'ncaltech101':
            self.n_outputs = 101
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
                f"{layers}{win}"
            )

        self.run_info = (
            f'||{self.dataset}'
            f'||{self.depths}depths'
            f'||{self.time_steps}time_steps'
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
        model_keys = ['model_name', 'time_steps', 'depths', 'num_heads', 'n_outputs', 
                      'init_tau', 'v_threshold', 'v_reset', 'surrogate_function_type',
                      'alpha', 'detach_reset']
        aug_keys = ['use_aug', 'mixup_off_epoch', 'mixup', 'cutmix', 'cutmix_minmax',
                    'mixup_prob', 'mixup_switch_prob', 'mixup_mode', 'smoothing']
        training_keys = ['epochs', 'warmup_epochs', 'batch_size', 'keep_ratio', 'loss', 'loss_fn']
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
    parser.add_argument('--dataset', type=str, choices=['cifar10dvs', 'dvs128', 'ncaltech101'], help='Dataset to use')
    parser.add_argument('--datasets-path', dest='datasets_path', type=str, help='Path to datasets directory')
    parser.add_argument('--log-dir', dest='log_dir', type=str, help='Directory for logging')
    parser.add_argument('--gpu', type=int, help='GPU device ID')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--seed-list', dest='seed_list', nargs='+', type=int, help='Seed list for outer-loop runs')
    parser.add_argument('--backend', type=str, choices=['cupy', 'torch'], help='Backend for computation')
    _bool_optional(parser, '--make-plot', help='Generate accuracy plots')

    # Model selection
    parser.add_argument('--model-name', dest='model_name', type=str,
                        choices=['sdt', 'spike_driven_transformer', 'spikformer', 'spikingformer', 'qkformer'],
                        help='Model architecture to use')

    # Core hyperparams
    parser.add_argument('--duration', dest='duration', type=int, help='Duration in ms per step')
    parser.add_argument('--n-bins', dest='n_bins', type=int, help='Number of bins for temporal encoding')

    # Model shape hyperparams
    parser.add_argument('--depths', type=int, help='Number of blocks/layers')
    parser.add_argument('--attention-window', dest='attention_window', type=int, help='Sliding attention window size')
    parser.add_argument('--n-hidden-neurons', dest='n_hidden_neurons', type=int, help='Hidden dimension (single run)')
    parser.add_argument('--num-heads', dest='num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--mlp-ratios', dest='mlp_ratios', type=int, help='MLP expansion ratio')
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
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')    
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=None, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')

    #Learning rate scheduler
    parser.add_argument('--t-max', dest='t_max', type=int, help='T_max for cosine scheduler')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=None, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=20, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
    # Mixed precision training parameters
    parser.add_argument('--amp', default=True, action='store_true',
                        help='Use AMP training')
    
    parser.add_argument('--run-tag', type=str, default=None, help='Optional custom run identifier')
    parser.add_argument('--tacos-mode', choices=['full','half','tacos'], default=None)
    parser.add_argument('--tacos-profile', choices=['paper'], default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    _bool_optional(parser, '--allow-download', dest='allow_download')
    return parser
