# TACOS

This guide runs TACOS training on CIFAR10-DVS and SHD. Run the commands from the repository root on Linux. Training uses an NVIDIA GPU.

## Set up the environment

The following environment uses Python 3.10 and the CUDA 12.8 PyTorch wheels. It requires a compatible NVIDIA driver.

```bash
git clone https://github.com/Gsh1111/TACOS.git
cd TACOS
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install timm==1.0.25 cupy-cuda12x==14.0.1 h5py numpy scipy matplotlib pyyaml tqdm
python -m pip install 'git+https://github.com/fangwei123456/spikingjelly.git@9b20afa7028f1bc2e0dcacdeaf0f805c90cb3999'
```

Check that PyTorch and CuPy can access the GPU and that the speech dataset loader is available.

```bash
python -c "import cupy, torch; from spikingjelly.datasets.shd import SpikingSpeechCommands; assert torch.cuda.is_available() and cupy.cuda.runtime.getDeviceCount() > 0; print(torch.cuda.get_device_name(0))"
```

## Prepare the data

Set each variable to the existing, writable SpikingJelly dataset root for that dataset. The DVS loader reads CIFAR10-DVS event recordings and can cache frames there. The SHD loader uses the SHD train and test data. Use separate output directories for each training run.

```bash
export DVS_DATA=/absolute/path/to/CIFAR10DVS
export SHD_DATA=/absolute/path/to/SHD
export RUNS="$PWD/outputs"
```

## Preview and run training

Each command below prints the resolved configuration, seed list and training command. It starts training only when you append `--execute`. The path supplied to `--output` must be new when training starts.

```bash
python paper_run.py dvs qkformer tacos \
  --data "$DVS_DATA" --output "$RUNS/dvs-qkformer-42" --gpu 0 --seeds 42

python paper_run.py speech spikescr tacos \
  --data "$SHD_DATA" --output "$RUNS/shd-spikescr-42" --gpu 0 --seeds 42
```

For example, start the SHD run with:

```bash
python paper_run.py speech spikescr tacos \
  --data "$SHD_DATA" --output "$RUNS/shd-spikescr-42" --gpu 0 --seeds 42 --execute
```

For multiple seeds, pass them together, for example `--seeds 42 43 44`. The first argument selects the data pipeline and the second selects the backbone:

| Pipeline | Example dataset | Backbone names |
| --- | --- | --- |
| `dvs` | `cifar10dvs` | `qkformer`, `spikformer`, `spikingformer`, `sdt` |
| `speech` | `shd` | `spikescr`, `qkformer`, `spikformer`, `spikcommander`, `sdt` |

The example configurations live in [the CIFAR10-DVS directory](code_DVS/configs/cifar10dvs/16T/) and [the SHD directory](code_speech/configs/T=100/SHD/). To use a different dataset, pass `--dataset` with a name that has a matching configuration and set `--data` to its root.

The speech presets record whether the embedding uses a second linear projection and batch normalization in `model.embed_projection`. Use a checkpoint trained with the same architecture when resuming a run.

## Check the run

Training writes per-seed logs, resolved configuration files and checkpoints under `<output>/runs/`. To list the artifacts for the SHD example:

```bash
find "$RUNS/shd-spikescr-42/runs" -type f \( -name '*.log' -o -name '*.yaml' -o -name '*.pth' \)
```
