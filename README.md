# SeMuOC

Code for time-series anomaly detection with temporal and behavior-semantic representations, reconstruction, and a fixed-center one-class branch.

The model, Python entry points, configuration package, study identifiers, and output directories consistently use `SeMuOC`.

## Setup

Use Python 3.12. Install a PyTorch build appropriate for your CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Training uses the frozen `intfloat/e5-base-v2` text encoder. It is downloaded on first use, or can be placed in `embedding/e5-base-v2`. The `hash` encoder is intended for offline structural checks and does not reproduce the reported E5 experiments.

`environment-observed.txt` records package versions from the preparation environment. A fresh installation and full GPU training have not been validated for this source export.

## Data

Place processed datasets under `data/`: `iops_competition`, `asd`, `exathlon`, and `wadi`. Use the data formats expected by `dataloader/data_preprocessing.py`. Some datasets have access restrictions; raw data and model weights are not distributed here.

## Training

```bash
python SeMuOC.py --selected_dataset ASD --device cuda --seed 1
bash run_semuoc.sh ASD cuda
```

The shell entry runs seeds 1–5 sequentially. Use `--device cpu` in the Python entry for CPU execution. Dataset defaults are defined in `conf/semuoc`; available arguments are listed by `python SeMuOC.py --help`.

Model checkpoints, diagnostic scores and metric summaries are saved in `best_network/`, `scores/` and `results/`. These generated outputs are excluded from Git.

## Dependencies and attribution

See [THIRD_PARTY.md](THIRD_PARTY.md). Copyright and license headers are retained in bundled third-party files.

## Model package

```text
models/
├── SeMuOC/
│   ├── model/
│   │   ├── __init__.py
│   │   └── semuoc.py
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── early_stopping.py
│   │   └── trainer.py
│   └── __init__.py
├── __init__.py
└── anomaly_predict.py
```

`semuoc.py` contains the full model; `trainer/` contains training and evaluation routines.
