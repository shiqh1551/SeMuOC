# SeMuOC

Code for time-series anomaly detection with temporal and behavior-semantic representations, reconstruction, and a fixed-center one-class branch.

The model, Python entry points, configuration package, study identifiers, and output directories consistently use `SeMuOC`.

## Setup

Use Python 3.12. Install a PyTorch build appropriate for your CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Training uses the frozen `intfloat/e5-base-v2` text encoder. It is downloaded on first use, or can be placed in `embedding/e5-base-v2`. The `hash` encoder is intended for offline structural checks and does not reproduce the reported E5 experiments.

## Data

Place processed datasets under `data/`: `iops_competition`, `asd`, `exathlon`, and `wadi`. Use the data formats expected by `dataloader/data_preprocessing.py`. Some datasets have access restrictions; raw data and model weights are not distributed here.

## Training

```bash
python SeMuOC.py --selected_dataset ASD --device cuda --seed 1
```

Use `--device cpu` for CPU execution. To run seeds 1–5 sequentially:

```bash
for seed in 1 2 3 4 5; do
    python SeMuOC.py --selected_dataset ASD --device cuda --seed "$seed" || break
done
```

Replace `ASD` with `KPI`, `Exathlon`, or `WADI` to select another dataset. Dataset defaults are defined in `conf/semuoc`; available arguments are listed by `python SeMuOC.py --help`. The four configuration files use the exact parameters of the five-seed results reported for SeMuOC: KPI v10/trial 12, ASD v11/trial 2, Exathlon v12/trial 1, and WADI v11/trial 5. Training uses seeds 1–5; the default seed 6 is not part of those reported runs.

Model checkpoints, diagnostic scores and metric summaries are saved in `best_network/`, `scores/` and `results/`. These generated outputs are excluded from Git.

## Dependencies and attribution

- `affiliation/` contains affiliation-based anomaly evaluation utilities. Upstream: https://github.com/ahstat/affiliation-metrics-py .
- RPA metrics use Salesforce Merlion; VUS metrics use the `vus` package.
- The frozen text encoder is `intfloat/e5-base-v2`: https://huggingface.co/intfloat/e5-base-v2 . Model weights are not included.

The bundled source inherits existing components from the local time-series anomaly detection codebase. No new blanket license is assigned by this export. Existing attribution and license notices must be preserved.

Copyright and license headers are retained in bundled third-party files.

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
