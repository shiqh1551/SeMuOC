# SeMuOC

Code for time-series anomaly detection with temporal and behavior-semantic representations, reconstruction, and a fixed-center one-class branch.

The implementation uses the internal name `SemSphere`. This name in Python modules and output directories refers to SeMuOC.

## Setup

Use Python 3.12. Install a PyTorch build appropriate for your CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Training uses the frozen `intfloat/e5-base-v2` text encoder. It is downloaded on first use, or can be placed in `embedding/e5-base-v2`. The `hash` encoder is intended for offline structural checks and does not reproduce the reported E5 experiments.

`environment-observed.txt` records package versions from the preparation environment. A fresh installation and full GPU training have not been validated for this source export.

## Data

Place processed datasets under `data/`: `iops_competition`, `asd`, `smd`, `exathlon`, `swat`, `wadi`, `esa`, and `ucr`. Use the data formats expected by `experiments/datasets.py`, `dataloader/dataset.py`, and the bundled `ts_datasets` loaders. Some datasets have access restrictions; raw data and model weights are not distributed here.

## Fixed-configuration experiments

Four reference configurations are included with their full parameter precision. They are snapshots of completed five-seed experiments, rather than the default search configurations. Keep the dataset, window, stride, and VUS tolerance consistent when comparing results.

```bash
python SemSphere_ablation_method.py \
  --selected_dataset ASD --baseline_config configs/reference/ASD.json \
  --study_name reference --ablation_case 06_full \
  --device cuda --seeds 1,2,3,4,5 --njobs 1
```

Replace `ASD` with `IOpsCompetition`, `Exathlon`, or `WADI` and use the corresponding JSON. CPU execution is supported with `--device cpu`. The five seeds run sequentially within each case. Completed seeds are skipped when rerunning the same unchanged configuration.

Results are saved under `results_SemSphere_ablation/<dataset>/<study>/<case>/`. Each case writes per-seed metrics and a `summary.csv`; scores and checkpoints are written to `scores/` and `best_network/` according to the configuration. Summary uncertainty is the population standard deviation across seeds. Primary metrics are VUS-ROC, VUS-PR, and RPA F1.

## Ablation experiments

```bash
python SemSphere_ablation_method.py \
  --selected_dataset ASD --baseline_config configs/reference/ASD.json \
  --study_name ablation_reference --run_all_ablation \
  --device cuda --seeds 1,2,3,4,5 --njobs 1
```

The four cases are `02_no_behavior`, `03_no_recon`, `04_no_oc`, and `06_full`. Removing reconstruction disables both its training loss and its contribution to the anomaly score. The combination cases are available with `--run_combination_ablation`. `--njobs` controls concurrent cases; each case processes its seeds sequentially. Use `--njobs 1` for memory-intensive WADI runs.

## Hyperparameter search

```bash
bash optuna/scripts/run_optuna_ASD.sh --device cuda --njobs 1
```

Dataset scripts are available for all eight datasets. `optuna/expanded` and `optuna/refined` provide additional search ranges. Each enabled script block runs a separate study; `--trials` adds trials on each invocation. Use distinct study names when changing the experimental protocol. Search outputs are stored in `results/`, `logs/`, and `optuna/semsphere_journals/`.

## Model and score

The full loss is the weighted sum of reconstruction, InfoNCE alignment, and one-class distance. The anomaly score combines batch-normalized reconstruction error and one-class distance; the training loss and detection score are distinct. Adam starts at `3e-4`, with weight decay `5e-4` and betas `(0.9, 0.99)`. A ReduceLROnPlateau scheduler monitors total training loss.

## Dependencies and attribution

See [THIRD_PARTY.md](THIRD_PARTY.md) for bundled loaders and evaluation utilities. Copyright and license headers are retained in third-party source files.
