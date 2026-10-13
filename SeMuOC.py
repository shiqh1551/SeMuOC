import argparse
import importlib
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from merlion.evaluate.anomaly import TSADScoreAccumulator as ScoreAcc, ScoreType
from tqdm import tqdm

from models.SeMuOC.model.semuoc import base_Model
from models.SeMuOC.trainer.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=6, type=int)
    parser.add_argument("--selected_dataset", default="ASD",
                        choices=["KPI", "IOpsCompetition", "ASD", "Exathlon", "WADI"])
    parser.add_argument("--selected_model", default="SeMuOC", choices=["SeMuOC"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def nanmean(values):
    values = np.asarray(values, dtype=float)
    return np.nan if not values.size or np.all(np.isnan(values)) else np.nanmean(values)


def weighted_mean(values, weights):
    weights = np.asarray(weights, dtype=float)
    return np.nansum(np.asarray(values) * weights / weights.sum()) if weights.sum() else np.nan


def main():
    args = parse_args()
    start_time = datetime.now()
    dataset = "IOpsCompetition" if args.selected_dataset == "KPI" else args.selected_dataset
    configs = importlib.import_module(f"conf.semuoc.{dataset}_Configs").Config()
    device = torch.device(args.device)

    from dataloader import data_generator1, other_datasets, wadi, load_dataset
    from utils import print_object

    dt = None if dataset == "WADI" else load_dataset(dataset)
    model_num = 1 if dt is None else len(dt)
    scores = {name: [] for name in ["rpa", "pa", "pw"]}
    continuous = {name: [] for name in ["Test ROC-AUC", "Test PR-AUC", "Test VUS-ROC", "Test VUS-PR"]}
    anomaly_windows, aff_precision, aff_recall = [], [], []
    print(f"Dataset: {dataset}\nMethod: {args.selected_model}\nRandom Seed: {args.seed}")

    for idx in tqdm(range(model_num)):
        set_seed(args.seed)
        data = wadi() if dataset == "WADI" else other_datasets(*dt[idx])
        train_dl, val_dl, test_dl, anomaly_num = data_generator1(*data, args.seed, configs)
        model = base_Model(configs, device).to(device)
        optimizer = torch.optim.Adam(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=configs.weight,
        )
        (_, aff, rpa, pa, pw, _, roc, pr, vus_roc, vus_pr) = Trainer(
            model, optimizer, train_dl, val_dl, test_dl, device, configs, idx, args.seed,
        )
        anomaly_windows.append(anomaly_num)
        aff_precision.append(aff["precision"])
        aff_recall.append(aff["recall"])
        for name, value in zip(scores, [rpa, pa, pw]):
            scores[name].append(value)
        for name, value in zip(continuous, [roc, pr, vus_roc, vus_pr]):
            continuous[name].append(value)

    precision = weighted_mean(aff_precision, anomaly_windows)
    recall = weighted_mean(aff_recall, anomaly_windows)
    aff_f1 = 0.0 if np.isnan(precision + recall) or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    totals = {name: sum(scores[name], ScoreAcc()) for name in ["rpa", "pa", "pw"]}
    results = {
        "Hyper-parameter": print_object(configs), "Polluted-rate": 0.0, "seed": args.seed,
        "Train Time": datetime.now() - start_time,
        "Affiliation Precision": round(precision, 5), "Affiliation Recall": round(recall, 5),
        "Affiliation F1": round(aff_f1, 5),
    }
    for name, score_type, prefix in [
        ("rpa", ScoreType.RevisedPointAdjusted, "RPA"),
        ("pa", ScoreType.PointAdjusted, "PA"),
        ("pw", ScoreType.Pointwise, "Point-wise"),
    ]:
        for label, method in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]:
            results[f"{prefix} {label}"] = round(getattr(totals[name], method)(score_type), 5)
    results.update({
        "NAB Score (balanced)": round(totals["rpa"].nab_score(), 5),
        "NAB Score (high precision)": round(totals["rpa"].nab_score(fp_weight=0.22), 5),
        "NAB Score (high recall)": round(totals["rpa"].nab_score(fn_weight=2.0), 5),
        **{name: round(nanmean(values), 5) for name, values in continuous.items()},
    })
    for name in ["Test VUS-ROC", "Test VUS-PR", "RPA F1"]:
        print(f"{name}: {results[name]:.5f}")

    Path("results").mkdir(exist_ok=True)
    summary = Path("results") / f"{args.selected_model}_{dataset}_summary.csv"
    frame = pd.read_csv(summary, index_col=0).astype(object) if summary.exists() else pd.DataFrame(dtype=object)
    frame.drop(index="UCR Accuracy", errors="ignore", inplace=True)
    frame[f"{args.selected_model}{frame.shape[1]}"] = pd.Series(results, dtype=object)
    frame.to_csv(summary)
    print(f"Saved: {summary}")
    return results


if __name__ == "__main__":
    main()
