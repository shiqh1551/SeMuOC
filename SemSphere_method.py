import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from merlion.evaluate.anomaly import TSADScoreAccumulator as ScoreAcc, ScoreType
from tqdm import tqdm

from models.SemSphere.network.model import base_Model
from models.SemSphere.trainer.trainer import Trainer
from models.reasonable_metric import reasonable_accumulator


def _parse_hpvalue(x):
    if isinstance(x, str):
        x = x.strip()
        for caster in (int, float):
            try:
                return caster(x)
            except ValueError:
                pass
    return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ed", "--experiment_description", default="Exp1")
    parser.add_argument("--run_description", default="run1")
    parser.add_argument("--seed", default=6, type=int)
    parser.add_argument(
        "--selected_dataset",
        default="ASD",
        help="Dataset: IOpsCompetition, UCR, SWaT, WADI, ESA, SMD, Exathlon, ASD",
    )
    parser.add_argument("--device", default="cpu", help="Device to use for training (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--polluted_rate", default=0.0, type=float)
    parser.add_argument("--project_dim", type=int, default=None)
    parser.add_argument("--oc_hidden_dim", type=int, default=None)
    parser.add_argument("--oc_pooling", choices=["flatten", "mean"], default=None)
    parser.add_argument("--center_eps", type=float, default=None)
    parser.add_argument("--lambda_oc", type=float, default=None)
    infonce = parser.add_mutually_exclusive_group()
    infonce.add_argument("--enable_infonce", dest="use_infonce", action="store_true",
                         help="Enable temporal-semantic InfoNCE alignment")
    infonce.add_argument("--disable_infonce", dest="use_infonce", action="store_false",
                         help="Skip InfoNCE; retain semantic encoding and fusion")
    parser.set_defaults(use_infonce=None)
    parser.add_argument("--study_name", default="manual")
    parser.add_argument("--config_json", default=None, help="JSON overrides for existing config fields")
    parser.add_argument("--entity_indices", default=None, help="Comma-separated original entity indices")
    parser.add_argument("--entity_records_dir", default=None, help="Save mergeable per-entity records")
    parser.add_argument("--summary_suffix", default="", help="Separate shard CSVs without changing their format")
    parser.add_argument("--cluster_plan_id", default=None)
    parser.add_argument("--cluster_worker_id", type=int, default=None)
    for name in ["window_size", "time_step", "d_model", "tcn_layers", "attention_heads", "num_epoch", "batch_size"]:
        parser.add_argument("--" + name, type=int, default=None)
    for name in ["lambda_align", "lambda_recon", "score_alpha", "score_beta", "lr"]:
        parser.add_argument("--" + name, type=float, default=None)
    parser.add_argument(
        "--text_model",
        default=None,
        help="SentenceTransformer model for semantic encoding, e.g. intfloat/e5-base-v2",
    )
    parser.add_argument(
        "--allow_hash_text_encoder",
        action="store_true",
        help="Allow fallback to the deterministic hash encoder if the text model cannot load.",
    )
    parser.add_argument(
        "--disable_text_embedding_cache",
        action="store_true",
        help="Disable prompt embedding cache for ablation or debugging.",
    )
    parser.add_argument("--hpname", default=None)
    parser.add_argument("--hpvalue", default=None, type=_parse_hpvalue)
    parser.add_argument(
        "--disable_timing",
        action="store_true",
        help="Disable SemSphere timing breakpoints.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers per train/val/test loader.",
    )
    return parser.parse_args()


def load_configs(dataset_name):
    dataset_name = "IOpsCompetition" if dataset_name == "KPI" else dataset_name
    module = __import__(f"conf.semsphere.{dataset_name}_Configs", fromlist=["Config"])
    return module.Config()


def load_dataset(dataset_name):
    if dataset_name in ("KPI", "IOpsCompetition", "SMD", "ASD", "Exathlon"):
        from experiments.datasets import load_dataset as load_entities
        return load_entities(dataset_name)
    # Delay dataset imports so --help works before external datasets are installed.
    from ts_datasets.ts_datasets.anomaly import ASD, Exathlon, IOpsCompetition, SMD, UCR
    if dataset_name == "IOpsCompetition":
        return IOpsCompetition()
    if dataset_name == "UCR":
        return UCR()
    if dataset_name == "SMD":
        return SMD()
    if dataset_name == "Exathlon":
        return Exathlon()
    if dataset_name == "ASD":
        return ASD()
    if dataset_name in ["SWaT", "WADI", "ESA"]:
        return None
    raise ValueError(f"Unknown dataset {dataset_name}")


def load_one_entity(dataset_name, dataloader, idx):
    from dataloader import esa, other_datasets, swat, wadi
    if dataset_name == "SWaT":
        return swat()
    if dataset_name == "WADI":
        return wadi()
    if dataset_name == "ESA":
        return esa()
    time_series, meta_data = dataloader[idx]
    return other_datasets(time_series, meta_data)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denom = np.sum(weights)
    if denom == 0:
        return np.nan
    return np.nansum(values * weights / denom)


def safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return np.nanmean(values)


def run_semsphere(args):
    if args.selected_dataset in ("KPI", "IOpsCompetition", "SMD", "ASD", "Exathlon"):
        from experiments.kpi_runtime import enable_kpi_imports
        enable_kpi_imports()
    from dataloader import data_generator1
    from utils import print_object
    start_time = datetime.now()
    dataset_name = "IOpsCompetition" if args.selected_dataset == "KPI" else args.selected_dataset
    device = torch.device(args.device)
    configs = load_configs(dataset_name)
    if getattr(args, "config_json", None):
        import json
        overrides = json.loads(open(args.config_json, encoding="utf-8").read())
        unknown = set(overrides) - set(vars(configs))
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        for key, value in overrides.items():
            setattr(configs, key, value)
    configs.num_workers = args.num_workers
    configs.study_name = args.study_name
    for name in ["project_dim", "oc_hidden_dim", "oc_pooling", "center_eps", "lambda_oc",
                 "window_size", "time_step", "d_model", "tcn_layers", "attention_heads", "num_epoch", "batch_size",
                 "lambda_align", "lambda_recon", "score_alpha", "score_beta", "lr"]:
        value = getattr(args, name)
        if value is not None:
            setattr(configs, name, value)
    if getattr(args, "use_infonce", None) is not None:
        configs.use_infonce = args.use_infonce
    if not isinstance(configs.use_infonce, bool):
        raise ValueError("use_infonce must be a JSON boolean or an explicit CLI switch")
    if args.text_model:
        configs.text_model = args.text_model
    if args.allow_hash_text_encoder:
        configs.allow_hash_text_encoder = True
    if args.disable_text_embedding_cache:
        configs.text_embedding_cache = False
    configs.profile_timing = not args.disable_timing
    if args.hpname:
        setattr(configs, args.hpname, args.hpvalue)
        experiment_description = f"{args.experiment_description}_{args.hpname}_{args.hpvalue}"
    else:
        experiment_description = args.experiment_description

    dt = load_dataset(dataset_name)
    model_num = 1 if dataset_name in ["SWaT", "WADI", "ESA"] else len(dt)

    print(f"Dataset: {dataset_name}")
    print("Method:  SemSphere")
    active_align = configs.use_infonce and configs.lambda_align != 0
    print(f"InfoNCE: {'enabled' if active_align else 'disabled'}; lambda_align={configs.lambda_align}")
    print(f"Random Seed:  {args.seed}")
    print("dataset entities", model_num)

    all_test_rpa_score, all_test_pa_score, all_test_pw_score = [], [], []
    all_anomaly_num, all_test_scores_reasonable = [], []
    all_test_aff_precision, all_test_aff_recall = [], []
    all_test_roc_auc, all_test_pr_auc = [], []
    all_test_vus_roc, all_test_vus_pr = [], []

    indices = list(range(model_num))
    if getattr(args, "entity_indices", None) is not None:
        indices = [int(value) for value in args.entity_indices.split(",") if value.strip()]
        if not indices or len(indices) != len(set(indices)) or any(i < 0 or i >= model_num for i in indices):
            raise ValueError("entity_indices must contain distinct valid entity indices")
    for idx in tqdm(indices):
        entity_start = time.perf_counter()
        set_seed(args.seed)
        load_start = time.perf_counter()
        train_data, test_data, train_labels, test_labels = load_one_entity(dataset_name, dt, idx)
        load_seconds = time.perf_counter() - load_start
        print(">" * 32, f"Entity {idx + 1}/{model_num}")    
        print(">" * 32, len(train_data))
        print(">" * 32, len(test_data))
        print("Data loaded ...")

        data_gen_start = time.perf_counter()
        train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator1(
            train_data,
            test_data,
            train_labels,
            test_labels,
            args.seed,
            configs,
        )
        data_gen_seconds = time.perf_counter() - data_gen_start

        model_start = time.perf_counter()
        model = base_Model(configs, device).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_seconds = time.perf_counter() - model_start
        optim_start = time.perf_counter()
        optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=configs.lr,
            betas=(configs.beta1, configs.beta2),
            weight_decay=configs.weight,
        )
        optimizer_seconds = time.perf_counter() - optim_start
        trainer_start = time.perf_counter()
        (
            _,
            test_aff,
            test_rpa,
            test_pa,
            test_pw,
            score_reasonable,
            _,
            test_roc_auc,
            test_pr_auc,
            test_vus_roc,
            test_vus_pr,
        ) = Trainer(model, optimizer, train_dl, val_dl, test_dl, device, configs, idx, args.seed)
        trainer_seconds = time.perf_counter() - trainer_start
        entity_seconds = time.perf_counter() - entity_start
        if configs.profile_timing:
            print(
                "[SemSphere timing] "
                f"entity={idx + 1}/{model_num} | "
                f"load_entity={load_seconds:.3f}s | "
                f"data_generator={data_gen_seconds:.3f}s | "
                f"model_init_to_device={model_seconds:.3f}s | "
                f"optimizer={optimizer_seconds:.3f}s | "
                f"trainer={trainer_seconds:.3f}s | "
                f"entity_total={entity_seconds:.3f}s"
            )

        all_anomaly_num.append(test_anomaly_window_num)
        all_test_scores_reasonable.append(score_reasonable)
        all_test_aff_precision.append(test_aff["precision"])
        all_test_aff_recall.append(test_aff["recall"])
        all_test_rpa_score.append(test_rpa)
        all_test_pa_score.append(test_pa)
        all_test_pw_score.append(test_pw)
        all_test_roc_auc.append(test_roc_auc)
        all_test_pr_auc.append(test_pr_auc)
        all_test_vus_roc.append(test_vus_roc)
        all_test_vus_pr.append(test_vus_pr)
        if getattr(args, "entity_records_dir", None):
            from experiments.cluster_results import save_entity_record
            save_entity_record(args.entity_records_dir, {
                "schema_version": 1, "dataset": dataset_name, "seed": args.seed,
                "entity_index": idx, "entity_count": model_num,
                "entity_id": str(dt.entity_ids[idx]) if hasattr(dt, "entity_ids") else (str(dt.kpi_ids[idx]) if hasattr(dt, "kpi_ids") else str(idx)),
                "plan_id": getattr(args, "cluster_plan_id", None),
                "worker_id": getattr(args, "cluster_worker_id", None),
                "experiment_description": experiment_description, "config": vars(configs),
                "elapsed_seconds": entity_seconds, "polluted_rate": args.polluted_rate, "anomaly_windows": test_anomaly_window_num,
                "affiliation": test_aff,
                "accumulators": {"rpa": vars(test_rpa), "pa": vars(test_pa), "pw": vars(test_pw)},
                "reasonable": vars(score_reasonable),
                "continuous": {"ROC_AUC": test_roc_auc, "PR_AUC": test_pr_auc,
                               "VUS_ROC": test_vus_roc, "VUS_PR": test_vus_pr},
            })

    all_anomaly_num = np.asarray(all_anomaly_num)
    test_aff_precision = safe_weighted_mean(all_test_aff_precision, all_anomaly_num)
    test_aff_recall = safe_weighted_mean(all_test_aff_recall, all_anomaly_num)
    if np.isnan(test_aff_precision) or np.isnan(test_aff_recall) or test_aff_precision + test_aff_recall == 0:
        test_aff_f1 = 0
    else:
        test_aff_f1 = 2 * test_aff_precision * test_aff_recall / (test_aff_precision + test_aff_recall)

    total_test_rpa_score = sum(all_test_rpa_score, ScoreAcc())
    total_test_pa_score = sum(all_test_pa_score, ScoreAcc())
    total_test_pw_score = sum(all_test_pw_score, ScoreAcc())
    total_test_scores_reasonable = sum(all_test_scores_reasonable, reasonable_accumulator())
    ucr_accuracy = total_test_scores_reasonable.get_all_metrics()
    test_roc_auc = safe_nanmean(all_test_roc_auc)
    test_pr_auc = safe_nanmean(all_test_pr_auc)
    test_vus_roc = safe_nanmean(all_test_vus_roc)
    test_vus_pr = safe_nanmean(all_test_vus_pr)

    print(">" * 32)
    if configs.dataset == "UCR":
        print("UCR metrics:\n", f"accuracy: {ucr_accuracy}\n")
    print(
        "affiliation metrics:\n",
        f"Precision: {test_aff_precision:.5f}\n",
        f"Recall:    {test_aff_recall:.5f}\n"
        f"F1 Scores: {test_aff_f1:.5f}\n"
        "Revised-point-adjusted metrics:\n",
        f"Precision: {total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted):.5f}\n",
        f"Recall:    {total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted):.5f}\n"
        f"F1 score:  {total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted):.5f}\n",
        "Point-adjusted metrics:\n",
        f"Precision: {total_test_pa_score.precision(ScoreType.PointAdjusted):.5f}\n",
        f"Recall:    {total_test_pa_score.recall(ScoreType.PointAdjusted):.5f}\n",
        f"F1 score:  {total_test_pa_score.f1(ScoreType.PointAdjusted):.5f}\n",
        "Point-wise metrics:\n",
        f"Precision: {total_test_pw_score.precision(ScoreType.Pointwise):.5f}\n",
        f"Recall:    {total_test_pw_score.recall(ScoreType.Pointwise):.5f}\n"
        f"F1 Scores:  {total_test_pw_score.f1(ScoreType.Pointwise):.5f}\n",
        "SemSphere continuous metrics:\n",
        f"Test ROC-AUC: {test_roc_auc:.5f}\n",
        f"Test PR-AUC:  {test_pr_auc:.5f}\n",
        f"Test VUS-ROC: {test_vus_roc:.5f}\n",
        f"Test VUS-PR:  {test_vus_pr:.5f}\n",
        "seed:", args.seed, "\n",
        "config setup:\n",
    )

    str_conf = print_object(configs)
    train_time = datetime.now() - start_time
    print(f"Training time is : {train_time}")

    result_values = {
        "Hyper-parameter": str_conf,
        "Polluted-rate": args.polluted_rate,
        "seed": args.seed,
        "Train Time": train_time,
        "UCR Accuracy": round(ucr_accuracy["accuracy"], 5),
        "Affiliation Precision": round(test_aff_precision, 5),
        "Affiliation Recall": round(test_aff_recall, 5),
        "Affiliation F1": round(test_aff_f1, 5),
        "RPA Precision": round(total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted), 5),
        "RPA Recall": round(total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted), 5),
        "RPA F1": round(total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted), 5),
        "PA Precision": round(total_test_pa_score.precision(ScoreType.PointAdjusted), 5),
        "PA Recall": round(total_test_pa_score.recall(ScoreType.PointAdjusted), 5),
        "PA F1": round(total_test_pa_score.f1(ScoreType.PointAdjusted), 5),
        "Point-wise Precision": round(total_test_pw_score.precision(ScoreType.Pointwise), 5),
        "Point-wise Recall": round(total_test_pw_score.recall(ScoreType.Pointwise), 5),
        "Point-wise F1": round(total_test_pw_score.f1(ScoreType.Pointwise), 5),
        "NAB Score (balanced)": round(total_test_rpa_score.nab_score(), 5),
        "NAB Score (high precision)": round(total_test_rpa_score.nab_score(fp_weight=0.22), 5),
        "NAB Score (high recall)": round(total_test_rpa_score.nab_score(fn_weight=2.0), 5),
        "Test ROC-AUC": round(test_roc_auc, 5),
        "Test PR-AUC": round(test_pr_auc, 5),
        "Test VUS-ROC": round(test_vus_roc, 5),
        "Test VUS-PR": round(test_vus_pr, 5),
    }

    os.makedirs("results", exist_ok=True)
    summary = os.path.join("results", f"SemSphere_{dataset_name}_summary_{experiment_description}{getattr(args, 'summary_suffix', '')}.csv")
    if os.path.exists(summary):
        df = pd.read_csv(summary, index_col=0)
        df = pd.DataFrame(df.to_numpy(dtype=object), index=df.index, columns=df.columns)
    else:
        df = pd.DataFrame(dtype=object)
    model_name = f"SemSphere{df.shape[1]}"
    df[model_name] = pd.Series(result_values, dtype=object)
    df.to_csv(summary, index=True)
    return result_values


def main():
    run_semsphere(parse_args())


if __name__ == "__main__":
    main()
