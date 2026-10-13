import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import subprocess
import copy
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from merlion.evaluate.anomaly import TSADScoreAccumulator as ScoreAcc, ScoreType
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.kpi_runtime import enable_kpi_imports
enable_kpi_imports()
from dataloader import data_generator1, esa, other_datasets, swat, wadi
from models.SemSphere.network.model import base_Model
from models.SemSphere.trainer.trainer import Trainer
from models.SemSphere.trainer import trainer as training
from models.reasonable_metric import reasonable_accumulator
from ts_datasets.ts_datasets.anomaly.ucr import UCR
from ts_datasets.ts_datasets.anomaly.base import TSADBaseDataset
from utils import print_object


@dataclass
class RunContext:
    args: Any
    configs: Any
    device: torch.device
    dataset_name: str
    experiment_description: str


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", default="v1")
    parser.add_argument("--experiment_description", default="test")
    parser.add_argument("--selected_dataset", default="ASD", choices=["KPI","IOpsCompetition","SMD","ASD","Exathlon","SWaT","WADI","ESA","UCR"])
    parser.add_argument("--window_size", type=int, default=None, help="Fixed training window unless window_choices is explicitly supplied")
    parser.add_argument("--window_choices", default=None, help="Optional comma-separated positive window lengths")
    parser.add_argument("--time_step", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--tcn_layers", type=int, default=None)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--njobs", type=int, default=1, help="Independent trial processes sharing one study")
    parser.add_argument("--trials", type=int, default=20, help="Trials to add on this invocation; 0 skips this study")
    parser.add_argument("--internal_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--old_best", default=None)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--text_model", default=None)
    parser.add_argument("--attention_heads", type=int, default=None)
    parser.add_argument("--project_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--vus_window", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Fixed optimizer learning rate")
    parser.add_argument("--weight", type=float, default=None, help="Fixed Adam weight decay")
    parser.add_argument("--dropout", type=float, default=None, help="Fixed fusion/decoder dropout probability")
    parser.add_argument("--no_pruning", action="store_true")
    parser.add_argument("--lambda_align_min", type=float, default=1e-4)
    parser.add_argument("--lambda_align_max", type=float, default=1.)
    parser.add_argument("--lambda_oc_min", type=float, default=1e-4)
    parser.add_argument("--lambda_oc_max", type=float, default=1.)
    parser.add_argument("--score_beta_min", type=float, default=0.)
    parser.add_argument("--score_beta_max", type=float, default=2.)
    parser.add_argument("--score_beta_step", type=float, default=.1)
    parser.add_argument(
        "--storage_backend",
        choices=["journal", "sqlite"],
        default="journal",
    )
    parser.add_argument("--storage_path", default=None)
    parser.add_argument("--objective_metric", default="RPA F1", choices=["RPA F1"])
    parser.add_argument("--profile_timing", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers per train/val/test loader. Use 0 or 1 for high Optuna concurrency.",
    )
    return parser.parse_args(argv)


def setup_logging(trial_number, study_name):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(f"trial_{trial_number}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join("logs", f"{study_name}.log"))
        handler.setFormatter(logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def setup_optuna_logging(study_name):
    optuna_logger = optuna.logging.get_logger("optuna")
    handler = logging.FileHandler(os.path.join("logs", f"{study_name}.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s - [OPTUNA] - %(levelname)s - %(message)s"))
    optuna_logger.addHandler(handler)


def load_configs(dataset_name):
    dataset_name = "IOpsCompetition" if dataset_name == "KPI" else dataset_name
    module = __import__(f"conf.semsphere.{dataset_name}_Configs", fromlist=["Config"])
    return module.Config()


def build_context(args):
    configs = load_configs(args.selected_dataset)
    for key in ("window_size","time_step","d_model","tcn_layers","num_epoch","attention_heads","project_dim","batch_size"):
        value=getattr(args,key)
        if value is not None:
            setattr(configs,key,value)
    for key in ("lr", "weight", "dropout"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(configs, key, value)
    if configs.lr <= 0 or configs.weight < 0 or not 0 <= configs.dropout < 1:
        raise ValueError("Require lr > 0, weight >= 0, and 0 <= dropout < 1")
    if args.text_model:
        configs.text_model=args.text_model
    configs.lambda_recon=1.0
    configs.score_alpha=1.0
    configs.save_diagnostics=True
    configs.profile_timing=args.profile_timing
    configs.num_workers=args.num_workers
    configs.drop_last=False
    configs.vus_window=args.vus_window or configs.window_size
    args.window_choices=([int(v) for v in args.window_choices.split(',')] if isinstance(args.window_choices,str)
                         else [configs.window_size])
    if configs.time_step<1 or not args.window_choices or any(w<1 for w in args.window_choices):
        raise ValueError("Positive window lengths and time_step required")
    args.seeds=([int(v) for v in args.seeds.split(',')] if isinstance(args.seeds,str) else args.seeds)
    if not args.seeds or len(set(args.seeds))!=len(args.seeds):
        raise ValueError("Distinct seeds required")
    if args.njobs<1:
        raise ValueError("njobs must be positive")
    if args.trials<1 or configs.d_model%configs.attention_heads:
        raise ValueError("Positive trials and d_model divisible by attention_heads required")
    for low,high in ((args.lambda_align_min,args.lambda_align_max),(args.lambda_oc_min,args.lambda_oc_max)):
        if not 0<low<=high:
            raise ValueError("Positive log-search bounds required")
    if args.score_beta_min<0 or args.score_beta_max<args.score_beta_min or args.score_beta_step<=0:
        raise ValueError("Invalid score_beta bounds")
    if args.device=='auto':
        args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu or auto")
    args.selected_dataset='IOpsCompetition' if args.selected_dataset=='KPI' else args.selected_dataset
    return RunContext(
        args=args,
        configs=configs,
        device=torch.device(args.device),
        dataset_name=args.selected_dataset,
        experiment_description=args.experiment_description,
    )


def build_storage(args):
    if args.storage_backend == "journal":
        path = args.storage_path or os.path.join("optuna", "semsphere_journals", f"{args.selected_dataset}.log")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        backend = optuna.storages.journal.JournalFileBackend(path)
        return optuna.storages.JournalStorage(backend)

    path = args.storage_path or "optuna/semsphere_db.sqlite3"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return optuna.storages.RDBStorage(
        f"sqlite:///{path}",
        engine_kwargs={"connect_args": {"timeout": 120}},
    )


def load_dataset(dataset_name, data_root=None):
    folder="iops_competition" if dataset_name=="IOpsCompetition" else dataset_name.lower()
    directory=Path(data_root) if data_root else ROOT/'data'/folder
    if dataset_name in ("IOpsCompetition","SMD","ASD","Exathlon"):
        from experiments.datasets import load_dataset as load_entities
        return load_entities(dataset_name,directory)
    if dataset_name=="UCR":
        files=sorted(directory.glob('UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/*.txt'))
        if not files:
            files=sorted(directory.glob('*.txt'))
        if not files:
            raise FileNotFoundError(f"No local UCR txt files in {directory}; no automatic download")
        dataset=UCR.__new__(UCR)
        TSADBaseDataset.__init__(dataset)
        dataset.time_series=[str(f) for f in files]
        return dataset
    if dataset_name in ("SWaT","WADI","ESA"):
        
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        old=Path.cwd()
        with tempfile.TemporaryDirectory(prefix='semsphere_data_') as temp:
            base=Path(temp)/'data';base.mkdir()
            (base/dataset_name.lower()).symlink_to(directory.resolve(),target_is_directory=True)
            try:
                os.chdir(temp)
                return {'SWaT':swat,'WADI':wadi,'ESA':esa}[dataset_name]()
            finally:
                os.chdir(old)
    raise ValueError(f"Unknown dataset {dataset_name}")


def load_one_entity(dataset_name, dataloader, idx):
    if dataset_name in ("SWaT","WADI","ESA"):
        return dataloader
    time_series,meta_data=dataloader[idx]
    return other_datasets(time_series,meta_data)


@contextmanager
def fixed_vus(configs):
    original=training._continuous_metrics
    def calculate(target,score,config,logger,split):
        metric_config=copy.copy(config)
        metric_config.window_size=configs.vus_window
        return original(target,score,metric_config,logger,split)
    training._continuous_metrics=calculate
    try:
        yield
    finally:
        training._continuous_metrics=original


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def collect_metrics(
    train_time,
    scores,
    anomaly_num,
    aff_precision,
    aff_recall,
    reasonable,
    test_roc_auc,
    test_pr_auc,
    test_vus_roc,
    test_vus_pr,
    dataset_name,
):
    all_anomaly_num = np.asarray(anomaly_num)
    test_aff_precision = safe_weighted_mean(aff_precision, all_anomaly_num)
    test_aff_recall = safe_weighted_mean(aff_recall, all_anomaly_num)
    if np.isnan(test_aff_precision) or np.isnan(test_aff_recall) or test_aff_precision + test_aff_recall == 0:
        test_aff_f1 = 0
    else:
        test_aff_f1 = 2 * test_aff_precision * test_aff_recall / (test_aff_precision + test_aff_recall)

    total_rpa = sum(scores["rpa"], ScoreAcc())
    total_pa = sum(scores["pa"], ScoreAcc())
    total_pw = sum(scores["pw"], ScoreAcc())
    total_reasonable = sum(reasonable, reasonable_accumulator())
    ucr_accuracy = total_reasonable.get_all_metrics()

    metrics = {
        "Train Time": round(train_time, 5),
        "Affiliation Precision": round(test_aff_precision, 5),
        "Affiliation Recall": round(test_aff_recall, 5),
        "Affiliation F1": round(test_aff_f1, 5),
        "RPA Precision": round(total_rpa.precision(ScoreType.RevisedPointAdjusted), 5),
        "RPA Recall": round(total_rpa.recall(ScoreType.RevisedPointAdjusted), 5),
        "RPA F1": round(total_rpa.f1(ScoreType.RevisedPointAdjusted), 5),
        "PA Precision": round(total_pa.precision(ScoreType.PointAdjusted), 5),
        "PA Recall": round(total_pa.recall(ScoreType.PointAdjusted), 5),
        "PA F1": round(total_pa.f1(ScoreType.PointAdjusted), 5),
        "Point-wise Precision": round(total_pw.precision(ScoreType.Pointwise), 5),
        "Point-wise Recall": round(total_pw.recall(ScoreType.Pointwise), 5),
        "Point-wise F1": round(total_pw.f1(ScoreType.Pointwise), 5),
        "NAB Score (balanced)": round(total_rpa.nab_score(), 5),
        "NAB Score (high precision)": round(total_rpa.nab_score(fp_weight=0.22), 5),
        "NAB Score (high recall)": round(total_rpa.nab_score(fn_weight=2.0), 5),
        "Test ROC-AUC": round(safe_nanmean(test_roc_auc), 5),
        "Test PR-AUC": round(safe_nanmean(test_pr_auc), 5),
        "Test VUS-ROC": round(safe_nanmean(test_vus_roc), 5),
        "Test VUS-PR": round(safe_nanmean(test_vus_pr), 5),
    }
    if dataset_name == "UCR":
        metrics["UCR Accuracy"] = round(ucr_accuracy["accuracy"], 5)
    return metrics


def run_one_seed(ctx, seed, dataloader):
    set_seed(seed)
    start_time = datetime.now()
    model_num = 1 if ctx.dataset_name in ["SWaT", "WADI", "ESA"] else len(dataloader)
    trial_id = getattr(ctx.configs, "trial_id", "manual")

    scores = {"rpa": [], "pa": [], "pw": []}
    anomaly_num, reasonable = [], []
    aff_precision, aff_recall = [], []
    all_test_roc_auc, all_test_pr_auc = [], []
    all_test_vus_roc, all_test_vus_pr = [], []

    for idx in tqdm(range(model_num)):
        set_seed(seed)
        train_data, test_data, train_labels, test_labels = load_one_entity(ctx.dataset_name, dataloader, idx)
        print(">" * 32, f"Entity {idx + 1}/{model_num}")
        print(">" * 32, len(train_data))
        print("Data loaded ...")
        train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator1(
            train_data,
            test_data,
            train_labels,
            test_labels,
            seed,
            ctx.configs,
        )

        model = base_Model(ctx.configs, ctx.device).to(ctx.device)
        optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=ctx.configs.lr,
            betas=(ctx.configs.beta1,ctx.configs.beta2),
            weight_decay=ctx.configs.weight,
        )
        (
            _,
            test_aff,
            test_rpa,
            test_pa,
            test_pw,
            score_reasonable,
            _,
            roc_auc,
            pr_auc,
            vus_roc,
            vus_pr,
        ) = Trainer(model, optimizer, train_dl, val_dl, test_dl, ctx.device, ctx.configs, idx, seed)

        anomaly_num.append(test_anomaly_window_num)
        reasonable.append(score_reasonable)
        aff_precision.append(test_aff["precision"])
        aff_recall.append(test_aff["recall"])
        scores["rpa"].append(test_rpa)
        scores["pa"].append(test_pa)
        scores["pw"].append(test_pw)
        all_test_roc_auc.append(roc_auc)
        all_test_pr_auc.append(pr_auc)
        all_test_vus_roc.append(vus_roc)
        all_test_vus_pr.append(vus_pr)

    return collect_metrics(
        (datetime.now() - start_time).total_seconds(),
        scores,
        anomaly_num,
        aff_precision,
        aff_recall,
        reasonable,
        all_test_roc_auc,
        all_test_pr_auc,
        all_test_vus_roc,
        all_test_vus_pr,
        ctx.dataset_name,
    )


def aggregate_seed_metrics(metrics_list):
    agg = defaultdict(list)
    for item in metrics_list:
        for key, value in item.items():
            if value is not None:
                agg[key].append(value)
    summary = {}
    for key, values in agg.items():
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary, agg


def safe_filename_part(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def save_summary_csv(ctx, raw_metrics, summary, description):
    result_dir = os.path.join("results", ctx.dataset_name, ctx.args.study_name)
    os.makedirs(result_dir, exist_ok=True)
    rpa_f1_mean = summary.get("RPA F1_mean", float("nan"))
    vus_roc_mean = summary.get("Test VUS-ROC_mean", float("nan"))
    filename = safe_filename_part(f"{description}_F1_{rpa_f1_mean:.5f}_ROC_{vus_roc_mean:.5f}") + ".csv"
    path = os.path.join(result_dir, filename)
    num_seeds = len(next(iter(raw_metrics.values())))
    seed_cols = [f"SemSphere_seed_{ctx.args.seeds[idx]}" for idx in range(num_seeds)]
    columns = seed_cols + ["mean", "std"]
    rows = list(raw_metrics.keys()) + ["Hyper-parameter"]
    df = pd.DataFrame(index=rows, columns=columns, dtype=object)
    for metric_name, values in raw_metrics.items():
        for idx, value in enumerate(values):
            df.loc[metric_name, seed_cols[idx]] = value
        df.loc[metric_name, "mean"] = summary[f"{metric_name}_mean"]
        df.loc[metric_name, "std"] = summary[f"{metric_name}_std"]
    df.loc["Hyper-parameter", :] = print_object(ctx.configs)
    df.to_csv(path, index=True)
    return path


def suggest_semsphere_params(trial, configs, args):
    configs.lambda_recon=1.0
    configs.score_alpha=1.0
    configs.lambda_align=trial.suggest_float("lambda_align",args.lambda_align_min,args.lambda_align_max,log=True)
    configs.score_beta=trial.suggest_float("score_beta",args.score_beta_min,args.score_beta_max,step=args.score_beta_step)
    configs.lambda_oc=trial.suggest_float("lambda_oc",args.lambda_oc_min,args.lambda_oc_max,log=True)
    if len(args.window_choices)>1:
        configs.window_size=trial.suggest_categorical("window_size",args.window_choices)
    else:
        configs.window_size=args.window_choices[0]


def objective(dataloader, trial, base_ctx):
    ctx=copy.deepcopy(base_ctx)
    ctx.configs.trial_id=trial.number
    ctx.configs.study_name=f"{ctx.dataset_name}_{ctx.args.study_name}"
    suggest_semsphere_params(trial,ctx.configs,ctx.args)
    trial.set_user_attr('config',vars(ctx.configs))
    logger=setup_logging(trial.number,f"SemSphere_{ctx.dataset_name}_{ctx.args.study_name}")
    metrics_list=[]
    with fixed_vus(ctx.configs):
        for step,seed in enumerate(ctx.args.seeds):
            metrics=run_one_seed(ctx,seed,dataloader)
            metrics_list.append(metrics)
            summary,raw=aggregate_seed_metrics(metrics_list)
            trial.set_user_attr('completed_seeds',ctx.args.seeds[:len(metrics_list)])
            trial.set_user_attr('metrics',summary)
            save_summary_csv(ctx,raw,summary,f"trial_{trial.number}_partial")
            score=summary['RPA F1_mean']
            if not np.isfinite(score):
                raise ValueError('Non-finite RPA F1; check labels and evaluation')
            trial.report(score,step=step)
            logger.info('[Seed %d] RPA F1: %.5f | VUS-ROC: %.5f | VUS-PR: %.5f',seed,metrics['RPA F1'],metrics['Test VUS-ROC'],metrics['Test VUS-PR'])
            if step+1<len(ctx.args.seeds) and trial.should_prune():
                raise optuna.TrialPruned()
    summary,raw=aggregate_seed_metrics(metrics_list)
    path=save_summary_csv(ctx,raw,summary,f"trial_{trial.number}")
    trial.set_user_attr('summary_csv',path)
    torch.cuda.empty_cache()
    return summary['RPA F1_mean']


def export_best(study,ctx):
    complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    if not complete:
        print('No completed trials; no best configuration exported')
        return
    def rank(t):
        m=t.user_attrs['metrics']
        vus=[m.get(k,float('nan')) for k in ('Test VUS-ROC_mean','Test VUS-PR_mean')]
        secondary=float(np.mean(vus)) if np.isfinite(vus).all() else -float('inf')
        return (t.value,secondary,-t.number)
    ordered=sorted(complete,key=rank,reverse=True)
    best=ordered[0]
    folder=ROOT/'results'/ctx.dataset_name/ctx.args.study_name
    folder.mkdir(parents=True,exist_ok=True)
    import json
    (folder/'best_params.json').write_text(json.dumps(best.params,indent=2))
    fields=set(vars(load_configs(ctx.dataset_name)))
    config={k:v for k,v in best.user_attrs['config'].items() if k in fields}
    (folder/'best_config.json').write_text(json.dumps(config,indent=2))
    (folder/'best_trial.json').write_text(json.dumps({'trial':best.number,'params':best.params,
        'config':best.user_attrs['config'],'metrics':best.user_attrs['metrics'],
        'seeds':best.user_attrs['completed_seeds'],'summary_csv':best.user_attrs['summary_csv']},indent=2))
    rows=[{'trial':t.number,'RPA F1':t.value,'VUS-ROC':t.user_attrs['metrics'].get('Test VUS-ROC_mean'),
           'VUS-PR':t.user_attrs['metrics'].get('Test VUS-PR_mean'),**t.params,
           'window_size':t.user_attrs['config']['window_size']} for t in ordered]
    pd.DataFrame(rows).to_csv(folder/'trials_rpa_vus.csv',index=False)
    print('Best trial:',best.number,'RPA F1:',best.value,'VUS-ROC:',rows[0]['VUS-ROC'],'VUS-PR:',rows[0]['VUS-PR'])
    print('Best params:',best.params)


def parallel_budgets(trials,jobs):
    jobs=min(trials,jobs)
    base,extra=divmod(trials,jobs)
    return [base+(i<extra) for i in range(jobs)]


def main():
    args = parse_args()
    if args.trials == 0:
        print(f"Skipping {args.selected_dataset} {args.study_name}: remaining trials=0",flush=True)
        return
    os.chdir(ROOT)
    ctx = build_context(args)
    storage = build_storage(args)
    study_name = f"SemSphere_{ctx.dataset_name}_{args.study_name}"
    old_study_name = f"SemSphere_{ctx.dataset_name}_{args.old_best}"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
    try:
        if args.old_best is None:
            raise ValueError("No historical study requested")
        old_study = optuna.load_study(study_name=old_study_name, storage=storage)
        best_params = {k:v for k,v in old_study.best_params.items() if k in {"lambda_align","score_beta","lambda_oc","window_size"}}
        print(f"[*] 发现历史最佳参数: {best_params}")
    except Exception:
        best_params = {}
        print("[!] 未发现可继承的旧实验")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.NopPruner() if args.no_pruning else pruner,
        sampler=optuna.samplers.TPESampler(constant_liar=True),
        storage=storage,
        load_if_exists=True,
    )
    # A resumed study must compare the same data, seeds, fixed model and search space.
    profile={'dataset':ctx.dataset_name,'data_root':str(Path(args.data_root).resolve()) if args.data_root else str(ROOT/'data'/('iops_competition' if ctx.dataset_name=='IOpsCompetition' else ctx.dataset_name.lower())),
        'config':vars(ctx.configs),'seeds':args.seeds,'window_choices':args.window_choices,
        'bounds':{k:v for k,v in vars(args).items() if k.endswith(('_min','_max','_step'))},
        'objective':'RPA F1','protocol':'full_test'}
    previous=study.user_attrs.get('semsphere_profile')
    if previous is not None and previous!=profile:
        raise ValueError('Study settings changed; use a new --study_name')
    study.set_user_attr('semsphere_profile',profile)
    if len(study.trials) == 0 and best_params:
        study.enqueue_trial(best_params)

    if args.njobs>1:
        budgets=parallel_budgets(args.trials,args.njobs)
        children=[]
        try:
            for budget in budgets:
                children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                    *sys.argv[1:],'--njobs','1','--trials',str(budget),'--internal_worker']))
            failures=[child.wait() for child in children]
            if any(failures):
                raise RuntimeError(f'Trial process failed: exit codes {failures}')
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait()
        export_best(study,ctx)
        return
    dataloader = load_dataset(ctx.dataset_name,ctx.args.data_root)
    os.makedirs("logs", exist_ok=True)
    setup_optuna_logging(study.study_name)
    study.optimize(
        lambda trial: objective(dataloader, trial, ctx),
        n_trials=args.trials,
        n_jobs=args.njobs,
    )
    if not args.internal_worker:
        export_best(study,ctx)


if __name__ == "__main__":
    main()
