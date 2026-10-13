import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from merlion.evaluate.anomaly import ScoreType
from sklearn.metrics import average_precision_score, roc_auc_score

from models import ad_predict
from models.reasonable_metric import reasonable_accumulator, tsad_reasonable
from models.SemSphere.trainer.early_stopping import EarlyStopping


def _sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _format_timing(timing):
    parts = []
    for key, value in timing.items():
        if "batches" in key:
            parts.append(f"{key}={int(value)}")
        else:
            parts.append(f"{key}={value:.3f}s")
    return " | ".join(parts)


def info_nce_loss(z1, z2, tau=0.2):
    z1 = F.normalize(z1.mean(dim=1), dim=-1)
    z2 = F.normalize(z2.mean(dim=1), dim=-1)
    logits = torch.matmul(z1, z2.t()) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def _batch_zscore(x, eps=1e-6):
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def _score(output, config):
    alpha = getattr(config, "score_alpha", 1.0)
    beta = getattr(config, "score_beta", 1.0)
    recon = _batch_zscore(output["recon_error"])
    oc = _batch_zscore(output["oc_distance"])
    return alpha * recon + beta * oc


def _with_reconstruction_error(output, x):
    output = dict(output)
    output["recon_error"] = F.mse_loss(
        output["x_recon"],
        x.float(),
        reduction="none",
    ).mean(dim=(1, 2))
    return output


def _losses(output, config):
    lambda_align = getattr(config, "lambda_align", 0.1)
    lambda_recon = getattr(config, "lambda_recon", 1.0)
    lambda_oc = getattr(config, "lambda_oc", 0.1)

    loss_recon = output["recon_error"].mean()
    loss_oc = output["oc_distance"].mean()
    if getattr(config, "use_infonce", True) and lambda_align != 0:
        loss_align = info_nce_loss(output["z_temporal"], output["z_semantic"])
    else:
        loss_align = loss_recon.new_zeros(())
    total = lambda_align * loss_align + lambda_recon * loss_recon + lambda_oc * loss_oc
    return total, loss_align, loss_recon, loss_oc


def _continuous_metrics(target, score, config, logger, split):
    target = np.asarray(target).reshape(-1)
    score = np.asarray(score).reshape(-1)
    metrics = {}
    if len(np.unique(target)) < 2:
        logger.warning("%s continuous metrics skipped: target contains only one class", split)
        metrics["ROC_AUC"] = np.nan
        metrics["PR_AUC"] = np.nan
    else:
        try:
            metrics["ROC_AUC"] = roc_auc_score(target, score)
            metrics["PR_AUC"] = average_precision_score(target, score)
        except ValueError as exc:
            logger.warning("%s ROC-AUC and PR-AUC unavailable: %s", split, exc)
            metrics["ROC_AUC"] = np.nan
            metrics["PR_AUC"] = np.nan

    try:
        from vus.metrics import get_metrics

        vus_results = get_metrics(
            score, target, metric="vus", slidingWindow=config.window_size
        )
        metrics["VUS_ROC"] = vus_results["VUS_ROC"]
        metrics["VUS_PR"] = vus_results["VUS_PR"]
    except Exception as exc:
        logger.warning(
            "%s VUS metrics unavailable; falling back to np.nan. This is usually "
            "caused by a vus package/dependency compatibility issue: %s",
            split, exc
        )
        metrics["VUS_ROC"] = np.nan
        metrics["VUS_PR"] = np.nan

    logger.info(
        "%s continuous metrics | ROC-AUC: %.5f | PR-AUC: %.5f | VUS-ROC: %.5f | VUS-PR: %.5f",
        split, metrics["ROC_AUC"], metrics["PR_AUC"], metrics["VUS_ROC"], metrics["VUS_PR"]
    )
    print(
        f"{split} continuous metrics | "
        f"ROC-AUC: {metrics['ROC_AUC']:.5f} | "
        f"PR-AUC: {metrics['PR_AUC']:.5f} | "
        f"VUS-ROC: {metrics['VUS_ROC']:.5f} | "
        f"VUS-PR: {metrics['VUS_PR']:.5f}"
    )
    return metrics


@torch.no_grad()
def initialize_oc_center(model, train_loader, config, device):
    """Cover the full training dataset once; never inspect val/test loaders.

    Labeled anomalous training
    windows are excluded from center estimation by default. The training loss
    retains the original all-window reduction.
    """
    if bool(model.oc_center_initialized.item()):
        return 0
    from torch.utils.data import DataLoader

    center_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size or getattr(config, "batch_size", 128),
        shuffle=False,
        drop_last=False,
        num_workers=train_loader.num_workers,
        collate_fn=train_loader.collate_fn,
        pin_memory=train_loader.pin_memory,
    )
    was_training = model.training
    model.eval()
    total = torch.zeros_like(model.oc_center)
    count = 0
    try:
        for x_s, target in center_loader:
            x = x_s.float().to(device).permute(0, 2, 1)
            q = model(x)["q"]
            if getattr(config, "center_normal_only", True):
                mask = target.to(device).reshape(-1) == 0
                q = q[mask]
            total += q.sum(dim=0)
            count += q.size(0)
        if count == 0:
            raise ValueError("No eligible training windows for OC center initialization")
        model.set_oc_center(total / count, eps=getattr(config, "center_eps", 0.1))
    finally:
        model.train(was_training)
    return count


def _run_epoch(model, data_loader, config, device, optimizer=None, collect_details=False):
    is_training = optimizer is not None
    if not bool(model.oc_center_initialized.item()):
        raise RuntimeError("Initialize the SemSphere center using training data before running an epoch")
    profile_timing = getattr(config, "profile_timing", False)
    model.train(is_training)
    totals, targets, scores = [], [], []
    timing = defaultdict(float)
    batch_count = 0
    details = {
        key: [] for key in ["score", "recon_error", "oc_distance", "q", "target"]
    }
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for x_s, target in data_loader:
            batch_count += 1
            step_start = time.perf_counter()
            x_s = x_s.float().to(device).permute(0, 2, 1)
            target = target.long().to(device).reshape(-1)
            if profile_timing:
                _sync_if_cuda(device)
                timing["to_device"] += time.perf_counter() - step_start
            if is_training:
                optimizer.zero_grad()
            forward_start = time.perf_counter()
            output = _with_reconstruction_error(model(x_s), x_s)
            score = _score(output, config)
            losses = _losses(output, config)
            if profile_timing:
                _sync_if_cuda(device)
                timing["forward_loss"] += time.perf_counter() - forward_start
            if is_training:
                backward_start = time.perf_counter()
                losses[0].backward()
                optimizer.step()
                if profile_timing:
                    _sync_if_cuda(device)
                    timing["backward_step"] += time.perf_counter() - backward_start
            collect_start = time.perf_counter()
            totals.append([item.detach().item() for item in losses])
            targets.append(target.detach())
            scores.append(score.detach())
            if collect_details:
                details["score"].append(score.detach().cpu())
                for key in ["recon_error", "oc_distance", "q"]:
                    details[key].append(output[key].detach().cpu())
                details["target"].append(target.detach().cpu())
            if profile_timing:
                timing["collect"] += time.perf_counter() - collect_start
    mean_losses = np.asarray(totals).mean(axis=0)
    joined_details = {
        key: torch.cat(value).numpy() for key, value in details.items() if value
    }
    timing["batches"] = float(batch_count)
    return torch.cat(targets).cpu().numpy(), torch.cat(scores).cpu().numpy(), mean_losses, joined_details, dict(timing)


def _save_diagnostics(config, idx, seed, trial_id, details, val_metrics, test_metrics):
    if not getattr(config, "save_diagnostics", True):
        return
    study_name = getattr(config, "study_name", "manual")
    path = os.path.join(
        "scores",
        "SemSphere",
        config.dataset,
        study_name,
        f"trial_{trial_id}",
        f"seed_{seed}",
    )
    os.makedirs(path, exist_ok=True)
    payload = dict(details)
    payload.update({f"validation_{key}": value for key, value in val_metrics.items()})
    payload.update({f"test_{key}": value for key, value in test_metrics.items()})
    np.savez(os.path.join(path, f"diagnostics_{idx}_{seed}.npz"), **payload)


def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, config, idx, seed):
    profile_timing = getattr(config, "profile_timing", False)
    trial_id = str(getattr(config, "trial_id", os.environ.get("CURRENT_TRIAL_ID", "manual")))
    study_name = getattr(config, "study_name", "manual")
    logger = logging.getLogger(f"trial_{trial_id}")
    save_path = os.path.join(
        "best_network",
        "SemSphere",
        config.dataset,
        study_name,
        f"trial_{trial_id}",
        f"seed_{seed}",
    )
    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(save_path, idx, patience=config.early_stop)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, "min")
    timing = defaultdict(float)
    center_count = initialize_oc_center(model, train_dl, config, device)
    logger.info("SemSphere center initialized from %d eligible training windows; fixed during training", center_count)

    for epoch in range(1, config.num_epoch + 1):
        epoch_start = time.perf_counter()
        _, _, train_losses, _, train_timing = _run_epoch(model, train_dl, config, device, model_optimizer)
        val_target, val_scores, val_losses, _, val_timing = _run_epoch(model, val_dl, config, device)
        timing["train_epoch_total"] += sum(value for key, value in train_timing.items() if key != "batches")
        timing["val_epoch_total"] += sum(value for key, value in val_timing.items() if key != "batches")
        timing["train_batches"] += train_timing.get("batches", 0.0)
        timing["val_batches"] += val_timing.get("batches", 0.0)
        timing["train_to_device"] += train_timing.get("to_device", 0.0)
        timing["train_forward_loss"] += train_timing.get("forward_loss", 0.0)
        timing["train_backward_step"] += train_timing.get("backward_step", 0.0)
        timing["train_collect"] += train_timing.get("collect", 0.0)
        timing["val_to_device"] += val_timing.get("to_device", 0.0)
        timing["val_forward_loss"] += val_timing.get("forward_loss", 0.0)
        timing["val_collect"] += val_timing.get("collect", 0.0)
        scheduler_start = time.perf_counter()
        scheduler.step(train_losses[0])
        early_stopping(0, 0, 0, -val_losses[0], val_scores, None, model)
        if profile_timing:
            timing["scheduler_early_stopping"] += time.perf_counter() - scheduler_start
            timing["epoch_wall"] += time.perf_counter() - epoch_start
        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                "Epoch: %d | total: %.4f | align: %.4f | recon: %.4f | oc: %.4f",
                epoch, *train_losses
            )
        if early_stopping.early_stop:
            break

    checkpoint = os.path.join(save_path, str(idx).zfill(2) + "_best_network.pth")
    checkpoint_start = time.perf_counter()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    if profile_timing:
        _sync_if_cuda(device)
        timing["load_checkpoint"] += time.perf_counter() - checkpoint_start
    final_val_start = time.perf_counter()
    val_target, val_scores, _, _, final_val_timing = _run_epoch(model, val_dl, config, device)
    if profile_timing:
        timing["final_val_wall"] += time.perf_counter() - final_val_start
        timing["final_val_batches"] += final_val_timing.get("batches", 0.0)
    final_test_start = time.perf_counter()
    test_target, test_scores, _, details, final_test_timing = _run_epoch(model, test_dl, config, device, collect_details=True)
    if profile_timing:
        timing["final_test_wall"] += time.perf_counter() - final_test_start
        timing["final_test_batches"] += final_test_timing.get("batches", 0.0)
    print(f"Test target shape: {test_target.shape}, \n"
          f"Test scores shape: {test_scores.shape}")
    metrics_start = time.perf_counter()
    test_metrics = _continuous_metrics(test_target, test_scores, config, logger, "Test")
    if profile_timing:
        timing["continuous_metrics"] += time.perf_counter() - metrics_start
    diagnostics_start = time.perf_counter()
    _save_diagnostics(config, idx, seed, trial_id, details, {}, test_metrics)
    if profile_timing:
        timing["save_diagnostics"] += time.perf_counter() - diagnostics_start
    if not getattr(config, "keep_checkpoints", False) and os.path.exists(checkpoint):
        remove_start = time.perf_counter()
        os.remove(checkpoint)
        if profile_timing:
            timing["remove_checkpoint"] += time.perf_counter() - remove_start

    predict_start = time.perf_counter()
    _, val_score, _, _, _ = ad_predict(val_target, val_scores, config.threshold_determine, config.detect_nu)
    test_aff, test_rpa, test_pa, test_pw, predict = ad_predict(
        test_target, test_scores, config.threshold_determine, config.detect_nu
    )
    if profile_timing:
        timing["ad_predict"] += time.perf_counter() - predict_start
    if config.dataset == "UCR":
        score_reasonable = tsad_reasonable(test_target, predict, config.time_step)
    else:
        score_reasonable = reasonable_accumulator(1, 0)
    logger.info("Best val RPA F1: %.4f", val_score.f1(ScoreType.RevisedPointAdjusted))
    if profile_timing:
        print(f"[SemSphere timing][Trainer entity {idx}] {_format_timing(timing)}")
        model_timing = getattr(model, "timing", None)
        if model_timing:
            print(f"[SemSphere timing][Model entity {idx}] {_format_timing(model_timing)}")
    return (
        test_scores,
        test_aff,
        test_rpa,
        test_pa,
        test_pw,
        score_reasonable,
        predict,
        test_metrics["ROC_AUC"],
        test_metrics["PR_AUC"],
        test_metrics["VUS_ROC"],
        test_metrics["VUS_PR"],
    )
