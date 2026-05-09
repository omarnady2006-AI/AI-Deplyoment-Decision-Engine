"""
src/training/train.py

Full training pipeline for DeploymentDecisionNet.

LOSS FUNCTION
=============
    Loss = CE(runtime_logits, runtime_label)
         + λ1 · MSE(latency_log_pred, latency_target)
         + λ2 · MSE(memory_log_pred, memory_target)
         + λ3 · BCE(failure_logits, failure_target)

λ defaults:
    λ1 = 1.0  (latency matters equally to runtime selection)
    λ2 = 0.5  (memory secondary)
    λ3 = 2.0  (failure heavily penalised)

INVALID DECISION PENALTY
=========================
Instead of a rule that rejects a decision, we add a differentiable penalty
to the loss when the model's top-1 prediction differs from the optimal label:

    invalid_penalty = γ · BCE(predicted_runtime_prob[optimal], ones)

where γ = 0.5. This teaches the model to assign high probability to the
optimal runtime rather than simply rejecting "wrong" decisions via rules.

CALIBRATION PHASE
=================
After main training converges, temperature T is fitted by minimising NLL
on the validation split. All other parameters remain frozen.

OOD DETECTOR FITTING
=====================
After calibration, the OOD energy threshold is fitted from validation logits.

EVALUATION METRICS (logged every epoch)
================
    - Runtime accuracy (top-1)
    - Latency MAE (ms)
    - Memory MAE (MB)
    - Failure AUC
    - Regret: average (predicted_latency - optimal_latency) / optimal_latency

USAGE
=====
    python -m training.train \\
        --parquet data/benchmarks.parquet \\
        --model-features data/model_features.pkl \\
        --output model_bundle_v2/decision_net.pt \\
        --epochs 50 \\
        --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model.multi_head_model import DeploymentDecisionNet, N_RUNTIMES
from src.model.ood_detector import EnergyOODDetector
from src.training.dataset import DeploymentDataset, split_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("training.train")

# ── Loss hyperparameters ──────────────────────────────────────────────────────
LAMBDA_LATENCY   = 1.0
LAMBDA_MEMORY    = 0.5
LAMBDA_FAILURE   = 2.0
GAMMA_INVALID    = 0.5   # invalid decision penalty weight


# ── Multi-Objective Loss ──────────────────────────────────────────────────────

class MultiObjectiveLoss(nn.Module):
    """
    Combined loss for DeploymentDecisionNet.

    CE(runtime) + λ1·MSE(latency) + λ2·MSE(memory) + λ3·BCE(failure)
    + γ·invalid_penalty

    No rule-based rejection. The penalty term learns to disfavour sub-optimal
    runtimes through gradient descent rather than hard exclusion.
    """

    def __init__(
        self,
        lambda_latency:  float = LAMBDA_LATENCY,
        lambda_memory:   float = LAMBDA_MEMORY,
        lambda_failure:  float = LAMBDA_FAILURE,
        gamma_invalid:   float = GAMMA_INVALID,
        class_weights:   torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lambda_latency = lambda_latency
        self.lambda_memory  = lambda_memory
        self.lambda_failure = lambda_failure
        self.gamma_invalid  = gamma_invalid

        self.ce_loss  = nn.CrossEntropyLoss(weight=class_weights)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        runtime_logits:  torch.Tensor,   # (B, N_RUNTIMES)
        latency_pred:    torch.Tensor,   # (B, N_RUNTIMES)
        memory_pred:     torch.Tensor,   # (B, N_RUNTIMES)
        failure_logits:  torch.Tensor,   # (B, N_RUNTIMES)
        runtime_label:   torch.Tensor,   # (B,) long
        latency_target:  torch.Tensor,   # (B, N_RUNTIMES)
        memory_target:   torch.Tensor,   # (B, N_RUNTIMES)
        failure_target:  torch.Tensor,   # (B, N_RUNTIMES)
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            total_loss: scalar tensor
            breakdown:  dict of component loss values (for logging)
        """
        # ── Component losses ──────────────────────────────────────────────────
        L_ce      = self.ce_loss(runtime_logits, runtime_label)
        L_latency = self.mse_loss(latency_pred, latency_target)
        L_memory  = self.mse_loss(memory_pred, memory_target)
        L_failure = self.bce_loss(failure_logits, failure_target)

        # ── Invalid decision penalty ──────────────────────────────────────────
        # For each example, encourage high probability on the optimal runtime.
        # One-hot encode the optimal label and compute BCE against the softmax.
        B = runtime_logits.size(0)
        optimal_mask = torch.zeros(B, N_RUNTIMES, device=runtime_logits.device)
        optimal_mask.scatter_(1, runtime_label.unsqueeze(1), 1.0)

        probs = torch.softmax(runtime_logits, dim=-1)
        # BCE: encourage model to output prob=1 at optimal index
        L_invalid = -torch.mean(
            optimal_mask * torch.log(probs + 1e-8)
        )

        # ── Total loss ────────────────────────────────────────────────────────
        total = (
            L_ce
            + self.lambda_latency * L_latency
            + self.lambda_memory  * L_memory
            + self.lambda_failure * L_failure
            + self.gamma_invalid  * L_invalid
        )

        breakdown = {
            "ce":      L_ce.item(),
            "latency": L_latency.item(),
            "memory":  L_memory.item(),
            "failure": L_failure.item(),
            "invalid": L_invalid.item(),
            "total":   total.item(),
        }
        return total, breakdown


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:      DeploymentDecisionNet,
    loader:     DataLoader,
    device:     torch.device,
    split_name: str = "val",
) -> dict[str, float]:
    """
    Compute evaluation metrics on a data split.

    Metrics:
        runtime_accuracy    — top-1 accuracy
        latency_mae_ms      — mean absolute error in ms (exp-space)
        memory_mae_mb       — mean absolute error in MB
        failure_auc         — AUROC for failure prediction
        regret              — (pred_latency[best] - opt_latency[opt]) / opt_latency[opt]
    """
    model.eval()

    all_preds, all_labels = [], []
    all_latency_pred, all_latency_true = [], []
    all_memory_pred,  all_memory_true  = [], []
    all_failure_pred, all_failure_true = [], []

    for batch in loader:
        feats   = batch["features"].to(device)
        labels  = batch["runtime_label"].to(device)
        lat_tgt = batch["latency_target"].to(device)
        mem_tgt = batch["memory_target"].to(device)
        fail_tgt = batch["failure_target"].to(device)

        out = model(feats)
        probs = model.calibrated_probs(out.runtime_logits)
        preds = probs.argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_latency_pred.append(out.latency_log.cpu())
        all_latency_true.append(lat_tgt.cpu())
        all_memory_pred.append(out.memory_log.cpu())
        all_memory_true.append(mem_tgt.cpu())
        all_failure_pred.append(torch.sigmoid(out.failure_logits).cpu())
        all_failure_true.append(fail_tgt.cpu())

    preds_all  = torch.cat(all_preds)
    labels_all = torch.cat(all_labels)
    lat_pred   = torch.cat(all_latency_pred)    # (N, N_RUNTIMES) log-space
    lat_true   = torch.cat(all_latency_true)
    mem_pred   = torch.cat(all_memory_pred)
    mem_true   = torch.cat(all_memory_true)
    fail_pred  = torch.cat(all_failure_pred)
    fail_true  = torch.cat(all_failure_true)

    acc = float((preds_all == labels_all).float().mean().item())

    # MAE in original space
    lat_mae = float((torch.exp(lat_pred) - torch.exp(lat_true)).abs().mean().item())
    mem_mae = float((torch.exp(mem_pred) - torch.exp(mem_true)).abs().mean().item())

    # Regret: for each example, cost of choosing predicted vs optimal runtime
    # cost_r = latency_ms[r], best = labels_all
    lat_ms_pred = torch.exp(lat_pred) - 1.0   # (N, N_RUNTIMES)
    best_lat     = lat_ms_pred.gather(1, labels_all.unsqueeze(1)).squeeze(1)   # optimal latency
    chosen_lat   = lat_ms_pred.gather(1, preds_all.unsqueeze(1)).squeeze(1)    # predicted latency
    safe_best    = best_lat.clamp(min=1.0)
    regret       = float(((chosen_lat - best_lat) / safe_best).mean().item())

    # Failure AUROC
    try:
        from sklearn.metrics import roc_auc_score
        fail_auc = roc_auc_score(
            fail_true.numpy().flatten(),
            fail_pred.numpy().flatten(),
        )
    except Exception:
        fail_auc = float("nan")

    metrics = {
        f"{split_name}/runtime_accuracy": acc,
        f"{split_name}/latency_mae_ms":   lat_mae,
        f"{split_name}/memory_mae_mb":    mem_mae,
        f"{split_name}/failure_auc":      fail_auc,
        f"{split_name}/regret":           regret,
    }
    model.train()
    return metrics


# ── Temperature Calibration ───────────────────────────────────────────────────

def calibrate_temperature(
    model:   DeploymentDecisionNet,
    loader:  DataLoader,
    device:  torch.device,
    lr:      float = 0.01,
    epochs:  int   = 30,
) -> float:
    """
    Post-hoc temperature scaling calibration.

    Freezes all weights except model.temperature and minimises NLL on
    the validation set.

    Returns the calibrated temperature value.
    """
    logger.info("Calibration: fitting temperature on validation set ...")
    model.freeze_for_calibration()
    model.to(device)

    optimizer = optim.LBFGS([model.temperature], lr=lr, max_iter=100)
    nll_loss  = nn.CrossEntropyLoss()

    def _closure():
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for batch in loader:
            feats  = batch["features"].to(device)
            labels = batch["runtime_label"].to(device)
            out = model(feats)
            T = model.temperature.clamp(min=1e-4)
            scaled_logits = out.runtime_logits / T
            loss = nll_loss(scaled_logits, labels)
            total_loss = total_loss + loss
        total_loss.backward()
        return total_loss

    for epoch in range(epochs):
        optimizer.step(_closure)

    T = float(model.temperature.item())
    logger.info("Calibration: done. T = %.4f", T)
    model.unfreeze()
    return T


# ── Class weight computation ──────────────────────────────────────────────────

def compute_class_weights(
    dataset: DeploymentDataset,
    device:  torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for CE loss."""
    from collections import Counter
    counts = Counter(dataset.label_by_model.values())
    total  = sum(counts.values())
    weights = torch.ones(N_RUNTIMES, device=device)
    for cls_idx, cnt in counts.items():
        weights[cls_idx] = total / (N_RUNTIMES * cnt)
    return weights


# ── Main Training Loop ────────────────────────────────────────────────────────

def train(
    parquet_path:      Path,
    model_features:    dict[str, np.ndarray],
    output_path:       Path,
    epochs:            int   = 50,
    batch_size:        int   = 64,
    lr:                float = 3e-4,
    weight_decay:      float = 1e-4,
    seed:              int   = 42,
    device_str:        str   = "auto",
    patience:          int   = 10,
) -> Path:
    """
    Full training pipeline:
      1. Build dataset
      2. Split train/val/test
      3. Train with multi-objective loss + early stopping
      4. Calibrate temperature
      5. Fit OOD threshold
      6. Evaluate on test set
      7. Save model bundle

    Returns path to saved model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )
    logger.info("train: using device=%s", device)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    full_dataset = DeploymentDataset(
        parquet_path   = parquet_path,
        model_features = model_features,
        augment        = False,  # augmentation is applied only on train subset
    )
    train_set, val_set, test_set = split_dataset(full_dataset, seed=seed)

    # Enable augmentation for training subset only
    if hasattr(train_set, "dataset"):
        # Wrap so augment fires only for train indices
        train_set.dataset.augment = False  # we handle it via AugmentedSubset below

    class AugmentedSubset:
        """Wraps a Subset to enable augmentation only during training."""
        def __init__(self, subset):
            self.subset = subset
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            item = self.subset[idx]
            # Apply noise to continuous features
            feats = item["features"].clone()
            noise = torch.randn_like(feats) * 0.01 * (feats.abs() + 1e-8)
            bool_indices = [2, 6, 7, 8, 9, 10, 11, 12, 15, 16, 28, 29, 30, 31, 32]
            noise[bool_indices] = 0.0
            item = dict(item)
            item["features"] = feats + noise
            return item

    train_set = AugmentedSubset(train_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ── 2. Model + Loss + Optimizer ───────────────────────────────────────────
    model = DeploymentDecisionNet().to(device)

    class_weights = compute_class_weights(full_dataset, device)
    criterion     = MultiObjectiveLoss(class_weights=class_weights).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── 3. Training loop with early stopping ──────────────────────────────────
    best_val_acc    = 0.0
    best_state      = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            feats    = batch["features"].to(device)
            labels   = batch["runtime_label"].to(device)
            lat_tgt  = batch["latency_target"].to(device)
            mem_tgt  = batch["memory_target"].to(device)
            fail_tgt = batch["failure_target"].to(device)

            optimizer.zero_grad()
            out = model(feats)

            total_loss, breakdown = criterion(
                runtime_logits = out.runtime_logits,
                latency_pred   = out.latency_log,
                memory_pred    = out.memory_log,
                failure_logits = out.failure_logits,
                runtime_label  = labels,
                latency_target = lat_tgt,
                memory_target  = mem_tgt,
                failure_target = fail_tgt,
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_sum += breakdown["total"]
            n_batches += 1

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, device, "val")
        val_acc = val_metrics["val/runtime_accuracy"]

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_acc=%.4f  latency_mae=%.1fms  "
            "memory_mae=%.1fMB  regret=%.4f  failure_auc=%.4f",
            epoch, epochs,
            epoch_loss_sum / max(n_batches, 1),
            val_acc,
            val_metrics["val/latency_mae_ms"],
            val_metrics["val/memory_mae_mb"],
            val_metrics["val/regret"],
            val_metrics["val/failure_auc"],
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logger.info("Epoch %d: new best val_acc=%.4f — checkpoint saved", epoch, val_acc)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    # ── Restore best checkpoint ───────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Training complete. Best val_accuracy=%.4f", best_val_acc)

    # ── 4. Temperature calibration ────────────────────────────────────────────
    T = calibrate_temperature(model, val_loader, device)

    # ── 5. OOD threshold fitting ──────────────────────────────────────────────
    logger.info("Fitting OOD energy threshold on validation set ...")
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in val_loader:
            feats = batch["features"].to(device)
            out   = model(feats)
            all_logits.append(out.runtime_logits.cpu())
    val_logits = torch.cat(all_logits, dim=0)

    ood_detector = EnergyOODDetector(temperature=T)
    ood_detector.fit(val_logits)

    # ── 6. Test set evaluation ────────────────────────────────────────────────
    test_metrics = evaluate(model, test_loader, device, "test")
    logger.info("Test metrics:")
    for k, v in test_metrics.items():
        logger.info("  %s = %.4f", k, v)

    # ── 7. Save model bundle ──────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    ood_path = output_path.parent / "ood_detector.json"
    ood_detector.save(ood_path)

    # Save training metadata
    meta = {
        "best_val_accuracy": best_val_acc,
        "temperature":       T,
        "test_metrics":      test_metrics,
        "epochs_trained":    epoch,
        "lambda_latency":    LAMBDA_LATENCY,
        "lambda_memory":     LAMBDA_MEMORY,
        "lambda_failure":    LAMBDA_FAILURE,
        "gamma_invalid":     GAMMA_INVALID,
    }
    meta_path = output_path.parent / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved model to %s, OOD to %s, meta to %s", output_path, ood_path, meta_path)

    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DeploymentDecisionNet")
    p.add_argument("--parquet",        type=Path,  required=True,  help="Benchmark Parquet file")
    p.add_argument("--model-features", type=Path,  required=True,  help="Pickle of model_features dict")
    p.add_argument("--output",         type=Path,  default=Path("model_bundle_v2/decision_net.pt"))
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         type=str,   default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.model_features, "rb") as f:
        model_features = pickle.load(f)
    train(
        parquet_path   = args.parquet,
        model_features = model_features,
        output_path    = args.output,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        seed           = args.seed,
        device_str     = args.device,
    )
