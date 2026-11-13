"""
Rapid Intensification Class Imbalance Experiments.
Compares:
1. Standard BCE (pos_weight = 1.0)
2. Weighted BCE (pos_weight = 10.0)
3. Class-Ratio BCE (pos_weight = 33.3, based on 1732 / 52)
4. Focal Loss (alpha = 0.25, gamma = 2.0)
Evaluates on the validation partition without leaking test data.
Outputs reports/phase2c/ri_class_imbalance_experiments.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        return loss.mean()


def compute_pr_auc(probs: np.ndarray, targets: np.ndarray, num_thresholds: int = 101) -> float:
    """Compute Area under the Precision-Recall Curve (PR-AUC) via trapezoidal integration."""
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    precisions = []
    recalls = []

    for th in thresholds:
        pred_bin = (probs >= th).astype(int)
        tp = np.sum((pred_bin == 1) & (targets == 1))
        fp = np.sum((pred_bin == 1) & (targets == 0))
        fn = np.sum((pred_bin == 0) & (targets == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    # Sort by recall ascending
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_idx = np.argsort(recalls)
    rec_sorted = recalls[sorted_idx]
    prec_sorted = precisions[sorted_idx]

    # Trapezoidal integration
    if hasattr(np, "trapezoid"):
        pr_auc = np.trapezoid(prec_sorted, rec_sorted)
    elif hasattr(np, "trapz"):
        pr_auc = np.trapz(prec_sorted, rec_sorted)
    else:
        pr_auc = np.sum((prec_sorted[:-1] + prec_sorted[1:]) / 2.0 * np.diff(rec_sorted))
    return float(np.clip(pr_auc, 0.0, 1.0))


def run_ri_experiments(sample_index_csv: str = "reports/pipeline/sample_index.csv",
                       output_csv: str = "reports/phase2e/ri_class_imbalance_experiments.csv"):
    df = pd.read_csv(sample_index_csv)

    # Filter to valid +24h intensity change observations
    df_valid = df[df["is_rapid_intensification_24h"].notna()].copy()
    train_df = df_valid[df_valid["partition"] == "TRAIN"]
    val_df = df_valid[df_valid["partition"] == "VAL"]

    print(f"[RI EXPERIMENT] Train valid RI samples: {len(train_df)} (Pos: {(train_df['is_rapid_intensification_24h']==1).sum()})")
    print(f"[RI EXPERIMENT] Val valid RI samples: {len(val_df)} (Pos: {(val_df['is_rapid_intensification_24h']==1).sum()})")

    # Features: current intensity, storm speed, distance to land
    feature_cols = ["intensity_vmax_kt", "storm_speed_mps", "distance_to_land_km"]
    X_train_raw = train_df[feature_cols].fillna(0.0).values
    y_train = train_df["is_rapid_intensification_24h"].values.astype(np.float32)

    X_val_raw = val_df[feature_cols].fillna(0.0).values
    y_val = val_df["is_rapid_intensification_24h"].values.astype(np.float32)

    # Standardize features using TRAIN partition only
    mean_X = np.mean(X_train_raw, axis=0)
    std_X = np.std(X_train_raw, axis=0) + 1e-6
    X_train = (X_train_raw - mean_X) / std_X
    X_val = (X_val_raw - mean_X) / std_X

    # PyTorch Tensors
    xt_t = torch.tensor(X_train, dtype=torch.float32)
    yt_t = torch.tensor(y_train, dtype=torch.float32)
    xv_t = torch.tensor(X_val, dtype=torch.float32)
    yv_t = torch.tensor(y_val, dtype=torch.float32)

    # 4 Loss configurations
    configs = [
        ("Standard BCE", "BCEWithLogitsLoss (pos_weight=1.0)", nn.BCEWithLogitsLoss()),
        ("Weighted BCE (w=10)", "BCEWithLogitsLoss (pos_weight=10.0)", nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))),
        ("Class-Ratio BCE (w=33.3)", "BCEWithLogitsLoss (pos_weight=33.3)", nn.BCEWithLogitsLoss(pos_weight=torch.tensor([33.3]))),
        ("Focal Loss", "FocalLoss (alpha=0.25, gamma=2.0)", FocalLoss(alpha=0.25, gamma=2.0)),
    ]

    results = []

    for name, desc, criterion in configs:
        torch.manual_seed(42)
        # 2-layer classifier
        model = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        # Train 100 epochs on train split
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            logits = model(xt_t).squeeze(-1)
            loss = criterion(logits, yt_t)
            loss.backward()
            optimizer.step()

        # Evaluate on validation split
        model.eval()
        with torch.no_grad():
            val_logits = model(xv_t).squeeze(-1)
            val_probs = torch.sigmoid(val_logits).numpy()

        # Standard threshold 0.5
        val_pred = (val_probs >= 0.5).astype(int)
        tp = np.sum((val_pred == 1) & (y_val == 1))
        fp = np.sum((val_pred == 1) & (y_val == 0))
        tn = np.sum((val_pred == 0) & (y_val == 0))
        fn = np.sum((val_pred == 0) & (y_val == 1))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        pr_auc = compute_pr_auc(val_probs, y_val)

        res = {
            "configuration": name,
            "loss_function": desc,
            "train_samples": len(train_df),
            "train_positives": int(yt_t.sum().item()),
            "val_samples": len(val_df),
            "val_positives": int(yv_t.sum().item()),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_alarm_rate": round(far, 4),
            "pr_auc": round(pr_auc, 4),
            "recommendation_status": "Candidate" if f1 > 0.1 else "Inferior"
        }
        results.append(res)
        print(f"[{name:25s}] Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f} | FAR: {far:.3f} | PR-AUC: {pr_auc:.3f}")

    df_res = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_res.to_csv(output_csv, index=False)
    print(f"\n[OK] RI class imbalance experiment table saved to: {output_csv}")
    return df_res


if __name__ == "__main__":
    run_ri_experiments()
