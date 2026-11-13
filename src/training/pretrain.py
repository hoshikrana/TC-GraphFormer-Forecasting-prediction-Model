"""
Self-Supervised Pretraining Workflow for TC-GraphFormer Encoders and Cross-Modal Fusion.
Implements Masked Feature Modeling (MFM) to learn cross-sensor atmospheric representations
with autograd gradients propagating into modality encoders and fusion layers.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import TCPRIMEDDataset
from src.data.collate import collate_multimodal_batch
from src.models.tc_graphformer import TCGraphFormer


class MaskedFeatureReconstructionHead(nn.Module):
    """Reconstructs masked spatial token features from fused representation."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def compute_module_grad_norm(module: nn.Module) -> float:
    """Compute L2 gradient norm across all parameters in a module."""
    total_norm_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            norm_val = p.grad.data.norm(2).item()
            total_norm_sq += norm_val ** 2
    return total_norm_sq ** 0.5


def run_pretraining(
    train_index: str = "reports/pipeline/sample_index.csv",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-4,
    d_model: int = 128,
    mask_ratio: float = 0.3,
    max_samples: int = None,
    checkpoint_dir: str = "checkpoints",
    output_report: str = "reports/phase2e/pretraining_summary.json",
    gradient_report_path: str = "reports/phase2e/pretraining_gradient_report.json",
    device: str = None,
) -> Dict[str, Any]:
    """
    Execute self-supervised masked feature pretraining with verifiable encoder gradients.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TC-GRAPHFORMER SELF-SUPERVISED PRETRAINING (MFM - PHASE 2E)")
    print(f"Device: {device} | Mask Ratio: {mask_ratio} | Epochs: {epochs} | Batch size: {batch_size}")
    print("=" * 60)

    dataset = TCPRIMEDDataset(
        sample_index_csv=train_index, partition="TRAIN", grid_size=64, max_samples=max_samples
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal_batch
    )

    model = TCGraphFormer(d_model=d_model).to(device)
    recon_head = MaskedFeatureReconstructionHead(d_model=d_model).to(device)

    params = (
        list(model.ir_encoder.parameters()) +
        list(model.pmw_encoder.parameters()) +
        list(model.gprof_encoder.parameters()) +
        list(model.radar_encoder.parameters()) +
        list(model.env_encoder.parameters()) +
        list(model.fusion.parameters()) +
        list(recon_head.parameters())
    )

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    history = []
    gradient_logs = []
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_report), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        recon_head.train()
        total_loss = 0.0
        total_batches = 0
        total_valid_tokens = 0
        total_masked_tokens = 0
        total_visible_tokens = 0

        latest_grad_norms = {}

        for batch in loader:
            ir = batch["ir"].to(device)
            pmw = batch["pmw"].to(device)
            gprof = batch["gprof"].to(device)
            radar = batch["radar"].to(device)
            env = batch["env"].to(device)
            masks = batch["modality_masks"].to(device) # (B, 5)
            B = ir.size(0)

            ch_masks = {k: v.to(device) for k, v in batch["channel_masks"].items()}

            m_ir = masks[:, 0]
            m_pmw = masks[:, 1]
            m_gp = masks[:, 2]
            m_rad = masks[:, 3]
            m_env = masks[:, 4]

            # 1. Student forward pass WITH autograd tracking
            t_ir = model.ir_encoder(ir, m_ir)
            t_pmw = model.pmw_encoder(pmw, m_pmw, channel_mask=ch_masks.get("pmw"))
            t_gp = model.gprof_encoder(gprof, m_gp, channel_mask=ch_masks.get("gprof"))
            t_rad = model.radar_encoder(radar, m_rad)
            t_env = model.env_encoder(env, m_env, channel_mask=ch_masks.get("env"))

            all_student_tokens = torch.cat([t_ir, t_pmw, t_gp, t_rad, t_env], dim=1) # (B, 80, d_model)

            # 2. Target representations: detached from gradient graph
            all_targets = all_student_tokens.detach().clone()

            # 3. Construct token-level validity mask: (B, 80, 1)
            # Only tokens belonging to genuinely present modalities can be masked or reconstructed
            valid_tok_mask = torch.cat([
                m_ir.view(B, 1, 1).repeat(1, 16, 1),
                m_pmw.view(B, 1, 1).repeat(1, 16, 1),
                m_gp.view(B, 1, 1).repeat(1, 16, 1),
                m_rad.view(B, 1, 1).repeat(1, 16, 1),
                m_env.view(B, 1, 1).repeat(1, 16, 1),
            ], dim=1) # (B, 80, 1)

            # 4. Generate random mask strictly over valid tokens
            mask_to_recon = torch.zeros((B, 80, 1), dtype=torch.float32, device=device)

            for b in range(B):
                valid_idx = torch.where(valid_tok_mask[b, :, 0] > 0.5)[0]
                n_v = len(valid_idx)
                if n_v > 1:
                    # Choose number of tokens to mask respecting mask_ratio
                    n_mask = max(1, min(n_v - 1, int(round(n_v * mask_ratio))))
                    perm = torch.randperm(n_v)
                    chosen = valid_idx[perm[:n_mask]]
                    mask_to_recon[b, chosen, 0] = 1.0
                elif n_v == 1:
                    mask_to_recon[b, valid_idx[0], 0] = 1.0

            n_valid_batch = int(valid_tok_mask.sum().item())
            n_masked_batch = int(mask_to_recon.sum().item())
            n_visible_batch = n_valid_batch - n_masked_batch

            total_valid_tokens += n_valid_batch
            total_masked_tokens += n_masked_batch
            total_visible_tokens += n_visible_batch

            # 5. Corrupt student tokens by zeroing masked valid positions
            # (Visible tokens keep active computational path to encoders)
            corrupted_tokens = all_student_tokens * (1.0 - mask_to_recon)

            s_ir = corrupted_tokens[:, 0:16]
            s_pmw = corrupted_tokens[:, 16:32]
            s_gp = corrupted_tokens[:, 32:48]
            s_rad = corrupted_tokens[:, 48:64]
            s_env = corrupted_tokens[:, 64:80]

            # Cross-modal fusion and reconstruction head
            fused = model.fusion([s_ir, s_pmw, s_gp, s_rad, s_env], masks)
            recon = recon_head(fused)

            # 6. Loss computed ONLY on masked valid tokens
            diff = F.mse_loss(recon, all_targets, reduction="none") # (B, 80, d_model)
            denom = mask_to_recon.sum().clamp(min=1.0)
            loss = (diff * mask_to_recon).sum() / denom

            optimizer.zero_grad()
            loss.backward()

            # Record gradient norms
            latest_grad_norms = {
                "ir_encoder": round(compute_module_grad_norm(model.ir_encoder), 6),
                "pmw_encoder": round(compute_module_grad_norm(model.pmw_encoder), 6),
                "gprof_encoder": round(compute_module_grad_norm(model.gprof_encoder), 6),
                "radar_encoder": round(compute_module_grad_norm(model.radar_encoder), 6),
                "env_encoder": round(compute_module_grad_norm(model.env_encoder), 6),
                "fusion": round(compute_module_grad_norm(model.fusion), 6),
                "recon_head": round(compute_module_grad_norm(recon_head), 6),
            }

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        print(
            f"[Pretrain Epoch {epoch}/{epochs}] MSE Loss: {avg_loss:.5f} | "
            f"Valid Tokens: {total_valid_tokens} | Masked: {total_masked_tokens} | Visible: {total_visible_tokens}"
        )
        print(f"  Gradient Norms: {latest_grad_norms}")

        history.append({
            "epoch": epoch,
            "loss_mse": round(avg_loss, 5),
            "valid_tokens": total_valid_tokens,
            "masked_tokens": total_masked_tokens,
            "visible_tokens": total_visible_tokens,
            "gradient_norms": latest_grad_norms,
        })
        gradient_logs.append(latest_grad_norms)

    # Save pretrained backbone
    pretrained_path = os.path.join(checkpoint_dir, "pretrained_backbone.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "d_model": d_model,
        "history": history,
        "latest_grad_norms": latest_grad_norms,
    }, pretrained_path)

    summary = {
        "workflow": "Self-Supervised Masked Feature Pretraining (Phase 2E)",
        "epochs": epochs,
        "mask_ratio": mask_ratio,
        "final_loss_mse": round(history[-1]["loss_mse"], 5) if history else None,
        "checkpoint_path": pretrained_path,
        "history": history,
        "final_gradient_norms": latest_grad_norms,
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(gradient_report_path, "w", encoding="utf-8") as f:
        json.dump(latest_grad_norms, f, indent=2)

    print(f"\n[OK] Pretraining complete. Pretrained weights saved to: {pretrained_path}")
    print(f"[OK] Pretraining gradient report saved to: {gradient_report_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TC-GraphFormer Self-Supervised Pretraining")
    parser.add_argument("--epochs", type=int, default=3, help="Pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--mask-ratio", type=float, default=0.3, help="Fraction of tokens masked")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--output-report", default="reports/phase2e/pretraining_summary.json", help="Summary report")
    args = parser.parse_args()

    run_pretraining(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        mask_ratio=args.mask_ratio,
        max_samples=args.max_samples,
        checkpoint_dir=args.checkpoint_dir,
        output_report=args.output_report,
    )
