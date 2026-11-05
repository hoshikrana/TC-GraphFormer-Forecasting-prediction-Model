"""
Self-Supervised Pretraining Workflow for TC-GraphFormer Encoders and Cross-Modal Fusion.
Implements Masked Feature Modeling (MFM) to learn cross-sensor atmospheric representations
without requiring target labels.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

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


def run_pretraining(
    train_index: str = "reports/pipeline/sample_index.csv",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-4,
    d_model: int = 128,
    mask_ratio: float = 0.3,
    max_samples: int = None,
    checkpoint_dir: str = "checkpoints",
    output_report: str = "reports/pretraining_summary.json",
    device: str = None,
) -> Dict[str, Any]:
    """
    Execute self-supervised masked feature pretraining.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TC-GRAPHFORMER SELF-SUPERVISED PRETRAINING (MFM)")
    print(f"Device: {device} | Mask Ratio: {mask_ratio} | Epochs: {epochs} | Batch size: {batch_size}")
    print("=" * 60)

    dataset = TCPRIMEDDataset(sample_index_csv=train_index, partition="TRAIN", grid_size=64, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal_batch)

    model = TCGraphFormer(d_model=d_model).to(device)
    recon_head = MaskedFeatureReconstructionHead(d_model=d_model).to(device)

    params = list(model.ir_encoder.parameters()) + \
             list(model.pmw_encoder.parameters()) + \
             list(model.gprof_encoder.parameters()) + \
             list(model.radar_encoder.parameters()) + \
             list(model.env_encoder.parameters()) + \
             list(model.fusion.parameters()) + \
             list(recon_head.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    history = []
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        recon_head.train()
        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            ir = batch["ir"].to(device)
            pmw = batch["pmw"].to(device)
            gprof = batch["gprof"].to(device)
            radar = batch["radar"].to(device)
            env = batch["env"].to(device)
            masks = batch["modality_masks"].to(device)
            B = ir.size(0)

            # Unpack masks
            m_ir, m_pmw, m_gp, m_rad, m_env = masks[:, 0], masks[:, 1], masks[:, 2], masks[:, 3], masks[:, 4]

            # Extract unmasked target representations
            with torch.no_grad():
                tgt_ir = model.ir_encoder(ir, m_ir)
                tgt_pmw = model.pmw_encoder(pmw, m_pmw)
                tgt_gp = model.gprof_encoder(gprof, m_gp)
                tgt_rad = model.radar_encoder(radar, m_rad)
                tgt_env = model.env_encoder(env, m_env)
                all_targets = torch.cat([tgt_ir, tgt_pmw, tgt_gp, tgt_rad, tgt_env], dim=1) # (B, 80, d_model)

            # Generate random patch mask (mask_ratio fraction masked)
            rand_mask = (torch.rand(B, 80, 1, device=device) < mask_ratio).float()

            # Corrupt input tokens by zeroing masked ones
            t_ir = tgt_ir * (1.0 - rand_mask[:, :16])
            t_pmw = tgt_pmw * (1.0 - rand_mask[:, 16:32])
            t_gp = tgt_gp * (1.0 - rand_mask[:, 32:48])
            t_rad = tgt_rad * (1.0 - rand_mask[:, 48:64])
            t_env = tgt_env * (1.0 - rand_mask[:, 64:80])

            fused = model.fusion([t_ir, t_pmw, t_gp, t_rad, t_env], masks)
            recon = recon_head(fused)

            # Compute MSE only on masked positions
            loss = (F.mse_loss(recon, all_targets, reduction="none") * rand_mask).sum() / rand_mask.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        print(f"[Pretrain Epoch {epoch}/{epochs}] Reconstruction MSE Loss: {avg_loss:.5f}")
        history.append({"epoch": epoch, "loss_mse": round(avg_loss, 5)})

    # Save pretrained backbone
    pretrained_path = os.path.join(checkpoint_dir, "pretrained_backbone.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "d_model": d_model,
        "history": history,
    }, pretrained_path)

    summary = {
        "workflow": "Self-Supervised Masked Feature Pretraining",
        "epochs": epochs,
        "mask_ratio": mask_ratio,
        "final_loss_mse": round(history[-1]["loss_mse"], 5) if history else None,
        "checkpoint_path": pretrained_path,
        "history": history,
    }

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Pretraining complete. Pretrained weights saved to: {pretrained_path}")
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
    parser.add_argument("--output-report", default="reports/pretraining_summary.json", help="Summary report")
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
