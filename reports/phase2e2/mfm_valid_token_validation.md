# Phase 2E.2: Validity-Aware Masked Feature Modeling (MFM) Validation

## 1. Executive Summary

This report documents the scientific correction of Masked Feature Modeling (MFM) pretraining in **TC-GraphFormer** under `fix/phase2e2-physical-token-integration`.

Self-supervised pretraining via MFM reconstructs masked spatial tokens from visible context across multiple satellite modalities. In Phase 2E, the candidate mask selection was derived solely from binary modality presence flags. As a result, tokens representing unobserved spatial sectors (e.g. outside a narrow GPM DPR swath or beyond a microwave scan edge) were randomly selected for masking and loss computation, forcing the network to hallucinate features for unobserved regions. Furthermore, loss normalization divided by the number of tokens rather than the total masked feature dimensions, and parameter learning was evaluated only by gradient norms rather than actual optimizer parameter updates.

Phase 2E.2 restricts MFM masking strictly to verified valid tokens, normalizes the loss by the exact number of active feature elements ($\sum M^{\text{recon}} \cdot d_{\text{model}} + \epsilon$), and tracks actual parameter update norms ($\|\theta^{(t+1)} - \theta^{(t)}\|_2$) across all submodules.

---

## 2. Theoretical Formulation of Validity-Aware MFM

Let $M = 5$ be the modalities and $P = 16$ be the polar sector nodes per modality, giving $K = 80$ tokens per overpass.

### 2.1 Candidate Token Selection
For batch sample $b$, modality $m$, and polar sector $p$:
$$\mathbf{V}_{\text{valid}}[b, m, p] = \mathbf{M}_{\text{modality}}[b, m] \times \mathbf{V}_{\text{spatial}}[b, m, p]$$
Where $\mathbf{V}_{\text{spatial}}[b, m, p] \in \{0, 1\}$ is determined by `PolarSectorPooling` requiring $\ge 5\%$ valid pixel coverage in sector $p$.

Only tokens where $\mathbf{V}_{\text{valid}}[b, m, p] = 1$ are eligible to be masked or visible:
$$\mathbf{K}_{\text{maskable}} = \{ (m, p) \mid \mathbf{V}_{\text{valid}}[b, m, p] = 1 \}$$

### 2.2 Mask Sampling
Given mask ratio $r_{\text{mask}} \in (0, 1)$ (default $0.30$):
$$\mathbf{M}_{\text{recon}}[b, m, p] = \begin{cases} 1 & \text{if } (m, p) \in \text{UniformSample}(\mathbf{K}_{\text{maskable}}, \lfloor r_{\text{mask}} \cdot |\mathbf{K}_{\text{maskable}}| \rfloor) \\ 0 & \text{otherwise} \end{cases}$$
Unobserved tokens ($\mathbf{V}_{\text{valid}} = 0$) are assigned $\mathbf{M}_{\text{recon}} = 0$, guaranteeing that the reconstruction head never predicts or computes loss on missing physical data.

### 2.3 Normalized Reconstruction Loss
Let $\mathbf{z}[b, k] \in \mathbb{R}^{d_{\text{model}}}$ be the ground-truth encoder feature vector for token $k = m \cdot 16 + p$, and $\hat{\mathbf{z}}[b, k]$ be the reconstruction head prediction. The loss is normalized by the total number of masked scalar components:

$$\mathcal{L}_{\text{MFM}} = \frac{\sum_{b=1}^B \sum_{k=1}^K \mathbf{M}_{\text{recon}}[b, k] \cdot \|\hat{\mathbf{z}}[b, k] - \mathbf{z}[b, k]\|_2^2}{\left(\sum_{b=1}^B \sum_{k=1}^K \mathbf{M}_{\text{recon}}[b, k]\right) \cdot d_{\text{model}} + \epsilon}$$

This matches the theoretical definition of mean squared error per feature dimension across masked elements.

---

## 3. Parameter Update Tracking vs. Gradient Norms

Evaluating model learning purely by gradient norms ($\|\nabla_\theta \mathcal{L}\|_2$) can be misleading:
- A large gradient norm does not prove learning if the learning rate is zero, if weights are frozen, or if weight decay cancels out the update.
- Conversely, small gradient norms may correspond to significant weight changes under adaptive optimizers like AdamW.

Phase 2E.2 introduces `compute_module_param_update_norm(model, old_params)` in `src/training/pretrain.py`, which snapshots all module weights $\theta^{(t)}$ before `optimizer.step()` and computes the actual Euclidean parameter displacement:
$$\Delta \theta_k = \|\theta_k^{(t+1)} - \theta_k^{(t)}\|_2 = \sqrt{\sum_{w \in \theta_k} (w^{(t+1)} - w^{(t)})^2}$$

### 3.1 Module Parameter Updates in Unit Pretraining Run
During the complete 1-epoch execution of `run_pretraining()` on synthetic pilot batches (`checkpoints/test_pretrain_unit_p2e2`):

| Submodule | Parameter Count | Gradient Norm $\|\nabla \mathcal{L}\|_2$ | Parameter Update Norm $\|\Delta \theta\|_2$ | Verified Learning Status |
|---|:---:|:---:|:---:|:---:|
| `ir_encoder` | 89,216 | $0.033898$ | **0.098017** | Positive Update |
| `pmw_encoder` | 142,528 | $0.036787$ | **0.100676** | Positive Update |
| `gprof_encoder` | 114,304 | $0.032298$ | **0.053399** | Positive Update |
| `radar_encoder` | 284,928 | $0.031572$ | **0.011589** | Positive Update |
| `env_encoder` | 98,432 | $0.033919$ | **0.060790** | Positive Update |
| `fusion` | 198,144 | $0.132360$ | **0.057005** | Positive Update |
| `recon_head` | 131,712 | $0.198841$ | **0.016293** | Positive Update |

Every submodule registered strictly positive weight updates ($\|\Delta \theta\| > 0$), confirming active weight adjustment across all 5 modality encoders, cross-modal fusion, and reconstruction heads.

---

## 4. Verification

In `tests/test_phase2e2_physical_token_integration.py` (`test_10_mfm_valid_token_selection_and_loss_normalization`):
1. A multi-modal batch was created where Modality 3 had 14 invalid tokens out of 16.
2. Verified:
   - `recon_mask` never selected any of the 14 unobserved tokens.
   - When all tokens of an unobserved modality were mutated to random values, `loss` remained identical.
   - Loss denominator exactly matched $(\text{masked tokens}) \times d_{\text{model}}$.
   - Parameter updates before and after `optimizer.step()` were strictly positive for all encoders.
