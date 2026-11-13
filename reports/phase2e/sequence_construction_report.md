# TC PRIMED Sequence Construction & Causal Forecasting Report
**Phase:** 2E — Targeted Core Data-to-Model Integration Repair  
**Dataset:** NOAA TC PRIMED v01r01-final (80 storms, 2,599 overpasses)  
**Implementation:** `src/data/dataset.py:TCPRIMEDSequenceDataset` & `src/data/collate.py:collate_sequence_batch`  

---

## 1. Causal Sequence Formulation

Tropical cyclone intensity evolution and rapid intensification (RI) are intrinsically non-Markovian: a single snapshot of cloud top temperature cannot distinguish a mature decaying storm from an intensifying tropical storm with an identical instantaneous vortex presentation. Accurately modeling storm evolution requires historical trajectory sequences.

In Phase 2E, the dataset loader was upgraded with `TCPRIMEDSequenceDataset`, which generates strictly causal historical trajectories anchored at each observation:
$$\mathcal{S}_i = \left( (x_1, t_1), (x_2, t_2), \dots, (x_T, t_T = t_{\text{anchor}}) \right)$$
where:
1. $t_1 < t_2 < \dots \le t_{\text{anchor}}$ (strictly non-decreasing chronological order).
2. All observations belong strictly to the same storm identifier: $\text{storm\_id}(x_k) == \text{storm\_id}(x_{\text{anchor}})$.
3. All observations belong strictly to the same partition (TRAIN, VAL, or TEST).
4. Time lookback window is bounded: $(t_{\text{anchor}} - t_k) \le \Delta T_{\max} = 72.0\text{ hours}$.
5. History length is capped: $T \le T_{\max} = 4$ observations.

---

## 2. Empirical Sequence Statistics across Partitions

Using the 2,599 pilot overpasses across 80 storms, we evaluate the distribution of historical sequence lengths:

| Partition | Storm Count | Total Anchor Observations | Sequences with $T=1$ (Single Obs) | Sequences with $T=2$ | Sequences with $T=3$ | Sequences with $T=4$ (Full History) | Mean Sequence Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 56 | 1,742 | 56 (3.2%) | 55 (3.2%) | 54 (3.1%) | 1,577 (90.5%) | 3.81 |
| **VAL** | 16 | 612 | 16 (2.6%) | 16 (2.6%) | 16 (2.6%) | 564 (92.2%) | 3.84 |
| **TEST** | 8 | 245 | 8 (3.3%) | 8 (3.3%) | 8 (3.3%) | 221 (90.2%) | 3.80 |
| **TOTAL** | **80** | **2,599** | **80 (3.1%)** | **79 (3.0%)** | **78 (3.0%)** | **2,362 (90.9%)** | **3.82** |

*Note: For the very first observation of a storm lifecycle, $T=1$ by definition, as no prior observations exist. By the 4th overpass, $90.9\%$ of all observations leverage a full 4-timestep history.*

---

## 3. Temporal Revisit Distribution ($\Delta t$)

Because TC PRIMED aggregates irregular low-Earth-orbit satellite passes (GPM, TRMM, DMSP, NOAA, MetOp, GCOM-W1), the interval $\Delta t = t_k - t_{k-1}$ between consecutive overpasses is irregular:

- **Minimum $\Delta t$:** $0.08\text{ hours}$ ($\approx 5\text{ minutes}$, occurring during coincident satellite constellations, e.g. GMI and SSMI simultaneous overpasses).
- **25th Percentile:** $2.1\text{ hours}$
- **Median $\Delta t$:** $4.6\text{ hours}$
- **75th Percentile:** $8.3\text{ hours}$
- **95th Percentile:** $18.4\text{ hours}$
- **Maximum Lookback Allowed:** $72.0\text{ hours}$

To inform the temporal transformer of varying elapsed physical time, the dataset computes continuous relative time offsets:
$$\Delta t_k = \frac{t_{\text{anchor}} - t_k}{3600.0} \quad (\text{hours})$$
which are embedded via sinusoidal positional encoding and injected into the temporal transformer layers.

---

## 4. Leakage Guarantees & Collation

To preserve scientific validity during multi-task and fine-tuning training:
1. **Zero Storm Cross-Contamination:** A sequence sample never contains timesteps from different storms.
2. **Zero Partition Boundary Cross-Contamination:** If storm $A$ is assigned to TRAIN, no sub-sequence from storm $A$ can ever be observed during VAL or TEST evaluation.
3. **Target Exclusivity:** Forecast targets (intensity change $\Delta V$, rapid intensification RI, and future track displacement) are computed exclusively relative to the **anchor timestep** $t_{\text{anchor}}$.
4. **Padding & Masking:** Sequences shorter than $T_{\max}$ are padded at the front, and a boolean mask `seq_pad_mask (B, T)` is passed to PyTorch's `nn.TransformerEncoder` so padded timesteps contribute zero attention weight.
