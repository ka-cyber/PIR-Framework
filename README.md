# PIR-Framework: Physiological Information Reliability

A unified state-space and contextual-bandit control framework for allocating communication redundancy and computation placement in multimodal cardiovascular sensing.

PIR-Framework brings together two related ideas:

- **CardioFusion-AI** — multimodal ECG/PPG signal-quality indices (SQI) and fusion.
- **APC-RLNC** — adaptive random linear network coding (RLNC) over GF(2^8) under bursty Gilbert–Elliott erasure channels, with energy and compute-latency models.

PIR treats the question of how much task-relevant information survives the sense–encode–transmit–decode pipeline, per joule and within a latency budget, as a resource-allocation problem. A contextual bandit selects coding redundancy and local-versus-edge processing based on medical, wireless, energy, and compute state.

This repository is a standalone implementation. It does not import code from the two prior repositories; it reimplements the relevant mechanisms (GF(256) coding utilities, a signal-quality index, and Gilbert–Elliott channel modelling) and documents the equations and references used by each module.

> **Scope of the reported study:** the primary evaluation uses synthetic ECG/PPG windows and an analytic decoding-probability model. It is not a clinical validation study, and the reported simulation results should not be interpreted as evidence of performance on patients or deployed hardware.

---

## 1. Architecture

```text
pir_framework/
├── state_space.py       Joint state X_t = [S_m, S_w, S_e, S_c]
├── pir_controller.py    LinUCB / Thompson Sampling contextual bandit
├── data_pipeline.py    Synthetic + PhysioNet-format data and SQI estimator
├── evaluate.py         Five-policy comparative evaluation harness
├── coding/             GF(256) arithmetic and RLNC utilities
├── channel/            Gilbert–Elliott channel with SNR-linked BER
└── utils/              Metrics, LaTeX export, and Matplotlib plotting
```

### 1.1 Unified state-space engine (`state_space.py`)

The joint state vector is \(X_t = [S_m, S_w, S_e, S_c]\).

| Component | Contents | Source |
|---|---|---|
| **Medical, \(S_m\)** | Per-modality SQI (`sqi_ecg`, `sqi_ppg`), modality-presence flags, and Physiological Information Value (`PIV`) | `MedicalState`, `compute_piv()` |
| **Wireless, \(S_w\)** | SNR (dB) and erasure probability from a two-state Gilbert–Elliott channel | `channel/gilbert_elliott.py` |
| **Energy, \(S_e\)** | Residual battery and per-packet energy expenditure, \(E = E_{tx}n_{tx} + E_{rx}n_{rx} + E_{comp}n_{ops}\) | `EnergyModel`, `EnergyState` |
| **Compute, \(S_c\)** | Modelled local and edge inference latency plus transmission latency | `ComputeModel` |

PIV combines a signal-quality term with an availability-gated penalty for cross-modal heart-rate disagreement. The quality term uses a presence-weighted average: a missing modality is excluded rather than automatically assigned a quality of zero. PIV is a hand-designed relative-value proxy, not a probability, clinically validated score, or information-theoretic guarantee.

The wireless model links SNR and erasure probability through a BPSK/AWGN bit-error model, \(BER = Q(\sqrt{2\,SNR_{linear}})\), followed by \(p_{erasure} = 1 - (1-BER)^L\) for an \(L\)-bit payload.

**Causal context, not future information.** `StateSpaceEngine` exposes `observe_context()` for previously observed telemetry separately from `act()`, which draws the channel realization used to score the current action. The bandit does not receive the not-yet-realized channel draw when choosing an action.

### 1.2 Contextual-bandit controller (`pir_controller.py`)

The repository implements two disjoint-linear-model contextual bandits:

- **LinUCB** (Li et al., 2010): ridge regression per arm with an upper-confidence-bound exploration bonus and rank-one Sherman–Morrison updates.
- **Thompson Sampling** (Agrawal & Goyal, 2013): Bayesian linear regression per arm with posterior sampling for exploration.

The default action set is \(R \in \{2,8,20\}\) redundant coded packets crossed with `local` or `edge` placement, for six arms total.

The model-defined utility and reward are:

```text
U_PIR(a, s) = PIV * P_decode(K, R, p_erasure(s))

reward(a, s) = U_PIR(a, s)
              - lambda_energy  * E(a, s)
              - lambda_latency * softplus(L(a, s) - 150 ms)
              - lambda_power   * softplus(P(a, s) - 100 mW)
```

`P_decode` is evaluated with a closed-form binomial-tail expression for an ideal erasure code. The primary results do **not** use a packet-level RLNC encoder/decoder in the end-to-end evaluation. Energy, latency, and power are modelled quantities; the constants were not fitted to a physical device.

### 1.3 Data pipeline (`data_pipeline.py`)

- `PhysioNetRecordLoader` is a dependency-guarded wrapper around `wfdb.rdrecord` for local WFDB-format records and best-effort ECG/PPG channel selection. It was not evaluated against downloaded real records in the reported experiment.
- `SyntheticECGPPGGenerator`, used by the main evaluation, generates synthetic ECG/PPG windows with controllable degradation and six regimes: both clean, ECG degraded, PPG degraded, both degraded, ECG missing, and PPG missing.
- `estimate_sqi_and_hr` estimates signal quality and heart rate using feasibility checks and template-correlation quality scoring.

### 1.4 Evaluation harness (`evaluate.py`)

Five policies are evaluated on the same per-seed physiological stream and channel trajectory using common random numbers.

| Policy | Behaviour |
|---|---|
| **Fixed-Rate** | Static action: \(K=32, R=8\), local processing. |
| **Always-Local** | Fixed high redundancy, \(R=20\), with local processing. |
| **Always-Edge** | Fixed high redundancy, \(R=20\), with edge processing. |
| **Heuristic-Rule** | Hand-written rule using the EWMA erasure estimate to select redundancy and a battery threshold to select placement. **It does not read PIV when selecting actions.** PIV remains part of the common reward calculation for all policies. |
| **PIR-LinUCB** | Learned contextual-bandit policy selecting redundancy and placement online from its context and reward feedback. |

Each seed contains 1,200 sequential windows (200 in each of six regimes). Results in the paper are steady-state metrics, computed over the last 50% of each session, and then aggregated across seeds. The full-session behaviour can differ because the bandit incurs exploration costs during early rounds.

---

## 2. Installation

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
```

Optional dependencies:

```bash
pip install torch   # optional NeuralUCB implementation
pip install wfdb    # optional PhysioNetRecordLoader
```

---

## 3. Running the evaluation

### Main 25-seed experiment

The primary paper evaluation uses 25 seeds and 1,200 windows per seed:

```bash
python -m pir_framework.evaluate \
  --n-per-regime 200 \
  --n-seeds 25 \
  --outdir results_25seed
```

### Original five-seed experiment

The following command reproduces the original exploratory five-seed configuration (assuming the same code, dependencies, and seed setup):

```bash
python -m pir_framework.evaluate \
  --n-per-regime 200 \
  --n-seeds 5 \
  --outdir results_5seed
```

To run the Thompson Sampling variant instead of LinUCB:

```bash
python -m pir_framework.evaluate \
  --n-per-regime 200 \
  --n-seeds 25 \
  --bandit thompson \
  --outdir results_thompson
```

A convenience wrapper is also available:

```bash
python scripts/run_experiment.py --n-seeds 25
```

Generated tables and figures are written to the selected output directory. The evaluator also writes per-seed outputs and aggregated metrics. The repository's committed `results/` directory contains the result artifacts currently published with this project; it should not be assumed to be a complete raw output folder for every seed unless all such files are present.

### Tests

```bash
pip install pytest
pytest tests/
```

If `pytest` is unavailable, use:

```bash
python tests/run_tests.py
```

The project README previously documented 94 tests covering coding utilities, the Gilbert–Elliott channel, state-space properties, controllers, metrics/LaTeX export, and a small end-to-end evaluator run. Run the test suite on the exact commit used for a submission and report the current result rather than assuming the historical count still applies.

---

## 4. Results

### 4.1 Main 25-seed comparison

The table below reports the finalized values used in the paper. Per-policy entries are mean ± 95% CI where supplied. PDR confidence intervals for the two adaptive policies are not listed in this table; the paired PDR difference and its confidence interval are reported in Table 2.

| Policy | HR-MAE (bpm) ↓ | Energy (mJ/window) ↓ | PDR (%) ↑ | Mean reward ↑ | Deadline success |
|---|---:|---:|---:|---:|---:|
| Fixed-Rate (R=8) | 6.913 ± 0.303 | 95.81 ± 0.00 | 82.37 ± 2.08 | 0.1715 ± 0.0137 | 100% |
| Always-Local (R=20) | **5.906 ± 0.227** | 119.89 ± 0.00 | 88.71 ± 1.34 | 0.1148 ± 0.0088 | 100% |
| Always-Edge (R=20) | **5.906 ± 0.227** | 119.89 ± 0.00 | 88.71 ± 1.34 | 0.1167 ± 0.0088 | 100% |
| Heuristic-Rule | 6.409 ± 0.341 | 96.66 ± 1.38 | 85.68 | 0.2015 ± 0.0140 | 100% |
| **PIR-LinUCB** | 6.100 ± 0.231 | **92.53 ± 1.42** | **87.56** | **0.2057 ± 0.0152** | 100% |

PIR-LinUCB has lower mean HR-MAE and energy consumption, higher mean reward, and higher PDR than Heuristic-Rule in the finalized 25-seed summary. Both policies achieved 100% deadline success. The statistical comparison below is the primary inferential comparison; other baseline comparisons are descriptive in this study.

### 4.2 Paired comparison: PIR-LinUCB vs Heuristic-Rule

Differences are PIR-LinUCB minus Heuristic-Rule. The paired tests use the 25-seed pairing. Holm correction is applied over the four outcomes HR-MAE, energy, PDR, and reward.

| Outcome | Mean paired difference | 95% CI of difference | Raw p-value | Holm-adjusted p-value | Direction |
|---|---:|---:|---:|---:|---|
| HR-MAE (bpm) ↓ | −0.309 | [−0.494, −0.124] | 0.0021 | 0.0042 | PIR-LinUCB better |
| Energy (mJ/window) ↓ | −4.135 | [−5.228, −3.042] | \(4.9\times10^{-8}\) | \(1.9\times10^{-7}\) | PIR-LinUCB better |
| PDR (percentage points) ↑ | +1.88 | [+0.94, +2.82] | 0.0004 | 0.0012 | PIR-LinUCB better |
| Mean reward ↑ | +0.0041 | [+0.0001, +0.0082] | 0.045 | 0.045 | PIR-LinUCB better |

These p-values describe seed-to-seed variability under the tested simulation configuration. They do not establish generalization to real physiological recordings, patients, or deployed wireless hardware. Statistical significance also does not establish that PIV itself caused the improvements; that would require ablations.

### 4.3 Initial exploratory five-seed experiment

The earlier five-seed experiment is retained for historical context. Its comparison between PIR-LinUCB and Heuristic-Rule was inconclusive, with wide confidence intervals.

| Policy | HR-MAE (bpm) ↓ | Energy (J/window) ↓ | Mean power (mW) ↓ | Mean reward ↑ |
|---|---:|---:|---:|---:|
| Fixed-Rate | 6.96 ± 1.11 | 0.096 ± 0.000 | 12.0 | 0.172 ± 0.054 |
| Always-Local | **5.95 ± 0.58** | 0.120 ± 0.000 | 15.0 | 0.113 ± 0.031 |
| Always-Edge | **5.95 ± 0.58** | 0.120 ± 0.000 | 15.0 | 0.115 ± 0.031 |
| Heuristic-Rule | 6.08 ± 0.57 | 0.096 ± 0.006 | 12.0 | 0.204 ± 0.054 |
| PIR-LinUCB | 6.82 ± 1.62 | **0.089 ± 0.004** | **11.2** | **0.205 ± 0.065** |

The initial five-seed run suggested higher HR-MAE for PIR-LinUCB than for Heuristic-Rule. The final 25-seed result reverses that ordering. The paper uses the 25-seed run as the primary comparison and treats the earlier result as an exploratory estimate with substantial uncertainty.

### 4.4 Interpretation and limits

- **Energy:** PIR-LinUCB uses about 4.3% less energy per window than Heuristic-Rule in the finalized means.
- **Estimation error:** PIR-LinUCB lowers mean HR-MAE by about 0.31 bpm relative to Heuristic-Rule in the finalized means.
- **Delivery:** PIR-LinUCB raises mean PDR by 1.88 percentage points relative to Heuristic-Rule.
- **Reward:** The mean reward advantage is smaller than the other reported effects and is close to the 0.05 threshold even after the specified correction.
- **Placement:** Always-Local and Always-Edge have the same redundancy, energy, HR-MAE, and PDR in this model; their main reported difference is latency. The deadline is met by all policies, so these experiments do not establish a strong advantage from adaptive placement.
- **PIV attribution:** The heuristic does not use PIV when selecting actions, but the experiment has no PIV-free bandit ablation or PIV-aware heuristic. Therefore, the measured difference cannot be attributed specifically to PIV.

---

## 5. Bugs found and fixed during development

The following issues were found during development and addressed in the code. They are documented to make the implementation history and potential pitfalls transparent.

1. **PPG peak-detection double-counting.** A height-based `find_peaks` threshold sometimes counted a PPG dicrotic notch as a second beat. The implementation was changed to use a prominence-based threshold and a tighter minimum-distance constraint.

2. **Missing-modality heart-rate leakage.** A fully missing modality could produce a finite but incorrect heart-rate estimate from the noise floor, which could leak into the fused result. The fallback is gated using the modality's `present_*` flag rather than only checking whether peak detection returned a number.

3. **Deadline-infeasible latency parameters.** An earlier payload choice caused every arm to exceed the deadline regardless of redundancy. The parameters were revised so that the simulation could meaningfully compare policies under the stated deadline.

4. **A channel model with little exploitable adaptivity.** An earlier Gilbert–Elliott configuration made the bad-state erasure probability so catastrophic that additional redundancy had little effect. The oracle/best-fixed/causal-policy gap was examined before trusting bandit results, and the channel configuration was adjusted to provide a more informative test of context-dependent allocation. Results are specific to the selected configuration and do not by themselves establish realism of the channel model.

5. **Context/action information leakage.** An earlier version of `StateSpaceEngine.step()` built the context using the same channel realization later used to score the action. This allowed the context to see information that would not yet be available at decision time. The implementation separated `observe_context()` (previously observed telemetry) from `act()` (the new channel realization). The relevant regression test is `test_state_space.py::test_observe_context_uses_previous_wireless_not_current`.

---

## 6. Known limitations

- **Synthetic physiological data.** The primary evaluation uses the controlled synthetic ECG/PPG generator, not real PhysioNet recordings. `PhysioNetRecordLoader` is implemented, but the reported study did not validate it against downloaded real records. Recorded-signal evaluation is necessary before making deployment or clinical claims.
- **Uncalibrated channel, energy, and latency models.** The constants are model assumptions and were not fitted to measurements from a specific radio, MCU, or edge device.
- **Analytic decoding approximation.** The main results use a closed-form decoding probability for an ideal erasure code, not packet-level RLNC encoding and decoding in the full experiment.
- **Hand-designed PIV.** PIV is derived from signal quality and cross-modal agreement. It is not clinically validated, and an ablation is needed to determine whether PIV itself contributes to the policy's performance.
- **Limited baseline set.** The heuristic is one hand-written rule. A tuned rule or a PIV-aware heuristic may reduce the measured gap. There is no oracle bound in the primary comparison.
- **Placement effect is limited under this configuration.** All policies meet the deadline and energy is largely determined by redundancy. Claims about learned local-versus-edge placement would be premature.
- **Synthetic waveform morphology.** The generator uses a sum-of-Gaussians approximation to ECG P–QRS–T and PPG morphology. It is suitable for controlled peak detection, SQI, and degradation experiments, but is not a waveform-level clinical simulator.
- **Limited modalities and environments.** Only ECG and PPG, one synthetic generator, and one channel configuration were tested. Transfer to tactile, inertial, RF, recorded cardiovascular data, or physical devices remains future work.

---

## 7. Repository layout

```text
pir_framework/                       Python package
configs/default.yaml                 Documented parameter reference
scripts/run_experiment.py            CLI convenience wrapper
tests/                               Unit and integration tests
results/
├── per_seed_metrics.csv              Per-seed metrics for the five policies
├── summary_25seed.csv                25-seed means and confidence intervals
├── PIR_paired_comparisons_25seed.csv Paired comparisons and corrected p-values
├── tables/                           Result tables from the original run
├── figures/                          Result figures from the original run
└── seed_0/ ... seed_4/                Original five-seed output sub-runs
requirements.txt
pyproject.toml
README.md
```

The committed `seed_0` through `seed_4` folders are the original five-seed outputs; the consolidated CSV files document the finalized 25-seed results. Keep the source seed-level data and analysis scripts aligned with the published summary when updating or regenerating these files.

---

## Citation

If you use this repository in your research, please cite:

```text
anonymous for double blind
```
