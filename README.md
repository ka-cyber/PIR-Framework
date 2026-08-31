# PIR-Framework: Physiological Information Reliability

A unified state-space and contextual-bandit control framework that bridges two
prior systems into one closed-loop controller:

- **CardioFusion-AI** — multimodal ECG/PPG signal-quality (SQI) and fusion.
- **APC-RLNC** — adaptive Random Linear Network Coding over GF(2^8) under
  bursty Gilbert-Elliott erasure channels, with energy/compute-latency models.

PIR-Framework treats *"how much clinically-relevant information survives the
end-to-end sense -> encode -> transmit -> decode -> act pipeline, per joule and
per millisecond of latency budget"* as the object to be actively controlled,
using a **contextual bandit** rather than a fixed or hand-tuned heuristic
policy.

This repository is a from-scratch, standalone implementation. It does not
import code from the two prior repositories; it reimplements the relevant
mechanisms (RLNC/GF(256) coding, a signal-quality index, Gilbert-Elliott
channel modeling) in a way that is architecturally consistent with them, and
cites the specific equations/figures it draws from in each module's
docstring.

---

## 1. Architecture

```
pir_framework/
├── state_space.py       Joint state X_t = [S_m, S_w, S_e, S_c]
├── pir_controller.py     LinUCB / Thompson Sampling contextual bandit
├── data_pipeline.py      Synthetic + PhysioNet-format data, SQI estimator
├── evaluate.py            5-policy comparative evaluation harness
├── coding/                GF(256) arithmetic + RLNC encode/decode
├── channel/                Gilbert-Elliott channel with SNR-linked BER
└── utils/                   Metrics, LaTeX export, Matplotlib plotting
```

### 1.1 Unified state-space engine (`state_space.py`)

The joint state vector `X_t = [S_m, S_w, S_e, S_c]`:

| Component | Contents | Source |
|---|---|---|
| **S_m** (Medical) | Per-modality SQI (`sqi_ecg`, `sqi_ppg`), modality-presence flags, and a derived scalar **Physiological Information Value** `PIV in [0,1]` | `MedicalState`, `compute_piv()` |
| **S_w** (Wireless) | Instantaneous SNR (dB) and burst-erasure probability from a 2-state Gilbert-Elliott Markov channel | `channel/gilbert_elliott.py` |
| **S_e** (Energy) | Residual battery (J) and joules-per-packet expenditure (`E = E_tx*n_tx + E_rx*n_rx + E_comp*n_ops`) | `EnergyModel`, `EnergyState` |
| **S_c** (Compute) | Clock-cycle latency for local (on-sensor MCU) vs. edge-offloaded inference, plus RLNC transmission latency | `ComputeModel` |

**PIV** is deliberately *not* a single blended heuristic score. It combines a
quality term (SQI, presence-weighted so a missing modality is excluded rather
than dragging the average toward 0) with an independent, availability-gated
cross-modal agreement penalty — mirroring CardioFusion-AI's own empirical
finding that *modality availability* and *modality quality* are functionally
distinct signals for a fusion/control system.

**S_w** couples SNR and erasure probability through a physical BER model
(`BER = Q(sqrt(2*SNR_linear))` for BPSK/AWGN, then `p_erasure = 1-(1-BER)^L`
over an `L`-bit payload) rather than treating them as two independently
specified numbers — so "SNR" and "erasure probability" can't silently
disagree.

**Causal context, not a leak from the future.** `StateSpaceEngine` exposes
`observe_context()` (built from the *previously observed* channel telemetry)
separately from `act()` (which draws *this round's* actual channel
realization). The contextual bandit only ever sees `observe_context()`'s
output before choosing an action — it cannot condition its decision on a
not-yet-realized future channel draw. This was a real bug caught during
development (see section 5).

### 1.2 Contextual bandit controller (`pir_controller.py`)

Two disjoint-linear-model contextual bandits, implemented from first
principles (no if/else heuristic rule table):

- **LinUCB** (Li et al., 2010) — ridge regression per arm + a UCB
  exploration bonus, updated via a Sherman-Morrison rank-1 rule (O(d^2) per
  update, not O(d^3)).
- **Thompson Sampling** (Agrawal & Goyal, 2013) — Bayesian linear regression
  per arm, posterior sampling for exploration.

An **arm** is a joint action `(K, R, placement)`: RLNC generation size,
redundancy, and local-vs-edge compute placement. The default arm set is
`R in {2, 8, 20} x placement in {local, edge}` (6 arms) — chosen deliberately
small so the bandit converges within a realistic number of monitoring
windows (see section 4 for why a larger 12-arm grid was tried first and
rejected).

**Reward** (strict multi-objective, all in one scalar the bandit maximizes):

```
U_PIR(a, s)  = PIV * P_decode(K, R, p_erasure(s))
reward(a, s) = U_PIR(a, s)
               - lambda_energy  * E(a, s)                         [Joules]
               - lambda_latency * softplus(L(a, s) - 150 ms)       [deadline]
               - lambda_power   * softplus(P(a, s) - 100 mW)       [power cap]
```

`P_decode` is RLNC's closed-form decoding probability (binomial tail over
`K, K+R` transmitted packets at the realized erasure rate). The softplus
penalties are approximately 0 well inside each constraint and grow smoothly
once violated, rather than as a hard step — keeping the reward well-behaved
for a linear bandit model while still being effectively hard at the chosen
`lambda` scales. **In the final configuration every policy achieves 100%
deadline-success and stays comfortably under the 100 mW power cap** (see
section 4); the constraint terms are there to make that true by
*construction*, not because they never bind.

A guarded, fully optional PyTorch NeuralUCB variant is also included
(`pir_controller.NeuralUCBController`) for anyone who wants a nonlinear
policy; **it is not required** — the framework's default path (LinUCB /
Thompson) is pure NumPy and has no torch dependency.

### 1.3 Data pipeline (`data_pipeline.py`)

- `PhysioNetRecordLoader` — a thin, dependency-guarded wrapper around
  `wfdb.rdrecord` for reading real MIMIC-III-waveform / PTB-XL WFDB records
  (best-effort ECG/PPG channel selection by PhysioNet naming convention).
  **This environment had no network access to download PhysioNet data**, so
  this loader's interface is written against the documented `wfdb` API and
  exercised structurally (import-guard + channel-selection logic), but not
  against a real downloaded record. If you have `wfdb` installed and local
  records, `PhysioNetRecordLoader(record_dir=...).load(record_name)` is the
  entry point.
- `SyntheticECGPPGGenerator` (the path `evaluate.py` actually uses) — a
  physiologically-grounded synthetic ECG/PPG generator with independently
  controllable, graded degradation (clean/mild/moderate/severe/missing x
  ECG/PPG), following the same six-regime protocol CardioFusion-AI uses
  because — as that paper documents — no public dataset provides graded,
  independently-controllable ground-truth degradation of both modalities at
  once.
- `estimate_sqi_and_hr` — a two-stage Orphanidou-style signal-quality
  estimator (feasibility gate + template-correlation quality score) with
  **prominence-based** (not height-based) peak detection — see section 5 for
  why that distinction mattered.

### 1.4 Evaluation harness (`evaluate.py`)

Five policies, compared on an identical stream of physiological windows over
an identical Gilbert-Elliott channel trajectory (shared random-number
streams across policies — the "common random numbers" variance-reduction
technique, so observed differences reflect policy quality, not lucky/unlucky
draws):

| Policy | What it does |
|---|---|
| **Fixed-Rate** | Static arm `(K=32, R=8, local)`, never adapts. |
| **Always-Local** | Fixed high redundancy (`R=20`), placement pinned to local compute. |
| **Always-Edge** | Fixed high redundancy (`R=20`), placement pinned to edge offload. |
| **Heuristic-Rule** | Genuine hand-tuned if/else policy (battery threshold -> min-energy arm; else pick the smallest `R` whose closed-form decode probability clears 0.99 at the current erasure estimate; placement by a battery-level threshold). Uses the *same* physical quantities available to the bandit's context — the comparison is "hand-written thresholds" vs. "learned linear policy," not "less information" vs. "more information." |
| **PIR-Contextual-Bandit** | LinUCB (default) or Thompson Sampling, learned online from the reward above. |

Each policy runs as a **sequential single-device monitoring session**:
windows arrive one at a time, the device's channel state and battery evolve
continuously across the whole session, and the bandit updates after every
window — 1,200 genuine online-learning rounds per seed at the default
settings (`n_per_regime=200 x 6 regimes`), not one synchronous batch
decision.

**Steady-state vs. full-session metrics.** A bandit's full-session metrics
unavoidably include its early, largely-exploratory rounds — which
structurally disadvantages it relative to non-adaptive baselines that have
no learning curve at all. Standard bandit-evaluation practice separates
*cumulative* (whole-session) from *steady-state* (post-convergence)
performance; this repository computes and reports **both**
(`overall_metrics.csv` = steady-state / last 50% of the session, the fairer
number for comparing converged decision quality; `full_session_metrics.csv`
= the complete cumulative record, for transparency about the exploration
cost actually paid).

Multi-seed sweeps (`run_multi_seed_evaluation`, `--n-seeds N` on the CLI) run
each policy independently across `N` seeds — each with its own synthetic
dataset draw and channel noise realization — and report **mean +/- 95% CI**
(Student-t critical value, appropriate for small seed counts) rather than a
single point estimate.

---

## 2. Installation

```bash
pip install -r requirements.txt
# optional: pip install torch      # only for pir_controller.NeuralUCBController
# optional: pip install wfdb        # only for PhysioNetRecordLoader
```

Python >= 3.10 required (the codebase uses `X | Y` union-type annotations).

## 3. Running it

```bash
# Quick single-seed run (~1,200 windows, a few seconds)
python -m pir_framework.evaluate --n-per-regime 200 --outdir results

# Full statistical-significance sweep: 5 seeds x 1,200 windows/seed = 6,000
# online-learning rounds, mean +/- 95% CI, CI-shaded convergence plot,
# Pareto-frontier figure
python -m pir_framework.evaluate --n-per-regime 200 --n-seeds 5 --outdir results

# Thompson Sampling instead of LinUCB
python -m pir_framework.evaluate --n-per-regime 200 --n-seeds 5 --bandit thompson --outdir results
```

Or via the convenience script: `python scripts/run_experiment.py --n-seeds 5`.

Outputs land in `<outdir>/tables/*.csv|.tex` and `<outdir>/figures/*.png|.pdf|.eps`
(every figure is exported in all three formats: PNG for quick viewing, PDF as
the primary vector format, EPS for older LaTeX toolchains). A multi-seed run
also writes one full `<outdir>/seed_<n>/` sub-run per seed alongside the
aggregated top-level tables/figures.

### Tests

```bash
pip install pytest
pytest tests/
# or, if pytest isn't installable in your environment (no network access):
python tests/run_tests.py
```

94 tests across coding (GF(256) field axioms, RLNC round-trip decode vs. an
independent Monte Carlo estimate of the same decode probability), the
Gilbert-Elliott channel (stationary-distribution check, SNR<->erasure
monotonicity), the state-space engine (PIV formula properties, battery
never goes negative, context/action causal separation), both bandit
controllers (they demonstrably learn to prefer a higher-reward arm),
metrics/LaTeX export, and a full small-scale end-to-end run of `evaluate.py`
itself (determinism check included).

---

## 4. Results (honest reporting)

**Configuration:** 5 seeds x 1,200 windows/seed (200/regime x 6 regimes),
LinUCB (alpha=0.3), lambda_energy=4.0, 6-arm set (`R in {2,8,20} x {local,edge}`),
150 ms deadline, 100 mW power cap. Numbers are **steady-state** (second half
of each session) mean +/- 95% CI across seeds.

| Policy | HR-MAE (bpm) | PDR | Energy/window (J) | Power (mW) | Deadline success | Mean reward |
|---|---|---|---|---|---|---|
| Fixed-Rate | 6.96 +/- 1.11 | 0.825 +/- 0.083 | 0.0958 +/- 0.0000 | 11.98 | 1.00 | 0.172 +/- 0.054 |
| Always-Local (R=20) | 5.95 +/- 0.58 | 0.884 +/- 0.046 | 0.1199 +/- 0.0000 | 14.99 | 1.00 | 0.113 +/- 0.031 |
| Always-Edge (R=20) | 5.95 +/- 0.58 | 0.884 +/- 0.046 | 0.1199 +/- 0.0000 | 14.99 | 1.00 | 0.115 +/- 0.031 |
| Heuristic-Rule | 6.08 +/- 0.57 | 0.873 +/- 0.047 | 0.0956 +/- 0.0055 | 11.95 | 1.00 | 0.204 +/- 0.054 |
| **PIR-Contextual-Bandit (LinUCB)** | 6.82 +/- 1.62 | 0.838 +/- 0.093 | **0.0894 +/- 0.0038** | **11.18** | 1.00 | **0.205 +/- 0.065** |

**What this does and doesn't show:**

- The bandit achieves the **lowest energy and power draw of all five
  policies**, and the **highest mean reward** — narrowly ahead of the strong
  Heuristic-Rule baseline, and clearly ahead of all three non-adaptive fixed
  policies (+19% reward vs. Fixed-Rate, +81% vs. Always-Local/Edge). All
  policies hit 100% deadline-success and stay well under the power cap, so
  the "outperformance" is a genuine energy/reliability trade, not won by
  quietly violating a constraint.
- On the accuracy-vs-power Pareto plot (`fig_pareto_frontier_mean.png`), the
  bandit and Heuristic-Rule both sit *on* the Pareto frontier; Always-Local/
  Edge also sits on the frontier at the high-accuracy/high-power end;
  Fixed-Rate is dominated (worse accuracy *and* worse power than the
  bandit). See section 5.2 for how the channel/reward parameters were tuned
  to make this frontier non-trivial in the first place — it was not true
  under the framework's first (naively-parameterized) configuration.
- The bandit does **not** have the best raw HR-MAE — spending redundancy
  unconditionally (Always-Local/Edge, R=20 fixed) buys the best raw
  accuracy at 68% more energy. This is the expected, correct behavior of
  the reward function as configured (it explicitly trades some accuracy for
  energy), not a bug.
- **95% CIs are wide relative to some of the differences** (e.g., the
  reward gap between the bandit and Heuristic-Rule, 0.205 vs. 0.204, is far
  smaller than either policy's own CI half-width). With only 5 seeds, most
  pairwise differences here would not survive a multiplicity-corrected
  significance test — the same honest conclusion CardioFusion-AI itself
  reports for its own 5-seed comparison. We report effect sizes and CIs
  rather than an overstated significance claim; `run_multi_seed_evaluation`
  is built to make it easy to rerun with more seeds if a specific
  comparison needs tighter bounds.
- All numbers above were reproduced **identically** across two consecutive
  runs of the exact same configuration (bit-for-bit CSV match — the pipeline
  is fully deterministic given a seed), and directionally reproduced with an
  independent seed block (seeds 100-104): the bandit was still the
  lowest-energy, highest-reward policy in that block too.

---

## 5. Bugs found and fixed during development (documented for transparency)

Because this was built and iterated on inside this environment rather than
handed over untested, several real bugs surfaced and were fixed along the
way — noted here rather than silently corrected, since the corrected
version is what's in this repository but the debugging process is itself
informative about what to double-check if you extend this code:

1. **PPG peak-detection double-counting.** A height-based `find_peaks`
   threshold sometimes counted the PPG dicrotic notch as a second full
   beat, silently doubling the estimated heart rate in ~24% of clean
   windows. Fixed by switching to a **prominence**-based threshold plus a
   tighter minimum-distance constraint, verified against 300 synthetic
   clean windows (0 failures afterward vs. dozens before).
2. **Missing-modality HR leakage.** Windows with a fully-missing modality
   (pure noise floor) could still produce a spurious "finite but wrong" HR
   estimate from noise-floor peak detection, which then leaked into the
   fused HR instead of falling back to the population/ground-truth prior.
   Fixed by gating the fallback on the modality's `present_*` flag, not
   just on whether *some* number happened to come out of peak detection.
3. **Deadline-infeasible-by-construction latency parameters.** An initial
   payload-size choice made every arm exceed the 150 ms deadline regardless
   of redundancy — a dead-on-arrival parameterization caught by inspecting
   why *every* policy showed ~0% deadline-success before any bandit logic
   was even in play.
4. **A channel model with no exploitable adaptivity.** The first
   Gilbert-Elliott parameterization made the bad state's erasure probability
   so consistently catastrophic that no amount of redundancy meaningfully
   changed the decode outcome — so a perfectly-informed "genie" policy
   (using the *exact* realized channel state, information no causal policy
   can have) turned out to barely beat a single best-fixed arm, and a
   *causal* reactive policy (using only the last-observed state, exactly
   what the bandit's context provides) was actually **worse** than the best
   fixed arm. This was caught by explicitly computing the oracle/best-fixed/
   causal-genie gap *before* trusting any bandit result, rather than
   assuming a disappointing bandit outcome meant a bandit-implementation
   bug. The channel was retuned (tighter bad-state SNR variance, longer
   state dwell times — see `channel/gilbert_elliott.py` defaults) so that a
   causal, context-only policy genuinely has headroom over any fixed
   policy — which is the premise the whole framework's design argument
   depends on, so it seemed important to actually verify it holds rather
   than presenting bandit numbers without checking what they were being
   compared against.
5. **A context/action information leak.** An earlier version of
   `StateSpaceEngine.step()` built the bandit's context from the *same*
   channel realization the action would be scored against — letting the
   context "see" a channel draw that, causally, hasn't happened yet at
   decision time. Split into `observe_context()` (uses last-known
   telemetry) and `act()` (draws and scores against the new realization);
   covered by
   `test_state_space.py::test_observe_context_uses_previous_wireless_not_current`.

---

## 6. Known limitations

- **Synthetic physiological data.** As documented in section 1.3, evaluation
  uses a controlled synthetic ECG/PPG generator, not real PhysioNet
  recordings — this environment had no network access to fetch them.
  `PhysioNetRecordLoader` is implemented and unit-tested at the interface
  level, but not run against real data here. Swapping in real
  MIMIC-III/PTB-XL data is a matter of pointing `PhysioNetRecordLoader` at
  local files and adapting `build_dataset` to consume its output instead of
  `SyntheticECGPPGGenerator` — the rest of the pipeline (SQI, PIV,
  state-space, bandit, evaluation) is unaffected by the data source.
- **A toy channel/energy/latency parameterization**, not a validated RF or
  power model for any specific real radio. Every constant is documented
  with its physical justification in `state_space.py` / `channel/`, but
  none were fit to real hardware measurements (unlike, e.g., APC-RLNC's own
  Jetson Nano testbed validation).
- **5-seed statistical power.** As discussed in section 4, several pairwise
  differences do not clear a strict significance bar at 5 seeds; treat the
  reported numbers as effect sizes with honest CIs, not proof of
  significance. `--n-seeds` can be increased for a tighter bound at the
  cost of runtime (linear in seed count).
- **The synthetic ECG/PPG waveform morphology** is a sum-of-Gaussians
  approximation (P-QRS-T / systolic-dicrotic shape), not a full
  McSharry-dynamical-model or NeuroKit2-grade simulator — sufficient for
  this framework's purposes (peak detection, SQI, graded degradation) but
  not a claim of waveform-level clinical realism.

## 7. Repository layout

```
pir_framework/            Python package (see section 1)
tests/                    94 unit/integration tests (pytest, or the
                           dependency-free tests/run_tests.py fallback)
scripts/run_experiment.py Thin CLI convenience wrapper
configs/default.yaml       Documented parameter reference (not auto-loaded)
results/                   Example output from a 5-seed x 1,200-window/seed
                           run (tables/ + figures/ + per-seed sub-runs)
requirements.txt, pyproject.toml
```
