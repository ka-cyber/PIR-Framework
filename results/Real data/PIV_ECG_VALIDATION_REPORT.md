# Real-data evidence for a multi-electrode ECG adaptation of PIV

## Experiment

We evaluated a multi-electrode ECG adaptation of the Physiological Information Value (PIV) proxy using the PhysioNet Abdominal and Direct Fetal ECG Database. Five recordings (`r01`, `r04`, `r07`, `r08`, `r10`) were processed. Four abdominal ECG leads supplied the signal-quality and inter-lead agreement features. Cardiologist-verified QRS annotations from the direct fetal ECG were used only to derive reference fetal heart rate and absolute estimation error; they were not used to compute PIV.

Each recording was filtered offline at 20–100 Hz and divided into non-overlapping 8-second windows. For each abdominal lead, fetal heart rate was estimated from the strongest overlap-corrected normalized autocorrelation peak over the 105–180 bpm range. The peak served as a periodicity-based SQI. The combined SQI was the arithmetic mean over available leads; lead heart rates were combined using SQI weights. Inter-lead disagreement was the mean absolute deviation from that consensus.

The ECG-adapted proxy was:

`PIV_ECG = clip(0.75 * SQI_combined - 0.25 * disagreement_bpm / 140, 0, 1)`.

This is a multi-electrode ECG adaptation.

## Main results

- Five recordings and 185 valid 8-second windows were analysed.
- The equal-record mean within-record Spearman correlation between PIV and absolute fetal heart-rate error was **−0.504**.
- The equal-record mean within-record correlation for the SQI-only comparison score was **−0.401**.
- The descriptive difference in mean correlations (PIV minus SQI-only) was **−0.103**, favouring a stronger inverse association for PIV. This is descriptive evidence; the five-record sample is too small to establish a reliable population-level incremental effect.
- PIV–error correlations were negative in all five recordings.
- Across four filter/window sensitivity configurations, the equal-record mean within-record correlation remained negative (range **−0.341 to −0.548**).
- The median absolute HR error in the lowest and highest PIV quintiles was **13.80 bpm** and **0.13 bpm**, respectively. Quintiles pool repeated windows and are descriptive, not independent-subject groups.

## Per-record results

| Recording | Windows | Mean PIV | Mean absolute HR error (bpm) | PIV vs error Spearman rho | SQI-only vs error Spearman rho |
|---|---:|---:|---:|---:|---:|
| r01 | 37 | 0.237 | 2.14 | −0.708 | −0.703 |
| r04 | 37 | 0.094 | 18.07 | −0.288 | −0.197 |
| r07 | 37 | 0.133 | 11.05 | −0.454 | −0.165 |
| r08 | 37 | 0.213 | 2.64 | −0.854 | −0.846 |
| r10 | 37 | 0.118 | 8.22 | −0.217 | −0.094 |

## Interpretation

These results provide preliminary real-physiology evidence that the multi-lead ECG adaptation of PIV is inversely associated with fetal heart-rate estimation error in this dataset. They are consistent with the hypothesis that combining per-lead signal quality and inter-lead agreement can provide a useful window-level reliability proxy.


## Reproducibility and data

Run from the repository root:

```bash
pip install -e '.[real-edf]'
python -m pir_framework.real_fetal_experiment \
  --data-dir /path/to/abdominal-and-direct-fetal-ecg-database-1.0.0 \
  --outdir results/real_fetal_ecg
```

