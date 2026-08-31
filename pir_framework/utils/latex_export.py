"""Publication-ready LaTeX table export (booktabs style)."""
from __future__ import annotations

import pandas as pd


DEFAULT_COLUMN_FORMAT = {
    "hr_mae_bpm": ("HR MAE (bpm)", "{:.2f}"),
    "event_f1": ("Event F1", "{:.3f}"),
    "pdr": ("PDR", "{:.3f}"),
    "mean_energy_j": ("Energy/window (J)", "{:.4f}"),
    "mean_power_mw": ("Power (mW)", "{:.2f}"),
    "mean_latency_ms": ("Latency (ms)", "{:.1f}"),
    "deadline_success_rate": ("Deadline success", "{:.3f}"),
    "power_cap_success_rate": ("Power-cap success", "{:.3f}"),
    "mean_reward": ("Mean reward", "{:.3f}"),
}


def dataframe_to_latex_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    caption: str = "Comparative evaluation of PIR-Framework policies.",
    label: str = "tab:pir_results",
    column_format_map: dict[str, tuple[str, str]] | None = None,
    bold_best: dict[str, str] | None = None,
    ci_df: pd.DataFrame | None = None,
) -> str:
    """
    Render a metrics DataFrame (one row per policy) as a booktabs-style
    LaTeX table.

    Parameters
    ----------
    columns : subset/order of metric columns to include (defaults to all
        keys in `column_format_map` that are present in `df`).
    bold_best : optional {column: "min"|"max"} map indicating, for each
        column, whether the best (lowest/highest) value across rows should
        be rendered in bold -- e.g. {"hr_mae_bpm": "min", "pdr": "max"}.
    ci_df : optional DataFrame, same shape/columns as `df` (indexed to
        align by row order), holding the half-width of a 95% confidence
        interval for each cell (e.g. from a multi-seed sweep). When
        supplied, cells render as "mean $\\pm$ ci" instead of a bare value.
    """
    fmt_map = column_format_map or DEFAULT_COLUMN_FORMAT
    if columns is None:
        columns = [c for c in fmt_map if c in df.columns]
    bold_best = bold_best or {}

    best_values: dict[str, float] = {}
    for col in columns:
        if col in bold_best:
            best_values[col] = df[col].min() if bold_best[col] == "min" else df[col].max()

    header_labels = [fmt_map.get(c, (c, "{}"))[0] for c in columns]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    col_spec = "l" + "r" * len(columns)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("Policy & " + " & ".join(header_labels) + " \\\\")
    lines.append("\\midrule")

    for row_i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for col in columns:
            _, fmt = fmt_map.get(col, (col, "{}"))
            value = row[col]
            cell = fmt.format(value)
            if ci_df is not None and col in ci_df.columns:
                ci_val = ci_df.iloc[row_i][col]
                cell = f"{cell} $\\pm$ {fmt.format(ci_val)}"
            if col in best_values and abs(value - best_values[col]) < 1e-9:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        policy_name = str(row["policy"]).replace("_", "\\_")
        lines.append(f"{policy_name} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)
