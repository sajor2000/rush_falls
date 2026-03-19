"""
07_threshold_analysis.py — Threshold Method Comparison (eFigure 3, eFigure 5)

Computes seven threshold selection methods for each model and overlays them
on ROC curves. Shows sensitivity/specificity trade-offs at each operating point.
Also generates score distribution histograms with flag rate annotations.

Inputs:  data/processed/analytic.parquet
Outputs: outputs/figures/efigure3_threshold_overlay.pdf/.png
         outputs/figures/efigure5_score_distributions.pdf/.png
         outputs/tables/flag_rate_summary.csv
"""
import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
r"""
# eFigure 3. Threshold Method Comparison

Seven threshold selection methods overlaid on ROC curves for Epic PMFRS
(left panel) and Morse Fall Scale (right panel). Each dot marks the
operating point — sensitivity/1-specificity — achieved by that method.

Methods compared: Youden index, closest-to-(0,1), fixed sensitivity at
60% and 80%, value-optimizing NMB, and standard MFS cutoffs (25 and 45).
"""
    )
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_curve
    return Path, mpatches, np, pl, plt, roc_curve


@app.cell
def _():
    from utils.constants import (
        EPIC_2TIER_HIGH,
        EPIC_3TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        MFS_HIGH,
        MFS_MODERATE,
    )
    from utils.metrics import (
        classification_metrics_at_threshold,
        closest_topleft_threshold,
        fixed_sensitivity_threshold,
        logistic_recalibration,
        value_optimizing_threshold,
        youden_threshold,
    )
    from utils.plotting import COLORS, JAMA_STYLE, save_figure
    return (
        COLORS,
        EPIC_2TIER_HIGH,
        EPIC_3TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        JAMA_STYLE,
        MFS_HIGH,
        MFS_MODERATE,
        classification_metrics_at_threshold,
        closest_topleft_threshold,
        fixed_sensitivity_threshold,
        logistic_recalibration,
        save_figure,
        value_optimizing_threshold,
        youden_threshold,
    )


# ── Load data ─────────────────────────────────────────────────────────
@app.cell
def _(Path, pl):
    _path = Path("data/processed/analytic.parquet")
    df = pl.read_parquet(_path)
    return (df,)


@app.cell
def _(df):
    y_true = df["fall_flag"].to_numpy()
    epic_score = df["epic_score_admission"].to_numpy()
    morse_score = df["morse_score_admission"].to_numpy()
    return epic_score, morse_score, y_true


# ── Recalibrate scores → probabilities (needed for value-optimizing) ─
@app.cell
def _(epic_score, logistic_recalibration, morse_score, y_true):
    epic_prob, _ = logistic_recalibration(epic_score, y_true)
    morse_prob, _ = logistic_recalibration(morse_score, y_true)
    return epic_prob, morse_prob


# ── Compute all threshold methods ────────────────────────────────────
@app.cell
def _(
    EPIC_2TIER_HIGH,
    EPIC_3TIER_HIGH,
    EPIC_3TIER_MEDIUM,
    MFS_HIGH,
    MFS_MODERATE,
    classification_metrics_at_threshold,
    closest_topleft_threshold,
    epic_prob,
    epic_score,
    fixed_sensitivity_threshold,
    morse_prob,
    morse_score,
    value_optimizing_threshold,
    y_true,
    youden_threshold,
):
    def _compute_thresholds(score, prob):
        """Return list of (method_label, threshold, metrics_dict, space_label) for one model."""
        _results = []

        # 1. Youden index (applied to ordinal score)
        _t = youden_threshold(y_true, score)
        _results.append(("Youden", _t, classification_metrics_at_threshold(y_true, score, _t), "Ordinal"))

        # 2. Closest to (0,1)
        _t = closest_topleft_threshold(y_true, score)
        _results.append(("Closest (0,1)", _t, classification_metrics_at_threshold(y_true, score, _t), "Ordinal"))

        # 3. Fixed sensitivity 60%
        _t = fixed_sensitivity_threshold(y_true, score, target_sens=0.60)
        _results.append(("Sens ≥ 60%", _t, classification_metrics_at_threshold(y_true, score, _t), "Ordinal"))

        # 4. Fixed sensitivity 80%
        _t = fixed_sensitivity_threshold(y_true, score, target_sens=0.80)
        _results.append(("Sens ≥ 80%", _t, classification_metrics_at_threshold(y_true, score, _t), "Ordinal"))

        # 5. Value-optimizing NMB (requires calibrated probability)
        _t_nmb, _, _ = value_optimizing_threshold(y_true, prob)
        _results.append(
            (
                "Value NMB",
                _t_nmb,
                classification_metrics_at_threshold(y_true, prob, _t_nmb),
                "Probability",
            )
        )

        return _results

    _epic_results = _compute_thresholds(epic_score, epic_prob)
    _morse_results = _compute_thresholds(morse_score, morse_prob)

    # Standard MFS cutoffs (applied directly to Morse ordinal scores)
    _t25 = float(MFS_MODERATE)
    _t45 = float(MFS_HIGH)
    _morse_cutoff25 = ("MFS ≥ 25", _t25, classification_metrics_at_threshold(y_true, morse_score, _t25), "Ordinal")
    _morse_cutoff45 = ("MFS ≥ 45", _t45, classification_metrics_at_threshold(y_true, morse_score, _t45), "Ordinal")
    _morse_results.extend([_morse_cutoff25, _morse_cutoff45])

    # Epic standard cutoffs (applied directly to Epic ordinal scores)
    _t35 = float(EPIC_3TIER_MEDIUM)
    _t70 = float(EPIC_3TIER_HIGH)
    _t50 = float(EPIC_2TIER_HIGH)
    _epic_cutoff35 = ("Epic ≥ 35 (3-tier med)", _t35, classification_metrics_at_threshold(y_true, epic_score, _t35), "Ordinal")
    _epic_cutoff70 = ("Epic ≥ 70 (3-tier high)", _t70, classification_metrics_at_threshold(y_true, epic_score, _t70), "Ordinal")
    _epic_cutoff50 = ("Epic ≥ 50 (2-tier high)", _t50, classification_metrics_at_threshold(y_true, epic_score, _t50), "Ordinal")
    _epic_results.extend([_epic_cutoff35, _epic_cutoff70, _epic_cutoff50])

    epic_thresholds = _epic_results
    morse_thresholds = _morse_results
    return epic_thresholds, morse_thresholds


# ── ROC curve arrays ──────────────────────────────────────────────────
@app.cell
def _(epic_score, morse_score, roc_curve, y_true):
    fpr_epic, tpr_epic, _ = roc_curve(y_true, epic_score)
    fpr_morse, tpr_morse, _ = roc_curve(y_true, morse_score)
    return fpr_epic, fpr_morse, tpr_epic, tpr_morse


# ── eFigure 3: Two-panel threshold overlay ────────────────────────────
@app.cell
def _(
    COLORS,
    JAMA_STYLE,
    epic_thresholds,
    fpr_epic,
    fpr_morse,
    mpatches,
    morse_thresholds,
    plt,
    save_figure,
    tpr_epic,
    tpr_morse,
):
    # Marker shapes and colors for each threshold method
    _METHOD_STYLES: dict[str, dict] = {
        "Youden":      {"marker": "o", "color": "#1B7837", "zorder": 5},
        "Closest (0,1)": {"marker": "s", "color": "#762A83", "zorder": 5},
        "Sens ≥ 60%":  {"marker": "^", "color": "#E08214", "zorder": 5},
        "Sens ≥ 80%":  {"marker": "v", "color": "#FDB863", "zorder": 5},
        "Value NMB":   {"marker": "D", "color": "#4393C3", "zorder": 5},
        "MFS ≥ 25":    {"marker": "P", "color": "#D6604D", "zorder": 5},
        "MFS ≥ 45":    {"marker": "*", "color": "#8C510A", "zorder": 5, "ms": 10},
        "Epic ≥ 35 (3-tier med)":  {"marker": "h", "color": "#2166AC", "zorder": 5},
        "Epic ≥ 70 (3-tier high)": {"marker": "H", "color": "#053061", "zorder": 5},
        "Epic ≥ 50 (2-tier high)": {"marker": "X", "color": "#4393C3", "zorder": 5, "ms": 8},
    }

    with plt.rc_context(JAMA_STYLE):
        fig_ef3, (ax_e, ax_m) = plt.subplots(1, 2, figsize=(7.0, 4.0))

        for _ax, _model_label, _color, _fpr, _tpr, _thresholds in [
            (ax_e, "Epic PMFRS", COLORS["epic"], fpr_epic, tpr_epic, epic_thresholds),
            (ax_m, "Morse Fall Scale", COLORS["morse"], fpr_morse, tpr_morse, morse_thresholds),
        ]:
            # ROC background
            _ax.plot(
                _fpr,
                _tpr,
                color=_color,
                linewidth=1.25,
                label="ROC curve",
                zorder=2,
            )
            _ax.plot([0, 1], [0, 1], color="#AAAAAA", linewidth=0.5, linestyle="--", zorder=1)

            # Threshold operating points
            for _method, _thresh, _mets, _space in _thresholds:
                _sens = _mets["sensitivity"]
                _spec = _mets["specificity"]
                _fpr_pt = 1 - _spec
                _style = _METHOD_STYLES.get(_method, {"marker": "o", "color": "#888888", "zorder": 5})
                _ms = _style.get("ms", 7)
                _ax.plot(
                    _fpr_pt,
                    _sens,
                    marker=_style["marker"],
                    color=_style["color"],
                    markersize=_ms,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    linestyle="none",
                    zorder=_style["zorder"],
                )

            _ax.set_xlim(-0.02, 1.02)
            _ax.set_ylim(-0.02, 1.02)
            _ax.set_xlabel("1 - Specificity", fontsize=9)
            _ax.set_ylabel("Sensitivity", fontsize=9)
            _ax.set_title(_model_label, fontsize=10, fontweight="bold")
            _ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            _ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            _ax.tick_params(labelsize=8)

        # Shared legend below the figure
        _handles = [
            mpatches.Patch(color=s["color"], label=method)
            for method, s in _METHOD_STYLES.items()
        ]
        fig_ef3.legend(
            handles=_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=4,
            fontsize=8,
            frameon=False,
            columnspacing=0.8,
            handlelength=1.0,
        )

        fig_ef3.subplots_adjust(wspace=0.35, bottom=0.28)

    save_figure(fig_ef3, "efigure3_threshold_overlay")
    fig_ef3
    return (fig_ef3,)


# ── Summary table: all thresholds with classification metrics ─────────
@app.cell
def _(epic_thresholds, mo, morse_thresholds, pl):
    _rows = []

    for _model_name, _thresholds in [("Epic PMFRS", epic_thresholds), ("Morse Fall Scale", morse_thresholds)]:
        for _method, _thresh, _mets, _space in _thresholds:
            _rows.append(
                {
                    "Model": _model_name,
                    "Method": _method,
                    "Threshold": round(_thresh, 4),
                    "Score space": _space,
                    "Sensitivity": round(_mets["sensitivity"], 3),
                    "Specificity": round(_mets["specificity"], 3),
                    "Flag rate, %": round(_mets["flag_rate"], 1),
                    "PPV": round(_mets["ppv"], 3),
                    "NPV": round(_mets["npv"], 3),
                    "NNE": round(_mets["nne"], 1) if _mets["nne"] != float("inf") else None,
                    "TP": _mets["tp"],
                    "FP": _mets["fp"],
                    "FN": _mets["fn"],
                    "TN": _mets["tn"],
                }
            )

    threshold_summary = pl.DataFrame(_rows)
    mo.md("## Threshold Comparison Summary")
    return (threshold_summary,)


@app.cell
def _(mo, threshold_summary):
    mo.ui.table(threshold_summary)
    return


# ── GT rendering: threshold summary table ────────────────────────────
@app.cell
def _(threshold_summary):
    from great_tables import GT, loc, style

    _gt_thresh = (
        GT(threshold_summary)
        .tab_header(
            title="Threshold Method Comparison: Epic PMFRS vs Morse Fall Scale",
            subtitle="Admission scores; seven threshold selection methods",
        )
        .tab_spanner(
            label="Classification metrics",
            columns=["Sensitivity", "Specificity", "Flag rate, %", "PPV", "NPV", "NNE"],
        )
        .tab_spanner(
            label="Confusion matrix",
            columns=["TP", "FP", "FN", "TN"],
        )
        .cols_label(
            **{
                "Method": "Method",
                "Threshold": "Cut-point",
                "Score space": "Score space",
            }
        )
        .sub_missing(missing_text="—")
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .opt_table_font(font="Arial")
        .opt_row_striping()
        .tab_options(
            table_font_size="11px",
            heading_title_font_size="13px",
            heading_subtitle_font_size="11px",
            source_notes_font_size="9px",
        )
        .tab_source_note(
            "Score space: 'Ordinal' = threshold applied to raw score; "
            "'Probability' = threshold applied to logistic-recalibrated probability."
        )
    )
    _gt_thresh
    return


# ── Export threshold summary CSV ─────────────────────────────────────
@app.cell
def _(Path, mo, threshold_summary):
    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _csv_path = _out_dir / "threshold_summary.csv"
    threshold_summary.write_csv(_csv_path)
    mo.md(f"**Saved**: `{_csv_path}` ({threshold_summary.height} rows)")
    return


# ── eFigure 5: Score distribution with threshold overlay (flag rate) ──
@app.cell
def _(mo):
    mo.md(
r"""
## eFigure 5. Score Distributions and Flag Rates

Histograms of admission scores with standard threshold lines overlaid.
Annotations show the percentage of encounters flagged at each threshold
(= flag rate), demonstrating alert burden implications.
"""
    )
    return


@app.cell
def _(
    COLORS,
    EPIC_2TIER_HIGH,
    EPIC_3TIER_HIGH,
    EPIC_3TIER_MEDIUM,
    JAMA_STYLE,
    MFS_HIGH,
    MFS_MODERATE,
    epic_score,
    morse_score,
    np,
    plt,
    save_figure,
    y_true,
):
    def _flag_rate(scores, threshold):
        return float(np.sum(scores >= threshold) / len(scores) * 100)

    with plt.rc_context(JAMA_STYLE):
        fig_dist, (ax_epic, ax_morse) = plt.subplots(1, 2, figsize=(7.0, 3.5))

        # ── Epic panel ──────────────────────────────────────────────
        ax_epic.hist(
            epic_score[y_true == 0],
            bins=100,
            range=(0, 100),
            color=COLORS["ci_fill"],
            edgecolor="none",
            alpha=0.8,
            label="Non-fallers",
        )
        ax_epic.hist(
            epic_score[y_true == 1],
            bins=100,
            range=(0, 100),
            color=COLORS["epic"],
            edgecolor="none",
            alpha=0.7,
            label="Fallers",
        )

        _epic_thresholds = [
            (float(EPIC_3TIER_MEDIUM), "≥35\n(3-tier med)"),
            (float(EPIC_2TIER_HIGH), "≥50\n(2-tier)"),
            (float(EPIC_3TIER_HIGH), "≥70\n(3-tier high)"),
        ]
        for _t, _lbl in _epic_thresholds:
            _fr = _flag_rate(epic_score, _t)
            ax_epic.axvline(_t, color="#333333", linewidth=0.8, linestyle="--", zorder=3)
            ax_epic.text(
                _t + 1, ax_epic.get_ylim()[1] * 0.85, f"{_lbl}\n{_fr:.1f}% flagged",
                fontsize=8, va="top", ha="left", color="#333333",
            )

        ax_epic.set_xlabel("Epic PMFRS score at admission", fontsize=9)
        ax_epic.set_ylabel("Encounters", fontsize=9)
        ax_epic.set_title("Epic PMFRS", fontsize=10, fontweight="bold")
        ax_epic.set_xlim(0, 100)
        ax_epic.tick_params(labelsize=8)
        ax_epic.text(
            -0.14, 1.06, "A", transform=ax_epic.transAxes,
            fontsize=10, fontweight="bold", va="top",
        )

        # ── Morse panel ────────────────────────────────────────────
        _morse_bins = np.arange(-0.5, 130, 5)
        ax_morse.hist(
            morse_score[y_true == 0],
            bins=_morse_bins,
            color=COLORS["ci_fill"],
            edgecolor="none",
            alpha=0.8,
            label="Non-fallers",
        )
        ax_morse.hist(
            morse_score[y_true == 1],
            bins=_morse_bins,
            color=COLORS["morse"],
            edgecolor="none",
            alpha=0.7,
            label="Fallers",
        )

        _morse_thresholds = [
            (float(MFS_MODERATE), "≥25\n(moderate)"),
            (float(MFS_HIGH), "≥45\n(high risk)"),
        ]
        for _t, _lbl in _morse_thresholds:
            _fr = _flag_rate(morse_score, _t)
            ax_morse.axvline(_t, color="#333333", linewidth=0.8, linestyle="--", zorder=3)
            ax_morse.text(
                _t + 2, ax_morse.get_ylim()[1] * 0.85, f"{_lbl}\n{_fr:.1f}% flagged",
                fontsize=8, va="top", ha="left", color="#333333",
            )

        ax_morse.set_xlabel("Morse Fall Scale score at admission", fontsize=9)
        ax_morse.set_ylabel("Encounters", fontsize=9)
        ax_morse.set_title("Morse Fall Scale", fontsize=10, fontweight="bold")
        ax_morse.set_xlim(-2, 130)
        ax_morse.tick_params(labelsize=8)
        ax_morse.text(
            -0.14, 1.06, "B", transform=ax_morse.transAxes,
            fontsize=10, fontweight="bold", va="top",
        )

        _handles, _labels = ax_epic.get_legend_handles_labels()
        fig_dist.legend(
            _handles,
            _labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            fontsize=8,
            frameon=False,
        )
        fig_dist.subplots_adjust(wspace=0.35)

    save_figure(fig_dist, "efigure5_score_distributions")
    fig_dist
    return (fig_dist,)


# ── Flag rate summary table ──────────────────────────────────────────
@app.cell
def _(mo, pl, threshold_summary):
    _standard_cutoffs = [
        "Epic ≥ 35 (3-tier med)",
        "Epic ≥ 50 (2-tier high)",
        "Epic ≥ 70 (3-tier high)",
        "MFS ≥ 25",
        "MFS ≥ 45",
    ]
    flag_rate_df = threshold_summary.filter(
        pl.col("Method").is_in(_standard_cutoffs)
    ).select(["Model", "Method", "Threshold", "Flag rate, %", "Sensitivity", "TP", "FP"])
    mo.md("## Flag Rate Summary")
    return (flag_rate_df,)


@app.cell
def _(flag_rate_df, mo):
    mo.ui.table(flag_rate_df)
    return


@app.cell
def _(Path, flag_rate_df, mo):
    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _csv_path = _out_dir / "flag_rate_summary.csv"
    flag_rate_df.write_csv(_csv_path)
    mo.md(f"**Saved**: `{_csv_path}`")
    return


if __name__ == "__main__":
    app.run()
