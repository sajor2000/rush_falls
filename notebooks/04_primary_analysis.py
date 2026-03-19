import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    return Path, matplotlib, mo, np, pl, plt


@app.cell
def _(mo):
    mo.md(
        """
        # 04 — Primary Discrimination Analysis

        **Purpose**: Evaluate and compare discrimination performance of the Epic Predictive
        Model Fall Risk Score (PMFRS) versus the Morse Fall Scale (MFS) on the analytic cohort.

        **Outputs**:
        - Table 2: AUROC, AUPRC, classification metrics, and flag rates at thresholds (CSV + great-tables)
        - Figure 1: 2×2 multi-panel — sensitivity, specificity, PPV, NPV vs threshold

        **Score timing**: First score at admission (primary analysis per TRIPOD+AI).
        """
    )
    return


# ── Imports: utilities ─────────────────────────────────────────────────────
@app.cell
def _():
    from utils.constants import (
        EPIC_2TIER_HIGH,
        EPIC_3TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        MFS_HIGH,
        MFS_MODERATE,
        N_BOOTSTRAP,
        RANDOM_SEED,
    )
    from utils.metrics import (
        classification_metrics_at_threshold,
        closest_topleft_threshold,
        delong_ci,
        delong_roc_test,
        fixed_sensitivity_threshold,
        logistic_recalibration,
        stratified_bootstrap,
        value_optimizing_threshold,
        youden_threshold,
    )
    from utils.plotting import COLORS, FIG_MULTI_PANEL, JAMA_STYLE, save_figure

    return (
        COLORS,
        EPIC_2TIER_HIGH,
        EPIC_3TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        FIG_MULTI_PANEL,
        JAMA_STYLE,
        MFS_HIGH,
        MFS_MODERATE,
        N_BOOTSTRAP,
        RANDOM_SEED,
        classification_metrics_at_threshold,
        closest_topleft_threshold,
        delong_ci,
        delong_roc_test,
        fixed_sensitivity_threshold,
        logistic_recalibration,
        save_figure,
        stratified_bootstrap,
        value_optimizing_threshold,
        youden_threshold,
    )


# ── 1. Load analytic dataset ───────────────────────────────────────────────
@app.cell
def _(Path, pl):
    df = pl.read_parquet(Path("data/processed/analytic.parquet"))
    return (df,)


@app.cell
def _(df, mo, pl):
    _n = df.height
    _n_falls = df.filter(pl.col("fall_flag") == 1).height
    mo.md(
        f"""
        ## Dataset loaded

        - **Encounters**: {_n:,}
        - **Falls**: {_n_falls:,} ({_n_falls / _n * 100:.1f}%)
        - **Prevalence**: {_n_falls / _n:.4f}
        """
    )
    return


# ── 2. Extract arrays at sklearn boundary ──────────────────────────────────
@app.cell
def _(df, pl):
    # Defensive: parquet is pre-filtered, but explicit for clarity
    df_cc = df.filter(
        pl.col("epic_score_admission").is_not_null()
        & pl.col("morse_score_admission").is_not_null()
    )
    y_true = df_cc["fall_flag"].to_numpy()
    epic_scores = df_cc["epic_score_admission"].to_numpy()
    morse_scores = df_cc["morse_score_admission"].to_numpy()
    return df_cc, epic_scores, morse_scores, y_true


# ── 2b. Logistic recalibration: score → probability (for Figure 1) ────
@app.cell
def _(epic_scores, logistic_recalibration, morse_scores, y_true):
    epic_prob, epic_lr = logistic_recalibration(epic_scores, y_true)
    morse_prob, morse_lr = logistic_recalibration(morse_scores, y_true)
    return epic_lr, epic_prob, morse_lr, morse_prob


# ── 3. AUROC with DeLong CIs ───────────────────────────────────────────────
@app.cell
def _(delong_ci, epic_scores, morse_scores, y_true):
    epic_auc, epic_ci_lo, epic_ci_hi = delong_ci(y_true, epic_scores)
    morse_auc, morse_ci_lo, morse_ci_hi = delong_ci(y_true, morse_scores)
    return epic_auc, epic_ci_hi, epic_ci_lo, morse_auc, morse_ci_hi, morse_ci_lo


# ── 4. AUPRC ─────────────────────────────────────────────────────────────
@app.cell
def _(epic_scores, morse_scores, y_true):
    from sklearn.metrics import average_precision_score

    epic_auprc = float(average_precision_score(y_true, epic_scores))
    morse_auprc = float(average_precision_score(y_true, morse_scores))
    return epic_auprc, morse_auprc


# ── 5. Paired DeLong test ─────────────────────────────────────────────────
@app.cell
def _(delong_roc_test, epic_scores, morse_scores, y_true):
    delong_p = delong_roc_test(y_true, epic_scores, morse_scores)
    return (delong_p,)


@app.cell
def _(boot_results, delong_p, epic_auc, epic_auprc, epic_ci_hi, epic_ci_lo, mo, morse_auc, morse_auprc, morse_ci_hi, morse_ci_lo):
    _epic_auprc_boot = boot_results["auprc_a"]
    _morse_auprc_boot = boot_results["auprc_b"]
    mo.md(
        f"""
        ## Discrimination Summary

        | Model | AUROC (95% CI) | AUPRC (95% CI) |
        |---|---|---|
        | Epic PMFRS | {epic_auc:.3f} ({epic_ci_lo:.3f}–{epic_ci_hi:.3f}) | {epic_auprc:.3f} ({_epic_auprc_boot['ci_lower']:.3f}–{_epic_auprc_boot['ci_upper']:.3f}) |
        | Morse Fall Scale | {morse_auc:.3f} ({morse_ci_lo:.3f}–{morse_ci_hi:.3f}) | {morse_auprc:.3f} ({_morse_auprc_boot['ci_lower']:.3f}–{_morse_auprc_boot['ci_upper']:.3f}) |

        **Paired DeLong p-value**: {delong_p:.4f}
        """
    )
    return


# ── 6. Threshold selection ────────────────────────────────────────────────
@app.cell
def _(
    EPIC_2TIER_HIGH,
    EPIC_3TIER_HIGH,
    EPIC_3TIER_MEDIUM,
    MFS_HIGH,
    MFS_MODERATE,
    closest_topleft_threshold,
    epic_prob,
    epic_scores,
    fixed_sensitivity_threshold,
    morse_prob,
    morse_scores,
    value_optimizing_threshold,
    y_true,
    youden_threshold,
):
    # Epic thresholds (data-driven, on ordinal scores)
    epic_t_youden = youden_threshold(y_true, epic_scores)
    epic_t_topleft = closest_topleft_threshold(y_true, epic_scores)
    epic_t_sens60 = fixed_sensitivity_threshold(y_true, epic_scores, 0.60)
    epic_t_sens80 = fixed_sensitivity_threshold(y_true, epic_scores, 0.80)
    # Value-optimizing requires calibrated probabilities, not ordinal scores
    epic_t_value, _, _ = value_optimizing_threshold(y_true, epic_prob)

    # Morse thresholds (data-driven, on ordinal scores)
    morse_t_youden = youden_threshold(y_true, morse_scores)
    morse_t_topleft = closest_topleft_threshold(y_true, morse_scores)
    morse_t_sens60 = fixed_sensitivity_threshold(y_true, morse_scores, 0.60)
    morse_t_sens80 = fixed_sensitivity_threshold(y_true, morse_scores, 0.80)
    # Value-optimizing requires calibrated probabilities
    morse_t_value, _, _ = value_optimizing_threshold(y_true, morse_prob)

    # Standard clinical cutoffs (fixed thresholds)
    morse_t_moderate = float(MFS_MODERATE)
    morse_t_high = float(MFS_HIGH)
    epic_t_3tier_medium = float(EPIC_3TIER_MEDIUM)
    epic_t_3tier_high = float(EPIC_3TIER_HIGH)
    epic_t_2tier_high = float(EPIC_2TIER_HIGH)

    return (
        epic_t_2tier_high,
        epic_t_3tier_high,
        epic_t_3tier_medium,
        epic_t_sens60,
        epic_t_sens80,
        epic_t_topleft,
        epic_t_value,
        epic_t_youden,
        morse_t_high,
        morse_t_moderate,
        morse_t_sens60,
        morse_t_sens80,
        morse_t_topleft,
        morse_t_value,
        morse_t_youden,
    )


# ── 6b. Probability equivalents at standard + Rush-optimal cutoffs ────────
@app.cell
def _(
    EPIC_2TIER_HIGH,
    EPIC_3TIER_HIGH,
    EPIC_3TIER_MEDIUM,
    MFS_HIGH,
    MFS_MODERATE,
    epic_lr,
    epic_scores,
    epic_t_youden,
    mo,
    morse_lr,
    morse_t_youden,
    np,
):
    # Probability equivalents at Epic standard cutoffs
    _epic_prob_at_35 = float(epic_lr.predict_proba([[float(EPIC_3TIER_MEDIUM)]])[0, 1])
    _epic_prob_at_50 = float(epic_lr.predict_proba([[float(EPIC_2TIER_HIGH)]])[0, 1])
    _epic_prob_at_70 = float(epic_lr.predict_proba([[float(EPIC_3TIER_HIGH)]])[0, 1])

    # Probability equivalents at Morse standard cutoffs
    _morse_prob_at_25 = float(morse_lr.predict_proba([[float(MFS_MODERATE)]])[0, 1])
    _morse_prob_at_45 = float(morse_lr.predict_proba([[float(MFS_HIGH)]])[0, 1])

    # Rush Youden probability equivalents
    _epic_prob_at_youden = float(epic_lr.predict_proba([[epic_t_youden]])[0, 1])
    _morse_prob_at_youden = float(morse_lr.predict_proba([[morse_t_youden]])[0, 1])

    _pct_below_35 = float(np.sum(epic_scores < 35) / len(epic_scores) * 100)

    mo.md(
        f"""
        ## Probability Equivalents at Standard and Rush-Optimal Cutoffs

        Logistic recalibration maps ordinal scores to predicted fall probabilities.
        These equivalents show where each threshold falls on the probability scale.

        | Model | Cutoff | Ordinal threshold | Probability equivalent |
        |---|---|---|---|
        | Epic PMFRS | 3-tier medium (≥35) | 35 | {_epic_prob_at_35:.4f} |
        | Epic PMFRS | 2-tier high (≥50) | 50 | {_epic_prob_at_50:.4f} |
        | Epic PMFRS | 3-tier high (≥70) | 70 | {_epic_prob_at_70:.4f} |
        | Epic PMFRS | Rush Youden | {epic_t_youden:.2f} | {_epic_prob_at_youden:.4f} |
        | Morse Fall Scale | Moderate (≥25) | 25 | {_morse_prob_at_25:.4f} |
        | Morse Fall Scale | High (≥45) | 45 | {_morse_prob_at_45:.4f} |
        | Morse Fall Scale | Rush Youden | {morse_t_youden:.2f} | {_morse_prob_at_youden:.4f} |

        **Note**: Epic thresholds (35/50/70) are designed for continuous monitoring, not
        admission screening. At admission, {_pct_below_35:.0f}% of encounters score < 35, so these
        cutoffs map to high probability equivalents with very low flag rates.
        """
    )
    return


# ── 7. Classification metrics at each threshold ───────────────────────────
@app.cell
def _(
    classification_metrics_at_threshold,
    epic_prob,
    epic_scores,
    epic_t_2tier_high,
    epic_t_3tier_high,
    epic_t_3tier_medium,
    epic_t_sens60,
    epic_t_sens80,
    epic_t_topleft,
    epic_t_value,
    epic_t_youden,
    morse_prob,
    morse_scores,
    morse_t_high,
    morse_t_moderate,
    morse_t_sens60,
    morse_t_sens80,
    morse_t_topleft,
    morse_t_value,
    morse_t_youden,
    y_true,
):
    # Epic: data-driven thresholds applied to ordinal scores
    _epic_threshold_labels = [
        ("Youden index", epic_t_youden, epic_scores),
        ("Closest to (0,1)", epic_t_topleft, epic_scores),
        ("Fixed sensitivity 60%", epic_t_sens60, epic_scores),
        ("Fixed sensitivity 80%", epic_t_sens80, epic_scores),
        # Value-optimizing threshold is in probability space
        ("Value-optimizing (NMB)", epic_t_value, epic_prob),
        # Epic standard cutoffs (ordinal score space)
        ("Standard cutoff: Epic ≥35 (3-tier medium)", epic_t_3tier_medium, epic_scores),
        ("Standard cutoff: Epic ≥70 (3-tier high)", epic_t_3tier_high, epic_scores),
        ("Standard cutoff: Epic ≥50 (2-tier high)", epic_t_2tier_high, epic_scores),
    ]

    # Morse: data-driven thresholds applied to ordinal scores
    _morse_threshold_labels = [
        ("Youden index", morse_t_youden, morse_scores),
        ("Closest to (0,1)", morse_t_topleft, morse_scores),
        ("Fixed sensitivity 60%", morse_t_sens60, morse_scores),
        ("Fixed sensitivity 80%", morse_t_sens80, morse_scores),
        # Value-optimizing threshold is in probability space
        ("Value-optimizing (NMB)", morse_t_value, morse_prob),
        ("Standard cutoff: MFS ≥25", morse_t_moderate, morse_scores),
        ("Standard cutoff: MFS ≥45", morse_t_high, morse_scores),
    ]

    epic_metrics_by_threshold: list[dict] = []
    for _label, _t, _score_arr in _epic_threshold_labels:
        _m = classification_metrics_at_threshold(y_true, _score_arr, _t)
        _m["label"] = _label
        epic_metrics_by_threshold.append(_m)

    morse_metrics_by_threshold: list[dict] = []
    for _label, _t, _score_arr in _morse_threshold_labels:
        _m = classification_metrics_at_threshold(y_true, _score_arr, _t)
        _m["label"] = _label
        morse_metrics_by_threshold.append(_m)

    return epic_metrics_by_threshold, morse_metrics_by_threshold


# ── 8. Bootstrap CIs for classification metrics ───────────────────────────
@app.cell
def _(N_BOOTSTRAP, RANDOM_SEED, epic_scores, morse_scores, stratified_bootstrap, y_true):
    boot_results = stratified_bootstrap(
        y_true, epic_scores, pred_b=morse_scores, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED
    )
    return (boot_results,)


@app.cell
def _(boot_results, mo):
    _epic_boot = boot_results["auc_a"]
    _morse_boot = boot_results["auc_b"]
    _diff_boot = boot_results["auc_diff"]
    _epic_auprc_b = boot_results["auprc_a"]
    _morse_auprc_b = boot_results["auprc_b"]
    _auprc_diff_b = boot_results["auprc_diff"]
    mo.md(
        f"""
        ## Bootstrap CIs (2000 stratified resamples)

        | Metric | Estimate | 95% CI |
        |---|---|---|
        | Epic AUROC | {_epic_boot['estimate']:.3f} | {_epic_boot['ci_lower']:.3f}–{_epic_boot['ci_upper']:.3f} |
        | Morse AUROC | {_morse_boot['estimate']:.3f} | {_morse_boot['ci_lower']:.3f}–{_morse_boot['ci_upper']:.3f} |
        | AUROC difference (Epic − Morse) | {_diff_boot['estimate']:.3f} | {_diff_boot['ci_lower']:.3f}–{_diff_boot['ci_upper']:.3f} |
        | Epic AUPRC | {_epic_auprc_b['estimate']:.3f} | {_epic_auprc_b['ci_lower']:.3f}–{_epic_auprc_b['ci_upper']:.3f} |
        | Morse AUPRC | {_morse_auprc_b['estimate']:.3f} | {_morse_auprc_b['ci_lower']:.3f}–{_morse_auprc_b['ci_upper']:.3f} |
        | AUPRC difference (Epic − Morse) | {_auprc_diff_b['estimate']:.3f} | {_auprc_diff_b['ci_lower']:.3f}–{_auprc_diff_b['ci_upper']:.3f} |
        """
    )
    return


# ── 9. Build Table 2 as Polars DataFrame ──────────────────────────────────
@app.cell
def _(
    boot_results,
    delong_p,
    epic_auc,
    epic_auprc,
    epic_ci_hi,
    epic_ci_lo,
    epic_metrics_by_threshold,
    morse_auc,
    morse_auprc,
    morse_ci_hi,
    morse_ci_lo,
    morse_metrics_by_threshold,
    pl,
):
    def _fmt_ci(est: float, lo: float, hi: float, decimals: int = 3) -> str:
        d = f".{decimals}f"
        return f"{est:{d}} ({lo:{d}}–{hi:{d}})"

    def _fmt_pct_ci(est: float, lo: float, hi: float) -> str:
        return f"{est * 100:.1f} ({lo * 100:.1f}–{hi * 100:.1f})"

    def _metrics_to_row(m: dict, model: str) -> dict:
        return {
            "Model": model,
            "Threshold label": m["label"],
            "Threshold value": round(m["threshold"], 2),
            "Sensitivity, %": round(m["sensitivity"] * 100, 1),
            "Specificity, %": round(m["specificity"] * 100, 1),
            "Flag rate, %": round(m["flag_rate"], 1),
            "PPV, %": round(m["ppv"] * 100, 1),
            "NPV, %": round(m["npv"] * 100, 1),
            "NNE": round(m["nne"], 1) if m["nne"] != float("inf") else None,
            "TP": m["tp"],
            "FP": m["fp"],
            "FN": m["fn"],
            "TN": m["tn"],
        }

    # Header rows (one per model — AUROC + AUPRC)
    _epic_header = {
        "Model": "Epic PMFRS",
        "Threshold label": "Overall",
        "Threshold value": None,
        "Sensitivity, %": None,
        "Specificity, %": None,
        "Flag rate, %": None,
        "PPV, %": None,
        "NPV, %": None,
        "NNE": None,
        "TP": None,
        "FP": None,
        "FN": None,
        "TN": None,
        "AUROC (95% CI)": _fmt_ci(epic_auc, epic_ci_lo, epic_ci_hi),
        "AUPRC (95% CI)": _fmt_ci(
            epic_auprc,
            boot_results["auprc_a"]["ci_lower"],
            boot_results["auprc_a"]["ci_upper"],
        ),
        "DeLong p": round(delong_p, 4),
        "AUROC bootstrap (95% CI)": _fmt_ci(
            boot_results["auc_a"]["estimate"],
            boot_results["auc_a"]["ci_lower"],
            boot_results["auc_a"]["ci_upper"],
        ),
    }
    _morse_header = {
        "Model": "Morse Fall Scale",
        "Threshold label": "Overall",
        "Threshold value": None,
        "Sensitivity, %": None,
        "Specificity, %": None,
        "Flag rate, %": None,
        "PPV, %": None,
        "NPV, %": None,
        "NNE": None,
        "TP": None,
        "FP": None,
        "FN": None,
        "TN": None,
        "AUROC (95% CI)": _fmt_ci(morse_auc, morse_ci_lo, morse_ci_hi),
        "AUPRC (95% CI)": _fmt_ci(
            morse_auprc,
            boot_results["auprc_b"]["ci_lower"],
            boot_results["auprc_b"]["ci_upper"],
        ),
        "DeLong p": round(delong_p, 4),
        "AUROC bootstrap (95% CI)": _fmt_ci(
            boot_results["auc_b"]["estimate"],
            boot_results["auc_b"]["ci_lower"],
            boot_results["auc_b"]["ci_upper"],
        ),
    }

    # Threshold rows
    _threshold_rows = []
    for _m in epic_metrics_by_threshold:
        _row = _metrics_to_row(_m, "Epic PMFRS")
        _row["AUROC (95% CI)"] = None
        _row["AUPRC (95% CI)"] = None
        _row["DeLong p"] = None
        _row["AUROC bootstrap (95% CI)"] = None
        _threshold_rows.append(_row)

    for _m in morse_metrics_by_threshold:
        _row = _metrics_to_row(_m, "Morse Fall Scale")
        _row["AUROC (95% CI)"] = None
        _row["AUPRC (95% CI)"] = None
        _row["DeLong p"] = None
        _row["AUROC bootstrap (95% CI)"] = None
        _threshold_rows.append(_row)

    _all_rows = [_epic_header, _morse_header] + _threshold_rows

    table2_df = pl.DataFrame(_all_rows, infer_schema_length=len(_all_rows))
    return (table2_df,)


@app.cell
def _(mo, table2_df):
    mo.md("## Table 2 Preview")
    mo.ui.table(table2_df)
    return


# ── 10. Export Table 2 CSV ────────────────────────────────────────────────
@app.cell
def _(Path, mo, table2_df):
    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _csv_path = _out_dir / "table2.csv"
    table2_df.write_csv(_csv_path)
    mo.md(f"**Saved**: `{_csv_path}` ({table2_df.height} rows)")
    return


# ── 11. Render Table 2 with great-tables ─────────────────────────────────
@app.cell
def _(table2_df):
    from great_tables import GT, loc, style

    _display_cols = [
        "Model",
        "Threshold label",
        "Threshold value",
        "Sensitivity, %",
        "Specificity, %",
        "Flag rate, %",
        "PPV, %",
        "NPV, %",
        "NNE",
        "AUROC (95% CI)",
        "AUPRC (95% CI)",
        "DeLong p",
    ]

    _df_display = table2_df.select(
        [c for c in _display_cols if c in table2_df.columns]
    )

    gt_table2 = (
        GT(_df_display)
        .tab_header(
            title="Table 2. Discrimination performance of Epic PMFRS vs Morse Fall Scale",
            subtitle="Admission scores, complete-case analytic cohort",
        )
        .tab_spanner(
            label="Classification metrics",
            columns=["Sensitivity, %", "Specificity, %", "Flag rate, %", "PPV, %", "NPV, %", "NNE"],
        )
        .tab_spanner(
            label="Overall discrimination",
            columns=["AUROC (95% CI)", "AUPRC (95% CI)", "DeLong p"],
        )
        .cols_label(
            **{
                "Threshold label": "Threshold method",
                "Threshold value": "Cut-point",
                "Sensitivity, %": "Sensitivity",
                "Specificity, %": "Specificity",
                "Flag rate, %": "Flag rate",
                "AUROC (95% CI)": "AUROC (95% CI)",
                "DeLong p": "DeLong p",
            }
        )
        .sub_missing(missing_text="—")
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.body(
                rows=[
                    i for i, v in enumerate(table2_df["Threshold label"].to_list())
                    if v == "Overall"
                ]
            ),
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
            "AUROC 95% CI via DeLong variance (Sun & Xu 2014). "
            "Bootstrap: 2000 stratified resamples, seed 42. "
            "Flag rate = proportion of encounters exceeding threshold. "
            "NNE = number needed to evaluate (1/PPV)."
        )
    )
    gt_table2
    return (gt_table2,)


# ── 12. Figure 1: 2×2 multi-panel discrimination plot ────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## Figure 1: Sensitivity, Specificity, PPV, NPV across thresholds

        Epic (blue #2166AC) vs Morse (red #B2182B). Thresholds derived from `roc_curve`.
        Panel layout: 2 rows × 2 columns. Full-width 7.0 × 8.0 in.
        """
    )
    return


@app.cell
def _(
    COLORS,
    FIG_MULTI_PANEL,
    JAMA_STYLE,
    epic_prob,
    matplotlib,
    morse_prob,
    np,
    plt,
    save_figure,
    y_true,
):
    from sklearn.metrics import roc_curve as _roc_curve

    # ── Compute per-threshold metrics on recalibrated probabilities ──
    def _threshold_curve(y_true_arr, probs):
        _fpr, _tpr, _thresholds = _roc_curve(y_true_arr, probs)
        _sens_arr = []
        _spec_arr = []
        _ppv_arr = []
        _npv_arr = []
        for _t in _thresholds:
            _pred = probs >= _t
            _tp = int((_pred & (y_true_arr == 1)).sum())
            _fp = int((_pred & (y_true_arr == 0)).sum())
            _fn = int((~_pred & (y_true_arr == 1)).sum())
            _tn = int((~_pred & (y_true_arr == 0)).sum())
            _sens_arr.append(_tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0)
            _spec_arr.append(_tn / (_tn + _fp) if (_tn + _fp) > 0 else 0.0)
            _ppv_arr.append(_tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0)
            _npv_arr.append(_tn / (_tn + _fn) if (_tn + _fn) > 0 else 0.0)
        return _thresholds, _sens_arr, _spec_arr, _ppv_arr, _npv_arr

    _epic_t, _epic_sens, _epic_spec, _epic_ppv, _epic_npv = _threshold_curve(
        y_true, epic_prob
    )
    _morse_t, _morse_sens, _morse_spec, _morse_ppv, _morse_npv = _threshold_curve(
        y_true, morse_prob
    )

    # Data-driven x-axis limit
    _x_max = min(max(float(np.max(epic_prob)), float(np.max(morse_prob))) * 1.15, 1.0)

    # ── Build figure ──────────────────────────────────────────────────
    with matplotlib.rc_context(JAMA_STYLE):
        _fig, _axes = plt.subplots(2, 2, figsize=FIG_MULTI_PANEL)
        _fig.subplots_adjust(hspace=0.42, wspace=0.38)

        _panels = [
            ("Sensitivity", _epic_sens, _morse_sens),
            ("Specificity", _epic_spec, _morse_spec),
            ("PPV", _epic_ppv, _morse_ppv),
            ("NPV", _epic_npv, _morse_npv),
        ]
        _panel_labels = ["A", "B", "C", "D"]

        for _idx, ((_title, _epic_y, _morse_y), _ax, _lbl) in enumerate(
            zip(_panels, _axes.flat, _panel_labels, strict=True)
        ):
            _ax.plot(
                _epic_t,
                _epic_y,
                color=COLORS["epic"],
                linewidth=1.0,
                label="Epic PMFRS",
            )
            _ax.plot(
                _morse_t,
                _morse_y,
                color=COLORS["morse"],
                linewidth=1.0,
                label="Morse Fall Scale",
                linestyle="--",
            )
            _ax.set_xlabel("Predicted probability", fontsize=9)
            _ax.set_ylabel(_title, fontsize=9)
            _ax.set_ylim(-0.02, 1.05)
            _ax.set_xlim(0, _x_max)
            _ax.tick_params(labelsize=8)

            # Panel label (bold, top-left)
            _ax.text(
                -0.18,
                1.06,
                _lbl,
                transform=_ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )

        # Shared legend below figure — outside all panels
        _handles, _labels_leg = _axes.flat[0].get_legend_handles_labels()
        _fig.legend(
            _handles,
            _labels_leg,
            loc="lower center",
            ncol=2,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )

        _fig.text(
            0.5, -0.06,
            "Figure 1. Classification metrics across score thresholds:\n"
            "Epic PMFRS vs Morse Fall Scale",
            ha="center", va="top", fontsize=10, fontweight="bold",
        )

        save_figure(_fig, "figure1_discrimination")

    figure1_fig = _fig
    return figure1_fig,


@app.cell
def _(mo):
    mo.md(
        """
        **Figure 1 saved** to `outputs/figures/figure1_discrimination.pdf` and `.png`.

        Legend placed below the panels; no data is obscured. Axis labels in sentence case
        per JAMA format. Minimum 8 pt text throughout.
        """
    )
    return


if __name__ == "__main__":
    app.run()
