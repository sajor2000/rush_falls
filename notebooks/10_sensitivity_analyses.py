import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl

    return Path, mo, np, pl


@app.cell
def _(mo):
    mo.md("""
    # 10 — Sensitivity Analyses (eTable 4)

    **Purpose**: Evaluate robustness of discrimination estimates across alternative
    score timing strategies and encounter-selection approaches.

    **Strategies tested**:
    1. Admission score (primary analysis)
    2. Score before fall / before discharge (`before_fall`)
    3. Maximum score during encounter (`max`)
    4. Mean score during encounter (`mean`)
    5. First encounter per patient only (de-duplicated cohort)

    **Output**: eTable 4 — AUROC (95% CI) and DeLong p-value for each strategy.

    **Reference**: TRIPOD+AI Items 13 (sensitivity analyses) and 10b (timing).
    """)
    return


@app.cell
def _():
    from utils.constants import SCORE_TIMING
    from utils.metrics import delong_ci, delong_roc_test

    return SCORE_TIMING, delong_ci, delong_roc_test


@app.cell
def _():
    from utils.cluster_auroc import (
        cluster_bootstrap_auroc_comparison,
        estimate_design_effect,
    )

    return cluster_bootstrap_auroc_comparison, estimate_design_effect


@app.cell
def _(Path, pl):
    df = pl.read_parquet(Path("data/processed/analytic.parquet"))
    return (df,)


@app.cell
def _(df, mo):
    mo.md(
        f"""
        ## Dataset

        - **Encounters**: {df.height:,}
        - **Falls**: {df['fall_flag'].sum():,} ({df['fall_flag'].mean() * 100:.1f}%)
        """
    )
    return


@app.cell
def _(delong_ci, delong_roc_test):

    def run_timing_analysis(
        sub_df,
        epic_col: str,
        morse_col: str,
        timing_label: str,
        feasibility_note: str = "",
    ) -> dict:
        """Compute AUROC + DeLong CI + paired p-value for one timing strategy.

        Returns a result dict ready for eTable 4.
        """
        # Filter to complete cases for this timing
        _complete = sub_df.filter(
            sub_df[epic_col].is_not_null() & sub_df[morse_col].is_not_null()
        )

        n = _complete.height
        n_falls = int(_complete["fall_flag"].sum())

        base_row = {
            "Timing strategy": timing_label,
            "N encounters": n,
            "N falls": n_falls,
        }

        # Feasibility check: need events for both fallers and non-fallers
        if n_falls < 20 or (n - n_falls) < 20:
            base_row.update(
                {
                    "Epic AUROC": "—",
                    "Epic 95% CI": "—",
                    "Morse AUROC": "—",
                    "Morse 95% CI": "—",
                    "DeLong p": "—",
                    "Note": feasibility_note or "Insufficient events",
                }
            )
            return base_row

        y_true = _complete["fall_flag"].to_numpy()
        epic_scores = _complete[epic_col].to_numpy()
        morse_scores = _complete[morse_col].to_numpy()

        # DeLong CIs for each model
        epic_auc, epic_lo, epic_hi = delong_ci(y_true, epic_scores)
        morse_auc, morse_lo, morse_hi = delong_ci(y_true, morse_scores)

        # Paired DeLong p-value
        p_val = delong_roc_test(y_true, epic_scores, morse_scores)

        # Format p-value
        if p_val < 0.001:
            p_str = "<0.001"
        elif p_val < 0.01:
            p_str = f"{p_val:.3f}"
        else:
            p_str = f"{p_val:.2f}"

        base_row.update(
            {
                "Epic AUROC": f"{epic_auc:.3f}",
                "Epic 95% CI": f"({epic_lo:.3f}–{epic_hi:.3f})",
                "Morse AUROC": f"{morse_auc:.3f}",
                "Morse 95% CI": f"({morse_lo:.3f}–{morse_hi:.3f})",
                "DeLong p": p_str,
                "Note": feasibility_note,
            }
        )
        return base_row

    return (run_timing_analysis,)


@app.cell
def _(SCORE_TIMING, df, run_timing_analysis):
    _timing = SCORE_TIMING["admission"]
    row_admission = run_timing_analysis(
        df,
        epic_col=_timing["epic"],
        morse_col=_timing["morse"],
        timing_label="Admission (primary)",
    )
    return (row_admission,)


@app.cell
def _(SCORE_TIMING, df, mo, pl, run_timing_analysis):
    _timing = SCORE_TIMING["before_fall"]
    _epic_col = _timing["epic"]
    _morse_col = _timing["morse"]

    # Feasibility check: for non-fallers, before_fall scores are inherently null
    # (they represent the last score before a fall that did not occur).
    # Assess data availability to determine if analysis is feasible.
    _n_epic_nonnull = df.filter(pl.col(_epic_col).is_not_null()).height
    _n_morse_nonnull = df.filter(pl.col(_morse_col).is_not_null()).height
    _n_fallers = int(df["fall_flag"].sum())
    _n_nonfallers_with_score = df.filter(
        (pl.col("fall_flag") == 0) & pl.col(_epic_col).is_not_null()
    ).height

    # If non-fallers have near-zero before_fall scores, analysis is biased/infeasible
    _feasibility_note = ""
    if _n_nonfallers_with_score < 100:
        _feasibility_note = (
            f"Caution: only {_n_nonfallers_with_score} non-fallers have "
            "before_fall scores; non-faller comparator is last pre-discharge score"
        )

    row_before_fall = run_timing_analysis(
        df,
        epic_col=_epic_col,
        morse_col=_morse_col,
        timing_label="Before fall / discharge",
        feasibility_note=_feasibility_note,
    )

    mo.md(
        f"""
        ### Before-Fall Score Feasibility Check

        - Non-null Epic before-fall scores: {_n_epic_nonnull:,}
        - Non-null Morse before-fall scores: {_n_morse_nonnull:,}
        - Non-fallers with before-fall score: {_n_nonfallers_with_score:,}

        *Note*: Before-fall scores carry moderate look-ahead risk. For fallers,
        the score immediately prior to the event is used; for non-fallers, the
        last documented score before discharge. Interpret with caution.
        """
    )
    return (row_before_fall,)


@app.cell
def _(SCORE_TIMING, df, run_timing_analysis):
    _timing = SCORE_TIMING["max"]
    row_max = run_timing_analysis(
        df,
        epic_col=_timing["epic"],
        morse_col=_timing["morse"],
        timing_label="Maximum (encounter)",
        feasibility_note="High look-ahead bias — post-fall scores included",
    )
    return (row_max,)


@app.cell
def _(SCORE_TIMING, df, run_timing_analysis):
    _timing = SCORE_TIMING["mean"]
    row_mean = run_timing_analysis(
        df,
        epic_col=_timing["epic"],
        morse_col=_timing["morse"],
        timing_label="Mean (encounter)",
        feasibility_note="Moderate look-ahead bias — post-fall scores included",
    )
    return (row_mean,)


@app.cell
def _(SCORE_TIMING, df, mo, run_timing_analysis):
    # De-duplicate to first encounter per patient (by admission date)
    _df_first = (
        df.sort("admission_date")
        .group_by("mrn")
        .first()
    )

    _n_first = _df_first.height
    _n_falls_first = int(_df_first["fall_flag"].sum())

    _timing = SCORE_TIMING["admission"]
    row_first_encounter = run_timing_analysis(
        _df_first,
        epic_col=_timing["epic"],
        morse_col=_timing["morse"],
        timing_label="Admission — first encounter per patient",
        feasibility_note="Reduced clustering; independent observations per patient",
    )

    mo.md(
        f"""
        ### First Encounter per Patient

        De-duplicated cohort (first admission per MRN):
        - Encounters: {_n_first:,}
        - Falls: {_n_falls_first:,} ({_n_falls_first / _n_first * 100:.1f}%)
        """
    )
    return (row_first_encounter,)


@app.cell
def _(
    SCORE_TIMING,
    cluster_bootstrap_auroc_comparison,
    df,
    estimate_design_effect,
    mo,
):
    _timing = SCORE_TIMING["admission"]
    _epic_col = _timing["epic"]
    _morse_col = _timing["morse"]

    _y = df["fall_flag"].to_numpy()
    _epic = df[_epic_col].to_numpy()
    _morse = df[_morse_col].to_numpy()
    _mrn = df["mrn"].to_numpy()

    # Design effect
    _deff = estimate_design_effect(_y, _epic, _mrn)

    # Cluster-bootstrap AUROC comparison
    _cb = cluster_bootstrap_auroc_comparison(
        _y, _epic, _morse, _mrn, n_bootstrap=2000, seed=42, method="bca"
    )

    # Format p-value
    if _cb.p_value < 0.001:
        _p_str = "<0.001"
    elif _cb.p_value < 0.01:
        _p_str = f"{_cb.p_value:.3f}"
    else:
        _p_str = f"{_cb.p_value:.2f}"

    row_cluster = {
        "Timing strategy": "Admission — cluster bootstrap",
        "N encounters": _cb.n_obs,
        "N falls": int(_y.sum()),
        "Epic AUROC": f"{_cb.auc_a:.3f}",
        "Epic 95% CI": f"({_cb.ci_lower:.3f}–{_cb.ci_upper:.3f})",
        "Morse AUROC": f"{_cb.auc_b:.3f}",
        "Morse 95% CI": "(cluster-adjusted)",
        "DeLong p": _p_str,
        "Note": (
            f"Cluster bootstrap (patient-level, n={_cb.n_clusters} clusters); "
            f"DEFF={_deff['deff_overall']}, ICC={_deff['icc_estimate']}"
        ),
    }

    mo.md(
        f"""
        ### Cluster-Adjusted AUROC (Sensitivity)

        - Design effect: {_deff['deff_overall']}
        - ICC (scores): {_deff['icc_estimate']}
        - CI widening factor: {_deff['ci_widening_factor']}
        - Cluster-bootstrap delta 95% CI: ({_cb.ci_lower:.4f}–{_cb.ci_upper:.4f})
        - Cluster-bootstrap p: {_p_str}
        """
    )
    return (row_cluster,)


@app.cell
def _(df, mo, np, pl):
    _fallers_raw = df.filter(
        pl.col("fall_flag") == 1,
        pl.col("fall_datetime").is_not_null(),
        pl.col("admission_date").is_not_null(),
    ).with_columns(
        ((pl.col("fall_datetime") - pl.col("admission_date")).dt.total_seconds() / 3600)
        .alias("lead_time_hours")
    )
    _n_raw = _fallers_raw.height
    _fallers = _fallers_raw.filter(pl.col("lead_time_hours") > 0)
    _n_excluded = _n_raw - _fallers.height

    if _fallers.height == 0:
        lead_time_stats = {"n": 0, "n_excluded": _n_excluded}
        _output = mo.md(
            f"**Warning**: No valid admission-to-fall lead times found "
            f"({_n_excluded} excluded with non-positive values)."
        )
    else:
        _lt = _fallers["lead_time_hours"].to_numpy()
        _median = float(np.median(_lt))
        _q25 = float(np.percentile(_lt, 25))
        _q75 = float(np.percentile(_lt, 75))
        _min_lt = float(np.min(_lt))
        _max_lt = float(np.max(_lt))
        _n = len(_lt)

        lead_time_stats = {
            "n": _n,
            "n_excluded": _n_excluded,
            "median_hours": round(_median, 1),
            "q25_hours": round(_q25, 1),
            "q75_hours": round(_q75, 1),
            "median_days": round(_median / 24, 1),
            "q25_days": round(_q25 / 24, 1),
            "q75_days": round(_q75 / 24, 1),
        }
        _output = mo.md(
            f"""
    ### Admission-to-Fall Lead Time

    Among fallers with valid timestamps (n={_n}, {_n_excluded} excluded
    with non-positive lead time):

    - **Median lead time**: {_median:.1f} hours ({_median / 24:.1f} days)
    - **IQR**: {_q25:.1f}–{_q75:.1f} hours ({_q25 / 24:.1f}–{_q75 / 24:.1f} days)
    - **Range**: {_min_lt:.1f}–{_max_lt:.1f} hours

    This represents the prediction-to-event window — the time between
    the admission score (predictor) and the fall event (outcome).
    Per Wong et al., this frames the clinical utility of admission-time
    screening: longer lead times give more opportunity for intervention.
    """
        )

    _output
    return (lead_time_stats,)


@app.cell
def _(
    mo,
    pl,
    row_admission,
    row_before_fall,
    row_cluster,
    row_first_encounter,
    row_max,
    row_mean,
):
    _rows = [
        row_admission,
        row_before_fall,
        row_max,
        row_mean,
        row_first_encounter,
        row_cluster,
    ]
    etable4_df = pl.DataFrame(_rows)

    mo.md("## eTable 4 — Sensitivity Analyses: AUROC Across Score Timing Strategies")
    return (etable4_df,)


@app.cell
def _(etable4_df):
    from great_tables import GT, loc, style

    _gt = (
        GT(etable4_df)
        .tab_header(
            title="eTable 4. Sensitivity Analyses: Discrimination by Score Timing Strategy",
            subtitle=(
                "AUROC (95% CI) via DeLong variance; "
                "paired DeLong p-value for Epic vs Morse on same encounters"
            ),
        )
        .tab_spanner(
            label="Epic PMFRS",
            columns=["Epic AUROC", "Epic 95% CI"],
        )
        .tab_spanner(
            label="Morse Fall Scale",
            columns=["Morse AUROC", "Morse 95% CI"],
        )
        .cols_label(
            cases={
                "Timing strategy": "Score timing",
                "N encounters": "N encounters",
                "N falls": "N falls",
                "Epic AUROC": "AUROC",
                "Epic 95% CI": "95% CI",
                "Morse AUROC": "AUROC",
                "Morse 95% CI": "95% CI",
                "DeLong p": "DeLong p",
                "Note": "Methodological note",
            }
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.body(rows=[0]),
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .tab_source_note(
            "Primary analysis (admission score) shown in bold. "
            "Before-fall scores carry moderate look-ahead risk. "
            "Max/mean scores carry high look-ahead risk as they include post-fall assessments."
        )
        .opt_table_font(font="Arial")
        .opt_row_striping()
        .tab_options(
            table_font_size="11px",
            heading_title_font_size="13px",
            heading_subtitle_font_size="11px",
            source_notes_font_size="9px",
        )
    )
    _gt
    return


@app.cell
def _(Path, etable4_df, mo):
    _out = Path("outputs/tables")
    _out.mkdir(parents=True, exist_ok=True)
    etable4_df.write_csv(_out / "etable4_sensitivity.csv")
    mo.md("**Saved**: `outputs/tables/etable4_sensitivity.csv`")
    return


@app.cell
def _(mo):
    mo.md("""
    ## eTable 10 — Classification Metrics Across Score Timing Strategies

    Full classification metrics (sensitivity, specificity, PPV, NPV, NNE) at
    data-driven and standard thresholds for alternative score timings. This enables
    direct comparison with Epic's published validation, which reported sensitivity
    at specific thresholds using max-score-before-fall.

    **Before-fall scores**: Included only if ≥100 non-fallers have scores; otherwise
    excluded (non-faller comparator is last pre-discharge score, which may not exist).

    **Bias caveats**: Maximum scores include post-fall assessments (high look-ahead bias).
    Mean scores include post-fall assessments (moderate-high look-ahead bias). Thresholds
    are applied to raw scores — no recalibration (admission recalibration model is
    distribution-specific and cannot be reused for max/mean distributions).
    """)
    return


@app.cell
def _(SCORE_TIMING, df, pl):
    from utils.metrics import timing_classification_metrics

    _etable10_rows = []

    # Define timings to evaluate (excluding admission — that's Table 2)
    _TIMINGS = [
        ("before_fall", "Before fall / discharge", "Moderate look-ahead bias"),
        ("max", "Maximum (encounter)", "HIGH look-ahead bias — post-fall scores included"),
        ("mean", "Mean (encounter)", "Moderate-high look-ahead bias — post-fall scores included"),
    ]

    for _key, _label, _note in _TIMINGS:
        _t = SCORE_TIMING[_key]
        _epic_col, _morse_col = _t["epic"], _t["morse"]

        # Complete cases for this timing
        _sub = df.filter(
            pl.col(_epic_col).is_not_null() & pl.col(_morse_col).is_not_null()
        )
        _n = _sub.height
        _n_falls = int(_sub["fall_flag"].sum())

        # Feasibility check for before-fall: need ≥100 non-fallers with scores
        if _key == "before_fall":
            _n_nonfallers = _n - _n_falls
            if _n_nonfallers < 100:
                continue  # Skip — insufficient non-faller comparator

        # Need minimum events for meaningful threshold analysis
        if _n_falls < 20 or (_n - _n_falls) < 20:
            continue

        _y = _sub["fall_flag"].to_numpy()
        _epic = _sub[_epic_col].to_numpy()
        _morse = _sub[_morse_col].to_numpy()

        _etable10_rows.extend(
            timing_classification_metrics(_y, _epic, _morse, _label, _note)
        )

    etable10_df = pl.DataFrame(_etable10_rows)
    return (etable10_df,)


@app.cell
def _(Path, etable10_df, mo):
    _out = Path("outputs/tables")
    _out.mkdir(parents=True, exist_ok=True)
    etable10_df.write_csv(_out / "etable10_timing_classification.csv")

    mo.vstack([
        mo.md("### eTable 10. Classification Metrics Across Score Timing Strategies"),
        mo.ui.table(etable10_df),
        mo.md("**Saved**: `outputs/tables/etable10_timing_classification.csv`"),
    ])
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **Interpretation — eTable 10**

            - **Max-timing vs admission**: At maximum encounter score, Epic's standard ≥35 threshold
              should flag substantially more encounters than at admission (where 97% score <35).
              This demonstrates the threshold calibration gap between continuous monitoring and
              admission screening.

            - **Comparison with Epic model brief**: Epic's validation reported 82–86% sensitivity
              at thresholds 7–21 using max-score-before-fall. Our max-timing MFS ≥45 sensitivity
              can be compared with their reported MFS sensitivity (80.5–87.6% at threshold 45).

            - **Bias warning**: Max and mean scores include post-fall assessments and represent
              an upper bound on achievable performance, not a clinically actionable estimate.
              These results contextualize the primary admission-time analysis, not replace it.

            - **No recalibration applied**: Thresholds are on raw scores. The admission logistic
              recalibration model is distribution-specific and cannot be reused for max/mean
              score distributions.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(Path, etable4_df, lead_time_stats, mo):
    import json

    # Extract key numeric results from the admission row (primary)
    _adm_row = etable4_df.filter(
        etable4_df["Timing strategy"] == "Admission (primary)"
    ).row(0, named=True)

    _results_dict = {
        "primary_epic_auroc": _adm_row["Epic AUROC"],
        "primary_epic_ci": _adm_row["Epic 95% CI"],
        "primary_morse_auroc": _adm_row["Morse AUROC"],
        "primary_morse_ci": _adm_row["Morse 95% CI"],
        "primary_delong_p": _adm_row["DeLong p"],
        "n_encounters": _adm_row["N encounters"],
        "n_falls": _adm_row["N falls"],
        "lead_time": lead_time_stats,
    }

    _out = Path("outputs/tables")
    _out.mkdir(parents=True, exist_ok=True)
    (_out / "sensitivity_key_results.json").write_text(
        json.dumps(_results_dict, indent=2)
    )

    mo.md(
        f"""
        **Saved**: `outputs/tables/sensitivity_key_results.json`

        Primary admission results:
        - Epic AUROC: {_results_dict['primary_epic_auroc']} {_results_dict['primary_epic_ci']}
        - Morse AUROC: {_results_dict['primary_morse_auroc']} {_results_dict['primary_morse_ci']}
        - DeLong p: {_results_dict['primary_delong_p']}
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Sensitivity Analyses Summary

    **Timing hierarchy** (ascending bias risk):

    1. **Admission score** (primary) — No look-ahead bias. First score documented
       after admission. This is the reference analysis for all primary comparisons.

    2. **Before fall / discharge** — Moderate bias risk. For fallers, uses the
       most recent score before the event; for non-fallers, the last pre-discharge
       score. Slightly higher AUROC expected due to temporal proximity to outcome.

    3. **Maximum score** — High bias risk. Includes all scores during the encounter,
       including those documented after the fall event. Represents an upper bound
       on achievable discrimination, not a clinically actionable estimate.

    4. **Mean score** — Moderate-to-high bias risk. Same caveat as maximum.

    5. **First encounter per patient** — Addresses within-patient correlation by
       restricting to one admission per MRN. Provides estimate closer to
       independence assumption; useful for methodological robustness check.

    **DeLong test**: Paired two-sided test (Sun & Xu 2014) for H₀: AUC_Epic = AUC_Morse
    on the same complete-case encounters. p < 0.05 indicates statistically significant
    discrimination difference.
    """)
    return


if __name__ == "__main__":
    app.run()
