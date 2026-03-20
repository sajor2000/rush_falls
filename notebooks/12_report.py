import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, json, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    # Validation of the Epic Predictive Model Fall Risk Score vs the Morse Fall Scale for Inpatient Fall Prediction

    **Study Design**: Single-center retrospective validation (TRIPOD Type 3)
    **Setting**: Rush University Medical Center
    **Target Journal**: JAMA Network Open
    """)
    return


@app.cell
def _(Path, json, mo):
    results_path = Path("outputs/tables/sensitivity_key_results.json")
    key_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    n_enc = key_results.get("n_encounters", 0)
    n_falls = key_results.get("n_falls", 0)
    epic_auroc = key_results.get("primary_epic_auroc", "N/A")
    epic_ci = key_results.get("primary_epic_ci", "N/A")
    morse_auroc = key_results.get("primary_morse_auroc", "N/A")
    morse_ci = key_results.get("primary_morse_ci", "N/A")
    delong_p = key_results.get("primary_delong_p", "N/A")
    _event_rate = f"{n_falls / n_enc * 100:.1f}" if n_enc else "N/A"

    mo.md(
        f"""
        ## Key Findings

        Among **{n_enc:,}** inpatient encounters ({n_falls:,} falls; {_event_rate}% event rate),
        discrimination at admission was:

        | Model | AUROC | 95% CI |
        |---|---|---|
        | Epic PMFRS | {epic_auroc} | {epic_ci} |
        | Morse Fall Scale | {morse_auroc} | {morse_ci} |

        Paired DeLong test: **p {'= ' if not str(delong_p).startswith('<') else ''}{delong_p}**
        """
    )
    return key_results, n_enc, n_falls


@app.cell
def _(Path, mo, pl):
    table1_path = Path("outputs/tables/table1.csv")
    mo.stop(not table1_path.exists(), mo.md("*Table 1 not found — run `03_table1.py` first.*"))
    table1_df = pl.read_csv(table1_path)
    return (table1_df,)


@app.cell
def _(mo, table1_df):
    mo.vstack([
        mo.md("## Table 1 — Patient Characteristics by Fall Status"),
        mo.ui.table(table1_df),
    ])
    return


@app.cell
def _(Path, mo, pl):
    table2_path = Path("outputs/tables/table2.csv")
    mo.stop(not table2_path.exists(), mo.md("*Table 2 not found — run `04_primary_analysis.py` first.*"))
    table2_df = pl.read_csv(table2_path)
    return (table2_df,)


@app.cell
def _(mo, table2_df):
    mo.vstack([
        mo.md("## Table 2 — Model Performance: PMFRS vs MFS"),
        mo.ui.table(table2_df),
    ])
    return


@app.cell
def _(Path, mo, pl):
    table3_path = Path("outputs/tables/table3.csv")
    mo.stop(not table3_path.exists(), mo.md("*Table 3 not found — run `08_reclassification.py` first.*"))
    table3_df = pl.read_csv(table3_path)
    return (table3_df,)


@app.cell
def _(mo, table3_df):
    mo.vstack([
        mo.md("## Table 3 — Reclassification Analysis (NRI, IDI)"),
        mo.ui.table(table3_df),
    ])
    return


@app.cell
def _(Path, mo):
    fig1_path = Path("outputs/figures/figure1_discrimination.png")
    mo.stop(not fig1_path.exists(), mo.md("*Figure 1 not found — run `04_primary_analysis.py` first.*"))
    mo.vstack([
        mo.md("## Figure 1 — Multi-Panel Discrimination"),
        mo.image(src=fig1_path, width=700),
    ])
    return


@app.cell
def _(Path, mo):
    fig2_path = Path("outputs/figures/figure2_dot_plot.png")
    mo.stop(not fig2_path.exists(), mo.md("*Figure 2 not found — run `11_dot_plot.py` first.*"))
    mo.vstack([
        mo.md("## Figure 2 — AUROC Comparison Across Score Timings"),
        mo.image(src=fig2_path, width=700),
    ])
    return


@app.cell
def _(Path, mo):
    fig3_path = Path("outputs/figures/figure3_dca.png")
    mo.stop(not fig3_path.exists(), mo.md("*Figure 3 not found — run `06_decision_curve.py` first.*"))
    mo.vstack([
        mo.md("## Figure 3 — Decision Curve Analysis"),
        mo.image(src=fig3_path, width=700),
    ])
    return


@app.cell
def _(Path, mo):
    efig1_path = Path("outputs/figures/efigure1_calibration_epic.png")
    mo.stop(not efig1_path.exists(), mo.md("*eFigure 1 not found — run `05_calibration.py` first.*"))
    mo.vstack([
        mo.md("## eFigure 1 — Calibration: Epic PMFRS"),
        mo.image(src=efig1_path, width=700),
    ])
    return


@app.cell
def _(Path, mo):
    efig2_path = Path("outputs/figures/efigure2_calibration_morse.png")
    mo.stop(not efig2_path.exists(), mo.md("*eFigure 2 not found — run `05_calibration.py` first.*"))
    mo.vstack([
        mo.md("## eFigure 2 — Calibration: Morse Fall Scale"),
        mo.image(src=efig2_path, width=700),
    ])
    return


@app.cell
def _(Path, mo):
    efig3_path = Path("outputs/figures/efigure3_threshold_overlay.png")
    mo.stop(not efig3_path.exists(), mo.md("*eFigure 3 not found — run `07_threshold_analysis.py` first.*"))
    mo.vstack([
        mo.md("## eFigure 3 — Threshold Comparison Overlay on ROC"),
        mo.image(src=efig3_path, width=700),
    ])
    return


@app.cell
def _(Path, mo):
    efig4_path = Path("outputs/figures/efigure4_cohort_flow.png")
    mo.stop(not efig4_path.exists(), mo.md("*eFigure 4 not found — run `02_cohort_flow.py` first.*"))
    mo.vstack([
        mo.md("## eFigure 4 — CONSORT-Style Cohort Flow Diagram"),
        mo.image(src=efig4_path, width=700),
    ])
    return


@app.cell
def _(Path, mo):
    efig5_path = Path("outputs/figures/efigure5_score_distributions.png")
    mo.stop(not efig5_path.exists(), mo.md("*eFigure 5 not found — run `07_threshold_analysis.py` first.*"))
    mo.vstack([
        mo.md("## eFigure 5 — Score Distributions With Flag Rate Annotations"),
        mo.image(src=efig5_path, width=700),
    ])
    return


@app.cell
def _(Path, mo, pl):
    etable1_path = Path("outputs/tables/etable1_age.csv")
    mo.stop(not etable1_path.exists(), mo.md("*eTable 1 not found — run `09_fairness_audit.py` first.*"))
    etable1_df = pl.read_csv(etable1_path)
    return (etable1_df,)


@app.cell
def _(etable1_df, mo):
    mo.vstack([
        mo.md("## eTable 1 — Stratified Performance by Age Group"),
        mo.ui.table(etable1_df),
    ])
    return


@app.cell
def _(Path, mo, pl):
    etable2_path = Path("outputs/tables/etable2_race.csv")
    mo.stop(not etable2_path.exists(), mo.md("*eTable 2 not found — run `09_fairness_audit.py` first.*"))
    etable2_df = pl.read_csv(etable2_path)
    return (etable2_df,)


@app.cell
def _(etable2_df, mo):
    mo.vstack([
        mo.md("## eTable 2 — Stratified Performance by Race/Ethnicity"),
        mo.ui.table(etable2_df),
    ])
    return


@app.cell
def _(Path, mo, pl):
    etable3_path = Path("outputs/tables/etable3_unit.csv")
    mo.stop(not etable3_path.exists(), mo.md("*eTable 3 not found — run `09_fairness_audit.py` first.*"))
    etable3_df = pl.read_csv(etable3_path)
    return (etable3_df,)


@app.cell
def _(etable3_df, mo):
    mo.vstack([
        mo.md("## eTable 3 — Stratified Performance by Unit Type"),
        mo.ui.table(etable3_df),
    ])
    return


@app.cell
def _(Path, mo, pl):
    etable4_path = Path("outputs/tables/etable4_sensitivity.csv")
    mo.stop(not etable4_path.exists(), mo.md("*eTable 4 not found — run `10_sensitivity_analyses.py` first.*"))
    etable4_sens_df = pl.read_csv(etable4_path)
    return (etable4_sens_df,)


@app.cell
def _(etable4_sens_df, mo):
    mo.vstack([
        mo.md("## eTable 4 — Sensitivity Analyses Summary"),
        mo.ui.table(etable4_sens_df),
    ])
    return


@app.cell
def _(Path, mo, pl):
    etable9_path = Path("outputs/tables/etable9_literature_benchmarking.csv")
    mo.stop(not etable9_path.exists(), mo.md("*eTable 9 not found — run `13_master_report.py` first.*"))
    etable9_df = pl.read_csv(etable9_path)
    return (etable9_df,)


@app.cell
def _(etable9_df, mo):
    mo.vstack([
        mo.md("## eTable 9 — Literature Benchmarking: MFS Validation Studies"),
        mo.ui.table(etable9_df),
    ])
    return


@app.cell
def _(Path, mo, pl):
    etable10_path = Path("outputs/tables/etable10_timing_classification.csv")
    mo.stop(not etable10_path.exists(), mo.md("*eTable 10 not found — run `10_sensitivity_analyses.py` first.*"))
    etable10_df = pl.read_csv(etable10_path)
    return (etable10_df,)


@app.cell
def _(etable10_df, mo):
    mo.vstack([
        mo.md("## eTable 10 — Classification Metrics Across Score Timing Strategies"),
        mo.ui.table(etable10_df),
    ])
    return


@app.cell
def _(key_results, mo, n_enc, n_falls):
    _epic_auroc = key_results.get("primary_epic_auroc", "N/A")
    _morse_auroc = key_results.get("primary_morse_auroc", "N/A")
    _delong_p = key_results.get("primary_delong_p", "N/A")
    _prev_pct = f"{n_falls / n_enc * 100:.1f}" if n_enc else "N/A"

    mo.md(
        f"""
        ## Interpretation

        ### Discrimination
        The Epic PMFRS achieved an AUROC of {_epic_auroc} and the Morse Fall
        Scale an AUROC of {_morse_auroc} for predicting inpatient falls using
        admission scores (paired DeLong p {'= ' if not str(_delong_p).startswith('<') else ''}{_delong_p}).
        These AUROCs are consistent with prospective cohort studies at comparable
        prevalence (Ji 2023: 0.63; Shim 2022: 0.65; Lohse 2021: 0.64; Jiang 2025:
        0.825 at 0.10% prevalence). Higher published MFS AUROCs (0.76–0.86) arise
        from case-control designs or max-score analyses that inflate accuracy
        (Haines 2007). No peer-reviewed publications of the Epic PMFRS exist;
        Epic validation data are from the unpublished model brief. See eTable 9
        for the full literature benchmarking comparison.

        ### Clinical Implications
        At {_prev_pct}% fall prevalence, both models generate substantial false positives.
        Decision curve analysis (Figure 3) should be consulted to determine the
        threshold range where either model provides net benefit over treat-all
        or treat-none strategies.

        ### Sensitivity Analyses
        Discrimination estimates were robust across score timing strategies
        (eTable 4). As expected, maximum and mean encounter scores showed
        inflated AUROCs due to look-ahead bias (inclusion of post-fall assessments).
        The admission score analysis remains the primary, unbiased estimate.

        ### Fairness
        Subgroup analyses (eTables 1-3, 8) should be reviewed for clinically
        meaningful disparities across age, race/ethnicity, gender, and unit type.
        Subgroups with fewer than 20 falls are reported as unreliable per
        pre-specified criteria.

        ### Limitations
        - Single-center retrospective design limits generalizability
        - Treatment paradox: patients flagged as high-risk may have received
          fall prevention interventions, attenuating observed discrimination
        - Score timing for non-fallers' before-fall analysis uses last
          pre-discharge score as comparator
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Methods Summary

    | Parameter | Specification |
    |---|---|
    | Study type | TRIPOD Type 3 (external temporal validation) |
    | Unit of analysis | Encounter (clustered by patient) |
    | Primary predictor timing | First score at admission |
    | Primary comparison | Paired DeLong test (AUROC) |
    | Bootstrap | 2,000 stratified resamples (BCa), seed = 42 |
    | Clustering adjustment | GEE (exchangeable, robust SE) |
    | Calibration | LOWESS (frac = 0.3) with spike/rug plot |
    | Threshold methods | Youden, closest-to-(0,1), fixed sensitivity (60%, 80%), value-optimizing (NMB), DCA-derived |
    | Reclassification | Event NRI and non-event NRI reported separately (Pepe 2015) |
    | Fairness | Stratified by age, race/ethnicity, gender, unit type |
    | Min subgroup events | 20 falls (below = unreliable) |
    | Alpha | 0.05, two-sided |
    | Reporting standard | TRIPOD+AI (Collins et al. BMJ 2024) |
    """)
    return


@app.cell
def _(Path, mo):
    figures_dir = Path("outputs/figures")
    tables_dir = Path("outputs/tables")

    expected_figures = [
        "figure1_discrimination",
        "figure2_dot_plot",
        "figure3_dca",
        "efigure1_calibration_epic",
        "efigure2_calibration_morse",
        "efigure3_threshold_overlay",
        "efigure4_cohort_flow",
        "efigure5_score_distributions",
    ]

    expected_tables = [
        "table1.csv",
        "table2.csv",
        "table3.csv",
        "etable1_age.csv",
        "etable2_race.csv",
        "etable3_unit.csv",
        "etable4_sensitivity.csv",
        "etable8_gender.csv",
        "etable9_literature_benchmarking.csv",
        "etable10_timing_classification.csv",
        "etable5_calibration_summary.csv",
        "dca_threshold_ranges.json",
        "efigure4_cohort_flow_counts.csv",
        "etable7_dca_net_benefit.csv",
        "etable6_threshold_methods.csv",
        "etable6_flag_rate_summary.csv",
        "sensitivity_key_results.json",
    ]

    fig_status = []
    for name in expected_figures:
        pdf_ok = (figures_dir / f"{name}.pdf").exists()
        png_ok = (figures_dir / f"{name}.png").exists()
        pdf_kb = round((figures_dir / f"{name}.pdf").stat().st_size / 1024, 1) if pdf_ok else 0
        png_kb = round((figures_dir / f"{name}.png").stat().st_size / 1024, 1) if png_ok else 0
        fig_status.append(f"| {name} | {'Yes' if pdf_ok else 'No'} | {pdf_kb} KB | {'Yes' if png_ok else 'No'} | {png_kb} KB |")

    tbl_status = []
    for name in expected_tables:
        exists = (tables_dir / name).exists()
        size_kb = round((tables_dir / name).stat().st_size / 1024, 1) if exists else 0
        tbl_status.append(f"| {name} | {'Yes' if exists else 'No'} | {size_kb} KB |")

    fig_lines = "\n".join(fig_status)
    tbl_lines = "\n".join(tbl_status)

    mo.md(
        f"""
        ## Output Inventory

        ### Figures

        | Name | PDF | Size | PNG | Size |
        |---|---|---|---|---|
        {fig_lines}

        ### Tables

        | Name | Exists | Size |
        |---|---|---|
        {tbl_lines}
        """
    )
    return


if __name__ == "__main__":
    app.run()
