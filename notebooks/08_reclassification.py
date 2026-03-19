"""
08_reclassification.py — Reclassification Analysis (Table 3)

Computes NRI (continuous + categorical) and IDI to evaluate whether Epic PMFRS
reclassifies patients better than Morse Fall Scale.

Reference model = Morse Fall Scale
New model       = Epic PMFRS
Threshold       = Youden index from Morse model (for categorical NRI)

Inputs:  data/processed/analytic.parquet
Outputs: outputs/tables/table3.csv
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
# Table 3. Reclassification Analysis

Net reclassification improvement (NRI) and integrated discrimination
improvement (IDI) comparing Epic PMFRS (new model) vs Morse Fall Scale
(reference model). Event NRI and non-event NRI reported separately
(Pepe MS et al. *Stat Med* 2015;34:110–128).

**Threshold for categorical NRI**: Youden index from Morse Fall Scale.
**Bootstrap CIs**: 2000 stratified replicates, seed 42.
"""
    )
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import polars as pl
    return Path, np, pl


@app.cell
def _():
    from utils.constants import (
        ALPHA,
        EPIC_2TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        MFS_HIGH,
        MFS_MODERATE,
        N_BOOTSTRAP,
        RANDOM_SEED,
    )
    from utils.metrics import (
        compute_categorical_nri,
        compute_nri_idi,
        logistic_recalibration,
        youden_threshold,
    )
    return (
        ALPHA,
        EPIC_2TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        MFS_HIGH,
        MFS_MODERATE,
        N_BOOTSTRAP,
        RANDOM_SEED,
        compute_categorical_nri,
        compute_nri_idi,
        logistic_recalibration,
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
    epic_scores = df["epic_score_admission"].to_numpy()
    morse_scores = df["morse_score_admission"].to_numpy()
    return epic_scores, morse_scores, y_true


# ── Logistic recalibration ────────────────────────────────────────────
@app.cell
def _(EPIC_2TIER_HIGH, EPIC_3TIER_MEDIUM, MFS_HIGH, MFS_MODERATE, epic_scores, logistic_recalibration, morse_scores, y_true):
    epic_prob, _epic_lr = logistic_recalibration(epic_scores, y_true)
    morse_prob, _morse_lr = logistic_recalibration(morse_scores, y_true)

    # Probability equivalents for standard cutoffs
    # Epic ≥70 omitted: too few encounters score ≥70 at admission for meaningful NRI
    morse_prob_at_25 = float(_morse_lr.predict_proba([[float(MFS_MODERATE)]])[0, 1])
    morse_prob_at_45 = float(_morse_lr.predict_proba([[float(MFS_HIGH)]])[0, 1])
    epic_prob_at_35 = float(_epic_lr.predict_proba([[float(EPIC_3TIER_MEDIUM)]])[0, 1])
    epic_prob_at_50 = float(_epic_lr.predict_proba([[float(EPIC_2TIER_HIGH)]])[0, 1])
    return epic_prob, epic_prob_at_35, epic_prob_at_50, morse_prob, morse_prob_at_25, morse_prob_at_45


# ── Youden threshold from Morse (reference model) ────────────────────
@app.cell
def _(epic_prob, epic_prob_at_35, epic_prob_at_50, epic_scores, morse_prob, morse_prob_at_25, morse_prob_at_45, mo, np, y_true, youden_threshold):
    morse_youden = youden_threshold(y_true, morse_prob)
    epic_youden = youden_threshold(y_true, epic_prob)
    pct_ge70 = float(np.sum(epic_scores >= 70) / len(epic_scores) * 100)
    mo.md(
        f"""
        ### Categorical NRI Thresholds

        | Method | Probability threshold |
        |---|---|
        | **Youden index (Epic recalibrated)** | **{epic_youden:.4f}** |
        | **Youden index (Morse recalibrated)** | **{morse_youden:.4f}** |
        | MFS ≥25 probability equivalent | {morse_prob_at_25:.4f} |
        | MFS ≥45 probability equivalent | {morse_prob_at_45:.4f} |
        | Epic ≥35 probability equivalent | {epic_prob_at_35:.4f} |
        | Epic ≥50 probability equivalent | {epic_prob_at_50:.4f} |

        These thresholds convert models' probabilities to binary
        high/low-risk classifications for categorical NRI calculations.

        **Note**: Epic ≥70 omitted — at admission only {pct_ge70:.1f}% of encounters
        score ≥70, making categorical NRI at this cutoff uninformative.
        """
    )
    return epic_youden, morse_youden, pct_ge70


# ── Point estimates ───────────────────────────────────────────────────
@app.cell
def _(compute_categorical_nri, compute_nri_idi, epic_prob, epic_prob_at_35, epic_prob_at_50, epic_youden, mo, morse_prob, morse_prob_at_25, morse_prob_at_45, morse_youden, y_true):
    point_estimates = compute_nri_idi(
        y_true=y_true,
        prob_ref=morse_prob,
        prob_new=epic_prob,
        threshold=morse_youden,
    )

    # Categorical NRI at all standard + data-derived cutoffs
    pe_nri_cat_epic_youden = compute_categorical_nri(y_true, morse_prob, epic_prob, epic_youden)
    pe_nri_cat_mfs25 = compute_categorical_nri(y_true, morse_prob, epic_prob, morse_prob_at_25)
    pe_nri_cat_mfs45 = compute_categorical_nri(y_true, morse_prob, epic_prob, morse_prob_at_45)
    pe_nri_cat_epic35 = compute_categorical_nri(y_true, morse_prob, epic_prob, epic_prob_at_35)
    pe_nri_cat_epic50 = compute_categorical_nri(y_true, morse_prob, epic_prob, epic_prob_at_50)

    mo.md(
        f"""
        ### Point Estimates (no CI)

        | Metric | Value |
        |---|---|
        | Continuous NRI (total) | {point_estimates['nri_continuous']:.4f} |
        | NRI events | {point_estimates['nri_events']:.4f} |
        | NRI non-events | {point_estimates['nri_nonevents']:.4f} |
        | Categorical NRI — Morse Youden | {point_estimates['nri_categorical']:.4f} |
        | Categorical NRI — Epic Youden | {pe_nri_cat_epic_youden:.4f} |
        | Categorical NRI — MFS ≥25 | {pe_nri_cat_mfs25:.4f} |
        | Categorical NRI — MFS ≥45 | {pe_nri_cat_mfs45:.4f} |
        | Categorical NRI — Epic ≥35 | {pe_nri_cat_epic35:.4f} |
        | Categorical NRI — Epic ≥50 | {pe_nri_cat_epic50:.4f} |
        | IDI (total) | {point_estimates['idi']:.4f} |
        | IDI events | {point_estimates['idi_events']:.4f} |
        | IDI non-events | {point_estimates['idi_nonevents']:.4f} |
        """
    )
    return pe_nri_cat_epic35, pe_nri_cat_epic50, pe_nri_cat_epic_youden, pe_nri_cat_mfs25, pe_nri_cat_mfs45, point_estimates


# ── Bootstrap CIs (stratified, seed=42, 2000 replicates) ─────────────
@app.cell
def _(ALPHA, N_BOOTSTRAP, RANDOM_SEED, compute_categorical_nri, compute_nri_idi, epic_prob, epic_prob_at_35, epic_prob_at_50, epic_youden, morse_prob, morse_prob_at_25, morse_prob_at_45, morse_youden, np, y_true):
    """
    Stratified bootstrap: preserve fall_flag ratio in each resample.
    For each replicate, compute NRI/IDI on the bootstrap sample.
    """
    _rng = np.random.RandomState(RANDOM_SEED)
    _events_idx = np.where(y_true == 1)[0]
    _nonevents_idx = np.where(y_true == 0)[0]

    _boot_keys = [
        "nri_continuous",
        "nri_events",
        "nri_nonevents",
        "nri_categorical",
        "idi",
        "idi_events",
        "idi_nonevents",
    ]
    _boot_results: dict[str, list[float]] = {k: [] for k in _boot_keys}
    _boot_nri_cat_mfs45: list[float] = []
    _boot_nri_cat_epic_youden: list[float] = []
    _boot_nri_cat_mfs25: list[float] = []
    _boot_nri_cat_epic35: list[float] = []
    _boot_nri_cat_epic50: list[float] = []

    for _ in range(N_BOOTSTRAP):
        _be = _rng.choice(_events_idx, size=len(_events_idx), replace=True)
        _bn = _rng.choice(_nonevents_idx, size=len(_nonevents_idx), replace=True)
        _idx = np.concatenate([_be, _bn])

        _y_b = y_true[_idx]
        _ref_b = morse_prob[_idx]
        _new_b = epic_prob[_idx]

        # Categorical NRI at Morse Youden threshold (fixed from full sample)
        _est = compute_nri_idi(
            y_true=_y_b,
            prob_ref=_ref_b,
            prob_new=_new_b,
            threshold=morse_youden,
        )
        for _k in _boot_keys:
            _v = _est[_k]
            if _v is not None:
                _boot_results[_k].append(float(_v))

        # Categorical NRI at additional thresholds (lightweight)
        _boot_nri_cat_mfs45.append(
            compute_categorical_nri(_y_b, _ref_b, _new_b, morse_prob_at_45)
        )
        _boot_nri_cat_epic_youden.append(
            compute_categorical_nri(_y_b, _ref_b, _new_b, epic_youden)
        )
        _boot_nri_cat_mfs25.append(
            compute_categorical_nri(_y_b, _ref_b, _new_b, morse_prob_at_25)
        )
        _boot_nri_cat_epic35.append(
            compute_categorical_nri(_y_b, _ref_b, _new_b, epic_prob_at_35)
        )
        _boot_nri_cat_epic50.append(
            compute_categorical_nri(_y_b, _ref_b, _new_b, epic_prob_at_50)
        )

    # Compute percentile CIs
    _lo_pct = 100 * ALPHA / 2
    _hi_pct = 100 * (1 - ALPHA / 2)
    boot_ci: dict[str, dict[str, float]] = {}
    for _k, _vals in _boot_results.items():
        _arr = np.array(_vals)
        boot_ci[_k] = {
            "estimate": float(np.mean(_arr)),
            "ci_lower": float(np.percentile(_arr, _lo_pct)),
            "ci_upper": float(np.percentile(_arr, _hi_pct)),
        }

    # CIs for additional categorical NRI thresholds
    for _label, _boot_list in [
        ("nri_categorical_mfs45", _boot_nri_cat_mfs45),
        ("nri_categorical_epic_youden", _boot_nri_cat_epic_youden),
        ("nri_categorical_mfs25", _boot_nri_cat_mfs25),
        ("nri_categorical_epic35", _boot_nri_cat_epic35),
        ("nri_categorical_epic50", _boot_nri_cat_epic50),
    ]:
        _arr = np.array(_boot_list)
        boot_ci[_label] = {
            "estimate": float(np.mean(_arr)),
            "ci_lower": float(np.percentile(_arr, _lo_pct)),
            "ci_upper": float(np.percentile(_arr, _hi_pct)),
        }

    return (boot_ci,)


# ── Assemble Table 3 ──────────────────────────────────────────────────
@app.cell
def _(boot_ci, pe_nri_cat_epic35, pe_nri_cat_epic50, pe_nri_cat_epic_youden, pe_nri_cat_mfs25, pe_nri_cat_mfs45, pl, point_estimates):
    # (section, point_estimate_value, boot_key, display_label)
    _sections = [
        # Continuous NRI
        ("Continuous NRI", point_estimates["nri_events"], "nri_events", "  Events (upward reclassification \u2212 downward)"),
        ("Continuous NRI", point_estimates["nri_nonevents"], "nri_nonevents", "  Non-events (downward reclassification \u2212 upward)"),
        ("Continuous NRI", point_estimates["nri_continuous"], "nri_continuous", "  Total continuous NRI"),
        # Categorical NRI — data-derived thresholds
        ("Categorical NRI", point_estimates["nri_categorical"], "nri_categorical", "  Total categorical NRI (Morse Youden threshold)"),
        ("Categorical NRI", pe_nri_cat_epic_youden, "nri_categorical_epic_youden", "  Total categorical NRI (Epic Youden threshold)"),
        # Categorical NRI — standard clinical cutoffs
        ("Categorical NRI", pe_nri_cat_mfs25, "nri_categorical_mfs25", "  Total categorical NRI (MFS \u226525 threshold)"),
        ("Categorical NRI", pe_nri_cat_mfs45, "nri_categorical_mfs45", "  Total categorical NRI (MFS \u226545 threshold)"),
        ("Categorical NRI", pe_nri_cat_epic35, "nri_categorical_epic35", "  Total categorical NRI (Epic \u226535 threshold)"),
        ("Categorical NRI", pe_nri_cat_epic50, "nri_categorical_epic50", "  Total categorical NRI (Epic \u226550 threshold)"),
        # IDI
        ("IDI", point_estimates["idi_events"], "idi_events", "  Events (mean new prob \u2212 mean ref prob)"),
        ("IDI", point_estimates["idi_nonevents"], "idi_nonevents", "  Non-events (mean new prob \u2212 mean ref prob)"),
        ("IDI", point_estimates["idi"], "idi", "  Total IDI"),
    ]

    _rows = []
    for _section, _pe_val, _boot_key, _label in _sections:
        _ci = boot_ci[_boot_key]
        _est_str = f"{_pe_val:.3f}" if _pe_val is not None else "\u2014"
        _ci_str = f"{_ci['ci_lower']:.3f} to {_ci['ci_upper']:.3f}"
        _rows.append(
            {
                "Section": _section,
                "Metric": _label,
                "Estimate": _est_str,
                "95% Bootstrap CI": _ci_str,
            }
        )

    table3 = pl.DataFrame(_rows)
    return (table3,)


@app.cell
def _(mo, table3):
    mo.md(
r"""
## Table 3. Reclassification Analysis: Epic PMFRS vs Morse Fall Scale
"""
    )
    return


@app.cell
def _(mo, table3):
    mo.ui.table(table3)
    return


# ── great-tables rendering ────────────────────────────────────────────
@app.cell
def _(mo, pct_ge70, table3):
    try:
        from great_tables import GT, loc, style

        _gt = (
            GT(table3.to_pandas(), rowname_col="Metric", groupname_col="Section")
            .tab_header(
                title="Table 3. Reclassification Analysis",
                subtitle="Epic PMFRS (new) vs Morse Fall Scale (reference); 2000 stratified bootstrap replicates",
            )
            .tab_spanner(label="Epic vs Morse", columns=["Estimate", "95% Bootstrap CI"])
            .cols_label(
                Estimate="Estimate",
            )
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
                "Direction: Epic PMFRS (new) vs Morse Fall Scale (reference). "
                "Positive values indicate Epic reclassifies better than Morse. "
                f"Epic \u226570 omitted: at admission only {pct_ge70:.1f}% of encounters score \u226570, "
                "making categorical NRI at this cutoff uninformative."
            )
        )
        gt_table = _gt
    except Exception as _e:
        gt_table = mo.md(f"*great-tables rendering unavailable: {_e}*")

    gt_table
    return (gt_table,)


# ── Export CSV ────────────────────────────────────────────────────────
@app.cell
def _(Path, mo, table3):
    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _out_path = _out_dir / "table3.csv"
    table3.write_csv(_out_path)
    mo.md(f"**Saved**: `{_out_path}` ({table3.height} rows)")
    return


if __name__ == "__main__":
    app.run()
