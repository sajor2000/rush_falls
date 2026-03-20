import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # 09 — Fairness Audit: Stratified Performance (eTables 1–3, 8)

    **Purpose**: Evaluate Epic PMFRS and Morse Fall Scale discrimination across
    clinically meaningful subgroups to identify potential performance disparities.

    **Output**:
    - eTable 1: AUROC by age group
    - eTable 2: AUROC by race/ethnicity
    - eTable 3: AUROC by unit type (top 10 departments)
    - eTable 8: AUROC by gender

    **Reference**: TRIPOD+AI Item 14e — subgroup performance across key demographics.
    """)
    return


@app.cell
def _():
    from utils.constants import (
        AGE_BINS,
        AGE_LABELS,
        MIN_SUBGROUP_EVENTS,
        N_BOOTSTRAP,
        RANDOM_SEED,
    )
    from utils.metrics import stratified_bootstrap

    return (
        AGE_BINS,
        AGE_LABELS,
        MIN_SUBGROUP_EVENTS,
        N_BOOTSTRAP,
        RANDOM_SEED,
        stratified_bootstrap,
    )


@app.cell
def _(Path, pl):
    df = pl.read_parquet(Path("data/processed/analytic.parquet"))
    return (df,)


@app.cell
def _(df, mo):
    mo.md(
        f"""
        ## Dataset Overview

        - **Encounters**: {df.height:,}
        - **Falls**: {df['fall_flag'].sum():,} ({df['fall_flag'].mean() * 100:.1f}%)
        - **Columns**: {df.width}
        """
    )
    return


@app.cell
def _(AGE_BINS, AGE_LABELS, df, pl):
    # AGE_BINS = [18, 65, 80, inf]; AGE_LABELS = ["18-64","65-79","≥80"]
    # Use pl.when chains for clarity and Polars compatibility
    df_with_age_grp = df.with_columns(
        pl.when(pl.col("age") < AGE_BINS[1])
        .then(pl.lit(AGE_LABELS[0]))
        .when(pl.col("age") < AGE_BINS[2])
        .then(pl.lit(AGE_LABELS[1]))
        .otherwise(pl.lit(AGE_LABELS[2]))
        .alias("age_group")
    )
    return (df_with_age_grp,)


@app.cell
def _(MIN_SUBGROUP_EVENTS, N_BOOTSTRAP, RANDOM_SEED, stratified_bootstrap):
    from sklearn.metrics import roc_auc_score

    def compute_subgroup_row(
        sub_df,
        group_label: str,
        group_value: str,
        epic_col: str = "epic_score_admission",
        morse_col: str = "morse_score_admission",
    ) -> dict:
        """Compute fairness audit row for a single subgroup."""
        n = sub_df.height
        n_falls = int(sub_df["fall_flag"].sum())
        event_rate = n_falls / n if n > 0 else 0.0

        row: dict = {
            "Subgroup": group_label,
            "Category": group_value,
            "N encounters": n,
            "N falls": n_falls,
            "Event rate, %": round(event_rate * 100, 1),
        }

        if n_falls < MIN_SUBGROUP_EVENTS or n - n_falls < MIN_SUBGROUP_EVENTS:
            row["Epic AUROC"] = "—"
            row["Epic 95% CI"] = "—"
            row["Morse AUROC"] = "—"
            row["Morse 95% CI"] = "—"
            row["Note"] = f"<{MIN_SUBGROUP_EVENTS} events — unreliable"
            return row

        y_true = sub_df["fall_flag"].to_numpy()
        epic_scores = sub_df[epic_col].to_numpy()
        morse_scores = sub_df[morse_col].to_numpy()

        # Point estimates
        epic_auc = float(roc_auc_score(y_true, epic_scores))
        morse_auc = float(roc_auc_score(y_true, morse_scores))

        # Paired bootstrap: same resamples for both models
        boot_paired = stratified_bootstrap(
            y_true, epic_scores, pred_b=morse_scores, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED
        )

        epic_lo = boot_paired["auc_a"]["ci_lower"]
        epic_hi = boot_paired["auc_a"]["ci_upper"]
        morse_lo = boot_paired["auc_b"]["ci_lower"]
        morse_hi = boot_paired["auc_b"]["ci_upper"]

        row["Epic AUROC"] = f"{epic_auc:.3f}"
        row["Epic 95% CI"] = f"({epic_lo:.3f}–{epic_hi:.3f})"
        row["Morse AUROC"] = f"{morse_auc:.3f}"
        row["Morse 95% CI"] = f"({morse_lo:.3f}–{morse_hi:.3f})"
        row["Note"] = ""

        return row

    return (compute_subgroup_row,)


@app.cell
def _(AGE_LABELS, compute_subgroup_row, df_with_age_grp, mo, pl):
    _rows_age = []
    for _lbl in AGE_LABELS:
        _sub = df_with_age_grp.filter(pl.col("age_group") == _lbl)
        _rows_age.append(
            compute_subgroup_row(_sub, "Age group", _lbl)
        )

    etable1_df = pl.DataFrame(_rows_age)
    mo.md("## eTable 1 — Stratified Performance by Age Group")
    return (etable1_df,)


@app.cell
def _(etable1_df):
    from great_tables import GT, loc, style

    _gt1 = (
        GT(etable1_df)
        .tab_header(
            title="eTable 1. Model Performance Stratified by Age Group",
            subtitle="Admission scores; 95% CIs from 2000 stratified bootstrap resamples",
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
                "Subgroup": "Stratification",
                "Category": "Subgroup",
                "N encounters": "N encounters",
                "N falls": "N falls",
                "Event rate, %": "Event rate, %",
                "Epic AUROC": "AUROC",
                "Epic 95% CI": "95% CI",
                "Morse AUROC": "AUROC",
                "Morse 95% CI": "95% CI",
                "Note": "Note",
            }
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .sub_missing(missing_text="—")
        .opt_table_font(font="Arial")
        .opt_row_striping()
        .tab_options(
            table_font_size="11px",
            heading_title_font_size="13px",
            heading_subtitle_font_size="11px",
            source_notes_font_size="9px",
        )
    )
    _gt1
    return GT, loc, style


@app.cell
def _(Path, etable1_df, mo):
    _out = Path("outputs/tables")
    _out.mkdir(parents=True, exist_ok=True)
    etable1_df.write_csv(_out / "etable1_age.csv")
    mo.md("**Saved**: `outputs/tables/etable1_age.csv`")
    return


@app.cell
def _(compute_subgroup_row, df, mo, pl):
    _race_cats = (
        df.group_by("race")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)["race"]
        .to_list()
    )

    _rows_race = []
    for _cat in _race_cats:
        if _cat is None:
            continue
        _sub = df.filter(pl.col("race") == _cat)
        _rows_race.append(
            compute_subgroup_row(_sub, "Race/ethnicity", str(_cat))
        )

    etable2_df = pl.DataFrame(_rows_race)
    mo.md("## eTable 2 — Stratified Performance by Race/Ethnicity")
    return (etable2_df,)


@app.cell
def _(GT, etable2_df, loc, style):
    _gt2 = (
        GT(etable2_df)
        .tab_header(
            title="eTable 2. Model Performance Stratified by Race/Ethnicity",
            subtitle="Admission scores; 95% CIs from 2000 stratified bootstrap resamples",
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
            **{
                "Subgroup": "Stratification",
                "Category": "Subgroup",
                "N encounters": "N encounters",
                "N falls": "N falls",
                "Event rate, %": "Event rate, %",
                "Epic AUROC": "AUROC",
                "Epic 95% CI": "95% CI",
                "Morse AUROC": "AUROC",
                "Morse 95% CI": "95% CI",
                "Note": "Note",
            }
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .sub_missing(missing_text="—")
        .opt_table_font(font="Arial")
        .opt_row_striping()
        .tab_options(
            table_font_size="11px",
            heading_title_font_size="13px",
            heading_subtitle_font_size="11px",
            source_notes_font_size="9px",
        )
    )
    _gt2
    return


@app.cell
def _(Path, etable2_df, mo):
    etable2_df.write_csv(Path("outputs/tables/etable2_race.csv"))
    mo.md("**Saved**: `outputs/tables/etable2_race.csv`")
    return


@app.cell
def _(compute_subgroup_row, df, mo, pl):
    # Get top 10 departments by encounter count
    _top10_depts = (
        df.group_by("admitting_department")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(10)["admitting_department"]
        .to_list()
    )

    _rows_unit = []
    for _dept in _top10_depts:
        if _dept is None:
            continue
        _sub = df.filter(pl.col("admitting_department") == _dept)
        _rows_unit.append(
            compute_subgroup_row(_sub, "Admitting department", str(_dept))
        )

    etable3_df = pl.DataFrame(_rows_unit)
    mo.md("## eTable 3 — Stratified Performance by Unit Type (Top 10 Departments)")
    return (etable3_df,)


@app.cell
def _(GT, etable3_df, loc, style):
    _gt3 = (
        GT(etable3_df)
        .tab_header(
            title="eTable 3. Model Performance Stratified by Admitting Department (Top 10)",
            subtitle="Admission scores; 95% CIs from 2000 stratified bootstrap resamples; — indicates <20 falls",
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
            **{
                "Subgroup": "Stratification",
                "Category": "Department",
                "N encounters": "N encounters",
                "N falls": "N falls",
                "Event rate, %": "Event rate, %",
                "Epic AUROC": "AUROC",
                "Epic 95% CI": "95% CI",
                "Morse AUROC": "AUROC",
                "Morse 95% CI": "95% CI",
                "Note": "Note",
            }
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .sub_missing(missing_text="—")
        .opt_table_font(font="Arial")
        .opt_row_striping()
        .tab_options(
            table_font_size="11px",
            heading_title_font_size="13px",
            heading_subtitle_font_size="11px",
            source_notes_font_size="9px",
        )
    )
    _gt3
    return


@app.cell
def _(Path, etable3_df, mo):
    etable3_df.write_csv(Path("outputs/tables/etable3_unit.csv"))
    mo.md("**Saved**: `outputs/tables/etable3_unit.csv`")
    return


@app.cell
def _(compute_subgroup_row, df, mo, pl):
    _gender_cats = (
        df.group_by("gender")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)["gender"]
        .to_list()
    )

    _rows_gender = []
    for _g in _gender_cats:
        if _g is None:
            continue
        _sub = df.filter(pl.col("gender") == _g)
        _rows_gender.append(
            compute_subgroup_row(_sub, "Gender", str(_g))
        )

    etable8_gender_df = pl.DataFrame(_rows_gender)
    mo.md("## eTable 8 — Stratified Performance by Gender")
    return (etable8_gender_df,)


@app.cell
def _(GT, etable8_gender_df, loc, style):
    _gt4 = (
        GT(etable8_gender_df)
        .tab_header(
            title="eTable 8. Model Performance Stratified by Gender",
            subtitle="Admission scores; 95% CIs from 2000 stratified bootstrap resamples",
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
            **{
                "Subgroup": "Stratification",
                "Category": "Subgroup",
                "N encounters": "N encounters",
                "N falls": "N falls",
                "Event rate, %": "Event rate, %",
                "Epic AUROC": "AUROC",
                "Epic 95% CI": "95% CI",
                "Morse AUROC": "AUROC",
                "Morse 95% CI": "95% CI",
                "Note": "Note",
            }
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .sub_missing(missing_text="—")
        .opt_table_font(font="Arial")
        .opt_row_striping()
        .tab_options(
            table_font_size="11px",
            heading_title_font_size="13px",
            heading_subtitle_font_size="11px",
            source_notes_font_size="9px",
        )
    )
    _gt4
    return


@app.cell
def _(Path, etable8_gender_df, mo):
    etable8_gender_df.write_csv(Path("outputs/tables/etable8_gender.csv"))
    mo.md("**Saved**: `outputs/tables/etable8_gender.csv`")
    return


@app.cell
def _(MIN_SUBGROUP_EVENTS, mo):
    mo.md(
        f"""
        ## Fairness Audit Summary

        Performance estimates reported as AUROC (95% CI) from 2000 stratified bootstrap
        resamples (stratified by `fall_flag` to preserve event ratio, seed = 42).

        Subgroups with fewer than {MIN_SUBGROUP_EVENTS} falls are reported as "—" per
        pre-specified reliability threshold (TRIPOD+AI Item 14e).

        **Interpretation caution**: Apparent performance differences across subgroups may
        reflect differences in case mix, documentation practices, or true score properties.
        Causal inference requires prospective study design.
        """
    )
    return


if __name__ == "__main__":
    app.run()
