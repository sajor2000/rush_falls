import marimo

__generated_with = "0.13.0"
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
    mo.md(
        """
        # 01 — Data Discovery & Standardization

        **Purpose**: Ingest raw Excel export, standardize column names, compute derived
        variables, run data quality checks, apply exclusions, and write the analytic
        parquet file used by all downstream notebooks.

        **Input**: `data/raw/output_table_v4.xlsx`
        **Output**: `data/processed/analytic.parquet`
        """
    )
    return


@app.cell
def _():
    from utils.constants import (
        CATEGORICAL_COLS,
        COLUMN_RENAME_MAP,
        DATETIME_COLS,
        ETHNICITY_CLEAN_MAP,
        GENDER_CLEAN_MAP,
        INPATIENT_CODES,
        RACE_CLEAN_MAP,
    )

    return (
        CATEGORICAL_COLS, COLUMN_RENAME_MAP, DATETIME_COLS,
        ETHNICITY_CLEAN_MAP, GENDER_CLEAN_MAP, INPATIENT_CODES, RACE_CLEAN_MAP,
    )


# ── 1. Load raw data ────────────────────────────────────────────────
@app.cell
def _(Path, pl):
    _path = Path("data/raw/output_table_v4.xlsx")
    df_raw = pl.read_excel(_path)
    n_raw = df_raw.height
    return df_raw, n_raw


@app.cell
def _(df_raw, mo, n_raw):
    mo.md(
        f"""
        ## Raw Data Profile

        - **Rows**: {n_raw:,}
        - **Columns**: {df_raw.width}
        - **Column names**: {', '.join(df_raw.columns)}
        """
    )
    return


# ── 2. Rename columns ──────────────────────────────────────────────
@app.cell
def _(COLUMN_RENAME_MAP, df_raw, pl):
    # Build rename map only for columns that actually exist in the data
    _rename = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df_raw.columns}
    df_renamed = df_raw.rename(_rename)

    # Lowercase any remaining columns not in the map
    _remaining = {
        c: c.lower()
        for c in df_renamed.columns
        if c not in _rename.values() and c != c.lower()
    }
    if _remaining:
        df_renamed = df_renamed.rename(_remaining)

    # Parse datetime columns
    for _col in ["admission_date", "discharge_date", "fall_datetime"]:
        if _col in df_renamed.columns:
            _dtype = df_renamed[_col].dtype
            if _dtype == pl.Date:
                df_renamed = df_renamed.with_columns(
                    pl.col(_col).cast(pl.Datetime).alias(_col)
                )
            elif (
                _dtype != pl.Datetime
                and _dtype != pl.Datetime("us")
                and _dtype != pl.Datetime("ns")
                and (_dtype == pl.String or _dtype == pl.Utf8)
            ):
                df_renamed = df_renamed.with_columns(
                    pl.col(_col).str.to_datetime(strict=False).alias(_col)
                )

    return (df_renamed,)


# ── 2b. Data QC: missingness, levels, label cleaning ─────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## Data Quality: Variable Missingness & Levels

        Profile every column for missingness, inspect categorical levels for
        dirty labels, then standardize demographics for publication tables and
        figures.
        """
    )
    return


@app.cell
def _(df_renamed, mo, pl):
    _n = df_renamed.height
    miss_profile_all = (
        pl.DataFrame(
            {
                "column": df_renamed.columns,
                "dtype": [str(df_renamed[c].dtype) for c in df_renamed.columns],
                "n_missing": [df_renamed[c].null_count() for c in df_renamed.columns],
            }
        )
        .with_columns(
            (pl.col("n_missing") / _n * 100).round(3).alias("pct_missing")
        )
        .sort("pct_missing", descending=True)
    )
    mo.md(
        f"### Variable Missingness Profile\n\n"
        f"All {df_renamed.width} columns, {_n:,} rows. "
        f"Sorted by percent missing (descending)."
    )
    return (miss_profile_all,)


@app.cell
def _(miss_profile_all, mo):
    mo.ui.table(miss_profile_all)
    return


@app.cell
def _(CATEGORICAL_COLS, df_renamed, mo, pl):
    _tabs = {}
    for _col in CATEGORICAL_COLS:
        if _col not in df_renamed.columns:
            continue
        _vc = (
            df_renamed.group_by(_col)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        _tabs[_col] = mo.ui.table(_vc)

    mo.vstack([
        mo.md("### Categorical Variable Levels (Pre-Cleaning)"),
        mo.ui.tabs(_tabs),
    ])
    return


@app.cell
def _(ETHNICITY_CLEAN_MAP, GENDER_CLEAN_MAP, RACE_CLEAN_MAP, df_renamed, mo, pl):
    # Strip whitespace from all string columns; empty strings → null
    _str_cols = [
        c for c in df_renamed.columns if df_renamed[c].dtype in (pl.String, pl.Utf8)
    ]
    _df = df_renamed.with_columns(
        [pl.col(c).str.strip_chars().alias(c) for c in _str_cols]
    ).with_columns(
        [
            pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c)
            for c in _str_cols
        ]
    )

    # Apply label standardization
    _remap_log = []
    if "race" in _df.columns:
        _before = _df["race"].drop_nulls().n_unique()
        _df = _df.with_columns(
            pl.col("race")
            .replace_strict(RACE_CLEAN_MAP, default="Other/Unknown")
            .alias("race")
        )
        _after = _df["race"].drop_nulls().n_unique()
        _remap_log.append(f"**race**: {_before} raw → {_after} standardized levels")

    if "ethnicity" in _df.columns:
        _before = _df["ethnicity"].drop_nulls().n_unique()
        _df = _df.with_columns(
            pl.col("ethnicity")
            .replace_strict(ETHNICITY_CLEAN_MAP, default="Unknown")
            .alias("ethnicity")
        )
        _after = _df["ethnicity"].drop_nulls().n_unique()
        _remap_log.append(
            f"**ethnicity**: {_before} raw → {_after} standardized levels"
        )

    if "gender" in _df.columns:
        _before = _df["gender"].drop_nulls().n_unique()
        _df = _df.with_columns(
            pl.col("gender")
            .replace_strict(GENDER_CLEAN_MAP, default="Unknown")
            .alias("gender")
        )
        _after = _df["gender"].drop_nulls().n_unique()
        _remap_log.append(
            f"**gender**: {_before} raw → {_after} standardized levels"
        )

    df = _df

    mo.md(
        "### Label Cleaning Applied\n\n"
        + "\n".join(f"- {line}" for line in _remap_log)
        + "\n\nAll string columns stripped of whitespace. Empty strings → null."
    )
    return (df,)


# ── 3. Derive variables ────────────────────────────────────────────
@app.cell
def _(df, pl):
    # Replace literal "NULL" strings with actual nulls in unit_fall_occurred
    _df = df.with_columns(
        pl.when(pl.col("unit_fall_occurred").str.to_uppercase() == "NULL")
        .then(None)
        .otherwise(pl.col("unit_fall_occurred"))
        .alias("unit_fall_occurred")
    )
    df_derived = _df.with_columns(
        [
            # fall_flag: 1 if fall_datetime OR unit_fall_occurred is not null
            pl.when(
                pl.col("fall_datetime").is_not_null()
                | pl.col("unit_fall_occurred").is_not_null()
            )
            .then(1)
            .otherwise(0)
            .alias("fall_flag"),
            # los_days: fractional days
            (
                (pl.col("discharge_date") - pl.col("admission_date")).dt.total_seconds()
                / 86400
            ).alias("los_days"),
        ]
    )
    return (df_derived,)


@app.cell
def _(df_derived, mo, pl):
    _n_total = df_derived.height
    _n_falls = df_derived.filter(pl.col("fall_flag") == 1).height
    _n_patients = df_derived["mrn"].n_unique()
    _n_fall_patients = (
        df_derived.filter(pl.col("fall_flag") == 1)["mrn"].n_unique()
    )

    mo.md(
        f"""
        ## Derived Variables Summary

        | Metric | Value |
        |---|---|
        | Total encounters | {_n_total:,} |
        | Fall encounters | {_n_falls:,} ({_n_falls / _n_total * 100:.1f}%) |
        | Unique patients | {_n_patients:,} |
        | Unique fall patients | {_n_fall_patients:,} |
        | Fall rate | {_n_falls / _n_total * 100:.2f}% |
        """
    )
    return


# ── 4. Data quality checks ─────────────────────────────────────────
@app.cell
def _(df_derived, mo, pl):
    _checks = []

    # 4a. Duplicate encounter check
    _n_unique_csn = df_derived["encounter_csn"].n_unique()
    _n_rows = df_derived.height
    _dupes = _n_rows - _n_unique_csn
    _checks.append(
        f"{'PASS' if _dupes == 0 else 'FAIL'}: Duplicate encounter_csn = {_dupes}"
    )

    # 4b. Fall datetime within admission window
    _fallers = df_derived.filter(pl.col("fall_datetime").is_not_null())
    _bad_timing = _fallers.filter(
        (pl.col("fall_datetime") < pl.col("admission_date"))
        | (pl.col("fall_datetime") > pl.col("discharge_date"))
    ).height
    _checks.append(
        f"{'PASS' if _bad_timing == 0 else 'WARN'}: "
        f"Falls outside admission window = {_bad_timing}"
    )

    # 4c. Age range
    _min_age = df_derived["age"].min()
    _max_age = df_derived["age"].max()
    _under_18 = df_derived.filter(pl.col("age") < 18).height
    _checks.append(
        f"{'PASS' if _under_18 == 0 else 'WARN'}: "
        f"Age range [{_min_age}, {_max_age}], under 18 = {_under_18}"
    )

    # 4d. Epic max >= admission sanity
    _bad_max = df_derived.filter(
        pl.col("epic_score_max").is_not_null()
        & pl.col("epic_score_admission").is_not_null()
        & (pl.col("epic_score_max") < pl.col("epic_score_admission"))
    ).height
    _checks.append(
        f"{'PASS' if _bad_max == 0 else 'WARN'}: "
        f"epic_score_max < epic_score_admission = {_bad_max}"
    )

    # 4e. Accommodation code profile
    _accom_vc = (
        df_derived.group_by("accommodation_code")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    mo.md(
        "## Data Quality Checks\n\n"
        + "\n".join(f"- {c}" for c in _checks)
        + "\n\n### Accommodation Code Distribution\n"
    )
    return


@app.cell
def _(df_derived, mo, pl):
    _accom = (
        df_derived.group_by("accommodation_code")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("fall_flag").sum().alias("n_falls"),
            ]
        )
        .sort("n", descending=True)
    )
    mo.ui.table(_accom)
    return


# ── 5. Missing data characterization ───────────────────────────────
@app.cell
def _(df_derived, mo, pl):
    _score_cols = [
        "epic_score_admission",
        "morse_score_admission",
        "epic_score_max",
        "epic_score_mean",
        "morse_score_max",
        "morse_score_mean",
    ]

    _missing_rows = []
    for _col in _score_cols:
        if _col in df_derived.columns:
            _n_miss = df_derived.filter(pl.col(_col).is_null()).height
            _n_miss_fall = df_derived.filter(
                pl.col(_col).is_null() & (pl.col("fall_flag") == 1)
            ).height
            _missing_rows.append(
                {
                    "column": _col,
                    "n_missing": _n_miss,
                    "pct_missing": round(_n_miss / df_derived.height * 100, 3),
                    "n_missing_fallers": _n_miss_fall,
                }
            )

    missing_profile = pl.DataFrame(_missing_rows)
    mo.md("## Missing Data Profile")
    return (missing_profile,)


@app.cell
def _(missing_profile, mo):
    mo.ui.table(missing_profile)
    return


# ── 6. Apply exclusions ────────────────────────────────────────────
@app.cell
def _(INPATIENT_CODES, df_derived, mo, pl):
    _flow = []
    _df = df_derived
    _flow.append(f"Starting encounters: {_df.height:,}")

    # 6a. Exclude age < 18
    _excluded = _df.filter(pl.col("age") < 18).height
    _df = _df.filter(pl.col("age") >= 18)
    _flow.append(f"Excluded age < 18: {_excluded:,} → remaining: {_df.height:,}")

    # 6b. Exclude non-inpatient
    _excluded = _df.filter(~pl.col("accommodation_code").is_in(INPATIENT_CODES)).height
    _df = _df.filter(pl.col("accommodation_code").is_in(INPATIENT_CODES))
    _flow.append(
        f"Excluded non-inpatient (accommodation_code not in {INPATIENT_CODES}): "
        f"{_excluded:,} → remaining: {_df.height:,}"
    )

    # 6c. Exclude missing discharge_department
    _excluded = _df.filter(pl.col("discharge_department").is_null()).height
    _df = _df.filter(pl.col("discharge_department").is_not_null())
    _flow.append(
        f"Excluded missing discharge_department: {_excluded:,} → remaining: {_df.height:,}"
    )

    df_eligible = _df

    # 6d. Flag missing scores (do not exclude yet)
    _miss_epic = df_eligible.filter(pl.col("epic_score_admission").is_null()).height
    _miss_morse = df_eligible.filter(pl.col("morse_score_admission").is_null()).height
    _miss_epic_fall = df_eligible.filter(
        pl.col("epic_score_admission").is_null() & (pl.col("fall_flag") == 1)
    ).height
    _miss_morse_fall = df_eligible.filter(
        pl.col("morse_score_admission").is_null() & (pl.col("fall_flag") == 1)
    ).height

    _flow.append(
        f"Missing Epic admission score: {_miss_epic:,} ({_miss_epic_fall} fallers)"
    )
    _flow.append(
        f"Missing Morse admission score: {_miss_morse:,} ({_miss_morse_fall} fallers)"
    )

    mo.md("## Cohort Flow (Exclusions)\n\n" + "\n".join(f"- {f}" for f in _flow))
    return (df_eligible,)


# ── 7. Define analytic cohorts ──────────────────────────────────────
@app.cell
def _(df_eligible, mo, pl):
    # Primary: both scores present
    df_analytic = df_eligible.filter(
        pl.col("epic_score_admission").is_not_null()
        & pl.col("morse_score_admission").is_not_null()
    )

    _n = df_analytic.height
    _n_falls = df_analytic.filter(pl.col("fall_flag") == 1).height
    _n_patients = df_analytic["mrn"].n_unique()

    mo.md(
        f"""
        ## Analytic Cohort (Complete Case)

        - **Encounters**: {_n:,}
        - **Falls**: {_n_falls:,} ({_n_falls / _n * 100:.1f}%)
        - **Unique patients**: {_n_patients:,}
        """
    )
    return (df_analytic,)


# ── 8. Score distributions ──────────────────────────────────────────
@app.cell
def _(df_analytic, mo, pl):
    _desc = df_analytic.select(
        [
            pl.col("epic_score_admission").mean().alias("epic_adm_mean"),
            pl.col("epic_score_admission").std().alias("epic_adm_sd"),
            pl.col("epic_score_admission").median().alias("epic_adm_median"),
            pl.col("morse_score_admission").mean().alias("morse_adm_mean"),
            pl.col("morse_score_admission").std().alias("morse_adm_sd"),
            pl.col("morse_score_admission").median().alias("morse_adm_median"),
            pl.col("epic_score_max").mean().alias("epic_max_mean"),
            pl.col("morse_score_max").mean().alias("morse_max_mean"),
        ]
    ).row(0, named=True)

    mo.md(
        f"""
        ## Score Distributions (Analytic Cohort)

        | Score | Mean ± SD | Median |
        |---|---|---|
        | Epic admission | {_desc['epic_adm_mean']:.2f} ± {_desc['epic_adm_sd']:.2f} | {_desc['epic_adm_median']:.1f} |
        | Morse admission | {_desc['morse_adm_mean']:.2f} ± {_desc['morse_adm_sd']:.2f} | {_desc['morse_adm_median']:.1f} |
        | Epic max | {_desc['epic_max_mean']:.2f} | — |
        | Morse max | {_desc['morse_max_mean']:.2f} | — |
        """
    )
    return


# ── 9. Write analytic parquet ───────────────────────────────────────
@app.cell
def _(Path, df_analytic, mo):
    _out = Path("data/processed")
    _out.mkdir(parents=True, exist_ok=True)
    df_analytic.write_parquet(_out / "analytic.parquet")
    mo.md(
        f"**Saved**: `data/processed/analytic.parquet` "
        f"({df_analytic.height:,} rows, {df_analytic.width} columns)"
    )
    return


if __name__ == "__main__":
    app.run()
