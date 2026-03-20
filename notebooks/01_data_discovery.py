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
    # 01 — Data Discovery & Standardization

    **Purpose**: Ingest raw Excel export, standardize column names, compute derived
    variables, run data quality checks, apply exclusions, and write the analytic
    parquet file used by all downstream notebooks.

    **Input**: `data/raw/output_table_v4.xlsx`
    **Output**: `data/processed/analytic.parquet`
    """)
    return


@app.cell
def _():
    from utils.constants import (
        CATEGORICAL_COLS,
        COLUMN_RENAME_MAP,
        ETHNICITY_CLEAN_MAP,
        GENDER_CLEAN_MAP,
        RACE_CLEAN_MAP,
    )

    return (
        CATEGORICAL_COLS,
        COLUMN_RENAME_MAP,
        ETHNICITY_CLEAN_MAP,
        GENDER_CLEAN_MAP,
        RACE_CLEAN_MAP,
    )


@app.cell
def _():
    import matplotlib
    import matplotlib.pyplot as plt

    from utils.plotting import FIG_DOUBLE_COL, FIG_SINGLE_COL, JAMA_STYLE, save_figure

    matplotlib.rcParams.update(JAMA_STYLE)
    return FIG_DOUBLE_COL, FIG_SINGLE_COL, plt, save_figure


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


@app.cell
def _(mo):
    mo.md("""
    ## Data Quality: Variable Missingness & Levels

    Profile every column for missingness, inspect categorical levels for
    dirty labels, then standardize demographics for publication tables and
    figures.
    """)
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
def _(
    ETHNICITY_CLEAN_MAP,
    GENDER_CLEAN_MAP,
    RACE_CLEAN_MAP,
    df_renamed,
    mo,
    pl,
):
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


@app.cell
def _(df_derived, mo, pl):
    _flow = []
    _df = df_derived
    _flow.append(f"Starting encounters: {_df.height:,}")

    # 6a. Exclude age < 18
    _excluded = _df.filter(pl.col("age") < 18).height
    _df = _df.filter(pl.col("age") >= 18)
    _flow.append(f"Excluded age < 18: {_excluded:,} → remaining: {_df.height:,}")

    # 6b. Exclude missing discharge_department
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


@app.cell
def _(df_analytic, mo, pl):
    _adm_min = df_analytic["admission_date"].min()
    _adm_max = df_analytic["admission_date"].max()
    _dc_max = df_analytic["discharge_date"].max()
    _fall_dates = df_analytic.filter(pl.col("fall_datetime").is_not_null())["fall_datetime"]
    _fall_min = _fall_dates.min()
    _fall_max = _fall_dates.max()
    _duration_days = (_dc_max - _adm_min).total_seconds() / 86400
    _duration_months = _duration_days / 30.44

    mo.md(
        f"""
        ## Study Period Summary (TRIPOD+AI Item 5b)

        | Parameter | Value |
        |---|---|
        | Earliest admission | {_adm_min.strftime('%Y-%m-%d')} |
        | Latest admission | {_adm_max.strftime('%Y-%m-%d')} |
        | Latest discharge | {_dc_max.strftime('%Y-%m-%d')} |
        | Earliest fall | {_fall_min.strftime('%Y-%m-%d')} |
        | Latest fall | {_fall_max.strftime('%Y-%m-%d')} |
        | Study duration | {_duration_days:.0f} days ({_duration_months:.1f} months) |
        """
    )
    return


@app.cell
def _(df_analytic, mo, np, pl):
    _fallers = df_analytic.filter(pl.col("fall_flag") == 1)
    _n_fallers = _fallers.height

    _age = _fallers["age"].to_numpy()
    _age_mean = np.mean(_age)
    _age_sd = np.std(_age, ddof=1)
    _age_med = np.median(_age)
    _age_q1, _age_q3 = np.percentile(_age, [25, 75])

    _gender_vc = (
        _fallers.group_by("gender").agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / _n_fallers * 100).round(1).alias("pct"))
        .sort("n", descending=True)
    )
    _race_vc = (
        _fallers.group_by("race").agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / _n_fallers * 100).round(1).alias("pct"))
        .sort("n", descending=True)
    )
    _eth_vc = (
        _fallers.group_by("ethnicity").agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / _n_fallers * 100).round(1).alias("pct"))
        .sort("n", descending=True)
    )

    mo.vstack([
        mo.md(
            f"""
            ## Faller Demographics Profile (n = {_n_fallers:,})

            **Age**: mean {_age_mean:.1f} ± {_age_sd:.1f}, median {_age_med:.0f} [{_age_q1:.0f}–{_age_q3:.0f}]
            """
        ),
        mo.md("### Gender"),
        mo.ui.table(_gender_vc),
        mo.md("### Race"),
        mo.ui.table(_race_vc),
        mo.md("### Ethnicity"),
        mo.ui.table(_eth_vc),
    ])
    return


@app.cell
def _(FIG_DOUBLE_COL, df_analytic, mo, pl, plt, save_figure):
    _fall_units = (
        df_analytic.filter(pl.col("unit_fall_occurred").is_not_null())
        .group_by("unit_fall_occurred")
        .agg(pl.len().alias("n_falls"))
        .sort("n_falls", descending=True)
        .head(15)
    )

    mo.vstack([
        mo.md("## Where Falls Occur (Top 15 Units)"),
        mo.ui.table(_fall_units),
    ])

    _units = _fall_units["unit_fall_occurred"].to_list()[::-1]
    _counts = _fall_units["n_falls"].to_list()[::-1]

    _fig, _ax = plt.subplots(figsize=FIG_DOUBLE_COL)
    _ax.barh(_units, _counts, color="#4A7C8A", edgecolor="white", linewidth=0.3)
    _ax.set_xlabel("Number of falls")
    _ax.set_title("Top 15 units by fall count", fontweight="bold")
    for _i, _v in enumerate(_counts):
        _ax.text(_v + max(_counts) * 0.01, _i, str(_v), va="center", fontsize=8)
    _fig.tight_layout()
    save_figure(_fig, "nb01_fall_locations")
    return


@app.cell
def _(FIG_DOUBLE_COL, df_analytic, mo, pl, plt, save_figure):
    _dept_stats = (
        df_analytic.group_by("admitting_department")
        .agg([
            pl.len().alias("n_encounters"),
            pl.col("fall_flag").sum().alias("n_falls"),
        ])
        .with_columns(
            (pl.col("n_falls") / pl.col("n_encounters") * 100).round(2).alias("fall_rate_pct")
        )
        .sort("n_encounters", descending=True)
        .head(15)
    )

    mo.vstack([
        mo.md("## Fall Rate by Admitting Department (Top 15 by Volume)"),
        mo.ui.table(_dept_stats),
    ])

    _sorted = _dept_stats.sort("fall_rate_pct", descending=False)
    _depts = _sorted["admitting_department"].to_list()
    _rates = _sorted["fall_rate_pct"].to_list()

    _fig, _ax = plt.subplots(figsize=FIG_DOUBLE_COL)
    _ax.barh(_depts, _rates, color="#4A7C8A", edgecolor="white", linewidth=0.3)
    _ax.set_xlabel("Fall rate, %")
    _ax.set_title("Fall rate by admitting department (top 15 by volume)", fontweight="bold")
    for _i, _v in enumerate(_rates):
        _ax.text(_v + max(_rates) * 0.01, _i, f"{_v:.1f}%", va="center", fontsize=8)
    _fig.tight_layout()
    save_figure(_fig, "nb01_department_fall_rates")
    return


@app.cell
def _(FIG_DOUBLE_COL, df_analytic, mo, pl, plt, save_figure):
    _with_dt = df_analytic.filter(pl.col("fall_datetime").is_not_null())
    _n_with_dt = _with_dt.height
    _n_without_dt = df_analytic.filter(
        (pl.col("fall_flag") == 1) & pl.col("fall_datetime").is_null()
    ).height

    _hours = _with_dt["fall_datetime"].dt.hour().to_numpy()
    _weekdays = _with_dt["fall_datetime"].dt.weekday().to_numpy()  # 0=Mon, 6=Sun
    _day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE_COL)

    # Panel A: hour of day
    _ax1.hist(_hours, bins=range(25), color="#4A7C8A", edgecolor="white", linewidth=0.3)
    _ax1.set_xlabel("Hour of day")
    _ax1.set_ylabel("Number of falls")
    _ax1.set_title("A. Falls by hour of day", fontweight="bold")
    _ax1.set_xticks(range(0, 24, 4))

    # Panel B: day of week
    _day_counts = [int((_weekdays == d).sum()) for d in range(7)]
    _ax2.bar(_day_labels, _day_counts, color="#4A7C8A", edgecolor="white", linewidth=0.3)
    _ax2.set_xlabel("Day of week")
    _ax2.set_ylabel("Number of falls")
    _ax2.set_title("B. Falls by day of week", fontweight="bold")

    _fig.tight_layout()
    save_figure(_fig, "nb01_fall_temporal")

    mo.md(
        f"**Temporal patterns**: {_n_with_dt:,} falls with valid datetime; "
        f"{_n_without_dt:,} fallers excluded (missing `fall_datetime`)."
    )
    return


@app.cell
def _(FIG_SINGLE_COL, df_analytic, mo, np, pl, plt, save_figure):
    # Admission-to-fall timing
    _fallers_dt = df_analytic.filter(
        (pl.col("fall_flag") == 1) & pl.col("fall_datetime").is_not_null()
    ).with_columns(
        ((pl.col("fall_datetime") - pl.col("admission_date")).dt.total_seconds() / 3600)
        .alias("hours_to_fall")
    ).filter(pl.col("hours_to_fall") > 0)

    _h2f = _fallers_dt["hours_to_fall"].to_numpy()
    _med = np.median(_h2f)
    _q1, _q3 = np.percentile(_h2f, [25, 75])

    _fig1, _ax1 = plt.subplots(figsize=FIG_SINGLE_COL)
    _ax1.hist(_h2f, bins=50, color="#4A7C8A", edgecolor="white", linewidth=0.3)
    _ax1.axvline(_med, color="#B2182B", linestyle="--", linewidth=1.0)
    _ax1.set_xlabel("Hours from admission to fall")
    _ax1.set_ylabel("Number of falls")
    _ax1.set_title("Admission-to-fall timing", fontweight="bold")
    _ax1.text(
        0.97, 0.95,
        f"Median: {_med:.0f}h [{_q1:.0f}–{_q3:.0f}]",
        transform=_ax1.transAxes, ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8"},
    )
    _fig1.tight_layout()
    save_figure(_fig1, "nb01_admission_to_fall")

    # LOS comparison
    _los_fallers = df_analytic.filter(pl.col("fall_flag") == 1)["los_days"].to_numpy()
    _los_nonfallers = df_analytic.filter(pl.col("fall_flag") == 0)["los_days"].to_numpy()
    _cap = np.percentile(np.concatenate([_los_fallers, _los_nonfallers]), 95)

    _fig2, _ax2 = plt.subplots(figsize=FIG_SINGLE_COL)
    _bp = _ax2.boxplot(
        [_los_nonfallers[_los_nonfallers <= _cap], _los_fallers[_los_fallers <= _cap]],
        labels=["Non-fallers", "Fallers"],
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "#B2182B", "linewidth": 1.0},
    )
    _bp["boxes"][0].set_facecolor("#D1E5F0")
    _bp["boxes"][1].set_facecolor("#FDDBC7")
    _ax2.set_ylabel("Length of stay, days")
    _ax2.set_title("LOS by fall status (capped at 95th pct)", fontweight="bold")
    _fig2.tight_layout()
    save_figure(_fig2, "nb01_los_comparison")

    mo.md(
        f"**Admission-to-fall timing**: median {_med:.0f}h [{_q1:.0f}–{_q3:.0f}], "
        f"n = {len(_h2f):,} (positive lead time).\n\n"
        f"**LOS**: fallers median {np.median(_los_fallers):.1f}d vs "
        f"non-fallers {np.median(_los_nonfallers):.1f}d."
    )
    return


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
