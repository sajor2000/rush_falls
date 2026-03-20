import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy import stats

    return Path, mo, np, pl, stats


@app.cell
def _(mo):
    mo.md("""
    # Table 1 — Patient Characteristics by Fall Status

    Stratified descriptive statistics for the complete-case analytic cohort
    (Epic PMFRS and Morse Fall Scale admission scores both present).

    - Continuous (normal): mean ± SD — age
    - Continuous (skewed): median [IQR] — los_days
    - Categorical: n (%) — gender, race, ethnicity
    - Scores: mean ± SD for all score columns
    - Standardized mean difference (SMD) for each variable
    - P-values: two-sample t-test (age, scores), Wilcoxon rank-sum (los_days),
      chi-square (categorical)

    **Input**: `data/processed/analytic.parquet`
    **Output**: `outputs/tables/table1.csv`
    """)
    return


@app.cell
def _(Path, pl):
    _path = Path("data/processed/analytic.parquet")
    df = pl.read_parquet(_path)
    return (df,)


@app.cell
def _(df, mo, pl):
    _n = df.height
    _n_fall = int(df.filter(pl.col("fall_flag") == 1).height)
    _n_nofall = int(df.filter(pl.col("fall_flag") == 0).height)
    mo.md(
        f"""
        ## Cohort Overview

        | | N |
        |---|---|
        | Total encounters | {_n:,} |
        | Fall (fall_flag = 1) | {_n_fall:,} ({_n_fall / _n * 100:.1f}%) |
        | No fall (fall_flag = 0) | {_n_nofall:,} ({_n_nofall / _n * 100:.1f}%) |
        """
    )
    return


@app.cell
def _(np, pl, stats):
    def smd_continuous(s1, s2) -> float:
        """Cohen's d-style SMD for two continuous groups."""
        _m1, _sd1 = s1.mean(), s1.std()
        _m2, _sd2 = s2.mean(), s2.std()
        if _m1 is None or _m2 is None or _sd1 is None or _sd2 is None:
            return float("nan")
        _pooled = np.sqrt((_sd1**2 + _sd2**2) / 2)
        return float(abs(_m1 - _m2) / _pooled) if _pooled > 0 else float("nan")

    def smd_categorical(df_grp, col: str) -> float:
        """SMD for a binary/categorical variable using proportions.

        For multi-category variables, uses the maximum pairwise proportion
        difference as a summary (consistent with TRIPOD/Table 1 convention).
        """
        _vals = df_grp[col].drop_nulls().unique().to_list()
        if len(_vals) < 2:
            return float("nan")
        _fall = df_grp.filter(pl.col("fall_flag") == 1)
        _nofall = df_grp.filter(pl.col("fall_flag") == 0)
        _n1 = max(_fall.height, 1)
        _n0 = max(_nofall.height, 1)
        _smds = []
        for _v in _vals:
            _p1 = _fall.filter(pl.col(col) == _v).height / _n1
            _p0 = _nofall.filter(pl.col(col) == _v).height / _n0
            _denom = np.sqrt((_p1 * (1 - _p1) + _p0 * (1 - _p0)) / 2)
            if _denom > 0:
                _smds.append(abs(_p1 - _p0) / _denom)
        return float(np.max(_smds)) if _smds else float("nan")

    def ttest_p(s1, s2) -> float:
        """Welch two-sample t-test p-value."""
        _a = s1.drop_nulls().to_numpy()
        _b = s2.drop_nulls().to_numpy()
        if len(_a) < 2 or len(_b) < 2:
            return float("nan")
        return float(stats.ttest_ind(_a, _b, equal_var=False).pvalue)

    def wilcoxon_p(s1, s2) -> float:
        """Wilcoxon rank-sum (Mann-Whitney U) p-value."""
        _a = s1.drop_nulls().to_numpy()
        _b = s2.drop_nulls().to_numpy()
        if len(_a) < 2 or len(_b) < 2:
            return float("nan")
        return float(stats.mannwhitneyu(_a, _b, alternative="two-sided").pvalue)

    def chisq_p(df_grp, col: str) -> float:
        """Chi-square test of independence p-value."""
        _ct = (
            df_grp.group_by(["fall_flag", col])
            .agg(pl.len().alias("n"))
            .pivot(index=col, on="fall_flag", values="n")
            .fill_null(0)
        )
        # Extract the numeric columns (all except the category column)
        _num_cols = [c for c in _ct.columns if c != col]
        if len(_num_cols) < 2:
            return float("nan")
        _matrix = _ct.select(_num_cols).to_numpy()
        _chi2, _p, _dof, _exp = stats.chi2_contingency(_matrix)
        return float(_p)

    def fmt_pval(p: float) -> str:
        if np.isnan(p):
            return "—"
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    def fmt_smd(s: float) -> str:
        if np.isnan(s):
            return "—"
        return f"{s:.3f}"

    return (
        chisq_p,
        fmt_pval,
        fmt_smd,
        smd_categorical,
        smd_continuous,
        ttest_p,
        wilcoxon_p,
    )


@app.cell
def _(df, pl):
    df_fall = df.filter(pl.col("fall_flag") == 1)
    df_nofall = df.filter(pl.col("fall_flag") == 0)
    n_fall = df_fall.height
    n_nofall = df_nofall.height
    n_total = df.height
    return df_fall, df_nofall, n_fall, n_nofall, n_total


@app.cell
def _(
    chisq_p,
    df,
    df_fall,
    df_nofall,
    fmt_pval,
    fmt_smd,
    n_fall,
    n_nofall,
    n_total,
    np,
    pl,
    smd_categorical,
    smd_continuous,
    ttest_p,
    wilcoxon_p,
):
    _rows: list[dict] = []

    # ── Header row ────────────────────────────────────────────────────
    _rows.append({
        "Variable": "Encounters, n",
        "Overall": str(n_total),
        "No fall": str(n_nofall),
        "Fall": str(n_fall),
        "SMD": "—",
        "P value": "—",
    })

    # ── Age (mean ± SD, t-test) ───────────────────────────────────────
    _age_all = df["age"].drop_nulls()
    _age_fall = df_fall["age"].drop_nulls()
    _age_nofall = df_nofall["age"].drop_nulls()
    _rows.append({
        "Variable": "Age, years — mean \u00b1 SD",
        "Overall": f"{_age_all.mean():.1f} \u00b1 {_age_all.std():.1f}",
        "No fall": f"{_age_nofall.mean():.1f} \u00b1 {_age_nofall.std():.1f}",
        "Fall": f"{_age_fall.mean():.1f} \u00b1 {_age_fall.std():.1f}",
        "SMD": fmt_smd(smd_continuous(_age_fall, _age_nofall)),
        "P value": fmt_pval(ttest_p(_age_fall, _age_nofall)),
    })

    # ── LOS (median [IQR], Wilcoxon) ─────────────────────────────────
    _los_all = df["los_days"].drop_nulls()
    _los_fall = df_fall["los_days"].drop_nulls()
    _los_nofall = df_nofall["los_days"].drop_nulls()
    _rows.append({
        "Variable": "Length of stay, days — median [IQR]",
        "Overall": (
            f"{_los_all.median():.1f} "
            f"[{np.percentile(_los_all.to_numpy(), 25):.1f}\u2013"
            f"{np.percentile(_los_all.to_numpy(), 75):.1f}]"
        ),
        "No fall": (
            f"{_los_nofall.median():.1f} "
            f"[{np.percentile(_los_nofall.to_numpy(), 25):.1f}\u2013"
            f"{np.percentile(_los_nofall.to_numpy(), 75):.1f}]"
        ),
        "Fall": (
            f"{_los_fall.median():.1f} "
            f"[{np.percentile(_los_fall.to_numpy(), 25):.1f}\u2013"
            f"{np.percentile(_los_fall.to_numpy(), 75):.1f}]"
        ),
        "SMD": fmt_smd(smd_continuous(_los_fall, _los_nofall)),
        "P value": fmt_pval(wilcoxon_p(_los_fall, _los_nofall)),
    })

    # ── Categorical helper ────────────────────────────────────────────
    def _cat_rows(col: str, section_label: str) -> list[dict]:
        """Produce one header row + one row per category value."""
        _out = []
        _vals = (
            df[col].drop_nulls().value_counts()
            .sort("count", descending=True)[col]
            .to_list()
        )
        # Section header
        _out.append({
            "Variable": section_label,
            "Overall": "",
            "No fall": "",
            "Fall": "",
            "SMD": fmt_smd(smd_categorical(df, col)),
            "P value": fmt_pval(chisq_p(df, col)),
        })
        for _v in _vals:
            _n_all_v = df.filter(pl.col(col) == _v).height
            _n_nofall_v = df_nofall.filter(pl.col(col) == _v).height
            _n_fall_v = df_fall.filter(pl.col(col) == _v).height
            _n_all_obs = df[col].drop_nulls().len()
            _n_nofall_obs = df_nofall[col].drop_nulls().len()
            _n_fall_obs = df_fall[col].drop_nulls().len()
            _out.append({
                "Variable": f"  {_v}",
                "Overall": f"{_n_all_v:,} ({_n_all_v / max(_n_all_obs, 1) * 100:.1f}%)",
                "No fall": f"{_n_nofall_v:,} ({_n_nofall_v / max(_n_nofall_obs, 1) * 100:.1f}%)",
                "Fall": f"{_n_fall_v:,} ({_n_fall_v / max(_n_fall_obs, 1) * 100:.1f}%)",
                "SMD": "—",
                "P value": "—",
            })
        return _out

    # Gender
    _rows.extend(_cat_rows("gender", "Sex — n (%)"))

    # Race
    _rows.extend(_cat_rows("race", "Race — n (%)"))

    # Ethnicity
    _rows.extend(_cat_rows("ethnicity", "Ethnicity — n (%)"))

    # ── Score variables (mean ± SD, t-test) ────────────────────────────
    _score_defs = [
        ("epic_score_admission", "Epic PMFRS at admission — mean \u00b1 SD"),
        ("morse_score_admission", "Morse Fall Scale at admission — mean \u00b1 SD"),
        ("epic_score_max", "Epic PMFRS, max during encounter — mean \u00b1 SD"),
        ("morse_score_max", "Morse Fall Scale, max during encounter — mean \u00b1 SD"),
        ("epic_score_mean", "Epic PMFRS, mean during encounter — mean \u00b1 SD"),
        ("morse_score_mean", "Morse Fall Scale, mean during encounter — mean \u00b1 SD"),
        ("epic_score_median", "Epic PMFRS, median during encounter — mean \u00b1 SD"),
        ("morse_score_median", "Morse Fall Scale, median during encounter — mean \u00b1 SD"),
    ]

    for _col, _label in _score_defs:
        if _col not in df.columns:
            continue
        _s_all = df[_col].drop_nulls()
        _s_fall = df_fall[_col].drop_nulls()
        _s_nofall = df_nofall[_col].drop_nulls()
        _rows.append({
            "Variable": _label,
            "Overall": f"{_s_all.mean():.2f} \u00b1 {_s_all.std():.2f}",
            "No fall": f"{_s_nofall.mean():.2f} \u00b1 {_s_nofall.std():.2f}",
            "Fall": f"{_s_fall.mean():.2f} \u00b1 {_s_fall.std():.2f}",
            "SMD": fmt_smd(smd_continuous(_s_fall, _s_nofall)),
            "P value": fmt_pval(ttest_p(_s_fall, _s_nofall)),
        })

    # ── Score-before-fall (fallers only) ──────────────────────────────
    _bf_defs = [
        ("epic_score_before_fall", "Epic PMFRS before fall (fallers only) — mean \u00b1 SD"),
        ("morse_score_before_fall", "Morse Fall Scale before fall (fallers only) — mean \u00b1 SD"),
    ]
    for _col, _label in _bf_defs:
        if _col not in df.columns:
            continue
        _s_fall = df_fall[_col].drop_nulls()
        if _s_fall.len() == 0:
            continue
        _rows.append({
            "Variable": _label,
            "Overall": "—",
            "No fall": "—",
            "Fall": f"{_s_fall.mean():.2f} \u00b1 {_s_fall.std():.2f}",
            "SMD": "—",
            "P value": "—",
        })

    table1_df = pl.DataFrame(_rows)
    return (table1_df,)


@app.cell
def _(mo, n_fall, n_nofall, n_total):
    mo.md(
        f"""
        ## Table 1. Patient Characteristics by Fall Status

        | | Overall (N\u202f=\u202f{n_total:,}) | No fall (n\u202f=\u202f{n_nofall:,}) | Fall (n\u202f=\u202f{n_fall:,}) | SMD | P value |
        |---|---|---|---|---|---|
        *(See interactive table below)*

        SMD\u202f=\u202fstandardized mean difference. IQR\u202f=\u202finterquartile range.
        P values: t-test for age and scores; Wilcoxon rank-sum for length of stay;
        chi-square for categorical variables. SMD\u202f>\u202f0.10 indicates potential
        imbalance.
        """
    )
    return


@app.cell
def _(mo, table1_df):
    mo.ui.table(table1_df)
    return


@app.cell
def _(n_fall, n_nofall, n_total, table1_df):
    from great_tables import GT, loc, style

    _gt = (
        GT(table1_df)
        .tab_header(
            title="Table 1. Patient Characteristics by Fall Status",
            subtitle=(
                f"Overall N\u202f=\u202f{n_total:,}; "
                f"No fall n\u202f=\u202f{n_nofall:,}; "
                f"Fall n\u202f=\u202f{n_fall:,}"
            ),
        )
        .cols_label(
            cases={
                "Variable": "Variable",
                "Overall": f"Overall\n(N\u202f=\u202f{n_total:,})",
                "No fall": f"No fall\n(n\u202f=\u202f{n_nofall:,})",
                "Fall": f"Fall\n(n\u202f=\u202f{n_fall:,})",
                "SMD": "SMD",
                "P value": "P value",
            }
        )
        .tab_source_note(
            "SMD = standardized mean difference. IQR = interquartile range. "
            "P values: t-test (age, scores); Wilcoxon rank-sum (LOS); "
            "chi-square (categorical). SMD >0.10 indicates potential imbalance."
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.body(
                columns="Variable",
                rows=[
                    i
                    for i, row in enumerate(table1_df.iter_rows(named=True))
                    if row["Overall"] == "" or (
                        not row["Variable"].startswith("  ")
                        and row["Variable"] != "Encounters, n"
                    )
                ],
            ),
        )
        .cols_align(align="left", columns="Variable")
        .cols_align(align="center", columns=["Overall", "No fall", "Fall", "SMD", "P value"])
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
def _(Path, mo, table1_df):
    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _csv_path = _out_dir / "table1.csv"
    table1_df.write_csv(_csv_path)
    mo.md(f"**Saved**: `{_csv_path}` ({table1_df.height} rows)")
    return


if __name__ == "__main__":
    app.run()
