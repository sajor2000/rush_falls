import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    return Path, matplotlib, mo, np, pl, plt, roc_auc_score


@app.cell
def _(mo):
    import textwrap

    mo.md(
        textwrap.dedent(
            """
            # 14 — Time-dependent (Landmark) AUROC by Hospital Day

            **Purpose**: Estimate a **dynamic/landmark AUROC curve** for Epic PMFRS and the
            Morse Fall Scale across hospital days since admission.

            **Design** (24h bins from admission):
            - **Day 0** predictor: admission score (`*_score_admission`).
            - **Day d ≥ 1** predictor: **daily max score from day (d−1)** (to avoid look-ahead).
            - Outcome: **fall occurs during hospital day d**.

            **Data requirement**: this notebook needs a **timestamped score extract** (or a
            pre-aggregated daily-max extract). If the extract is not present, the notebook
            will **skip** the analysis.
            """
        ).strip()
    )
    return


@app.cell
def _():
    from utils.constants import ALPHA, MIN_SUBGROUP_EVENTS, N_BOOTSTRAP, RANDOM_SEED
    from utils.plotting import COLORS, FIG_DOUBLE_COL, JAMA_STYLE, save_figure

    return (
        ALPHA,
        COLORS,
        FIG_DOUBLE_COL,
        JAMA_STYLE,
        MIN_SUBGROUP_EVENTS,
        N_BOOTSTRAP,
        RANDOM_SEED,
        save_figure,
    )


@app.cell
def _(JAMA_STYLE, matplotlib):
    matplotlib.rcParams.update(JAMA_STYLE)
    return


@app.cell
def _(Path):
    analytic_path = Path("data/processed/analytic.parquet")
    score_extract_candidates = [
        Path("data/raw/scores_timeseries.parquet"),
        Path("data/raw/scores_timeseries.csv"),
        Path("data/raw/scores_daily_max.parquet"),
        Path("data/raw/scores_daily_max.csv"),
    ]
    scores_path = next((p for p in score_extract_candidates if p.exists()), None)
    has_scores_extract = scores_path is not None
    return analytic_path, has_scores_extract, score_extract_candidates, scores_path


@app.cell
def _(has_scores_extract, mo, score_extract_candidates, scores_path):
    if has_scores_extract:
        _status = mo.md(f"**Score extract found**: `{scores_path}`")
    else:
        _cands = "\n".join(f"- `{p}`" for p in score_extract_candidates)
        _status = mo.md(
            """
            **No score time-series extract found** — skipping time-dependent AUROC.

            To run this notebook, place one of the following files in `data/raw/`:
            """
            + "\n"
            + _cands
            + """

            **Expected schema (time-series)**:
            - `encounter_csn`
            - `score_datetime`
            - `instrument` ("epic" | "morse")
            - `score`

            (Optional but helpful: `mrn`, source IDs for de-duplication.)
            """
        )

    _status
    return


@app.cell
def _(analytic_path, mo, pl):
    df = pl.read_parquet(analytic_path)
    mo.md(
        f"**Loaded**: `{analytic_path}` ({df.height:,} rows, {df.width} columns)"
    )
    return (df,)


@app.cell
def _(df, mo, pl):
    # Time-dependent analysis requires fall_datetime for fallers.
    df_td = df.filter(
        (pl.col("fall_flag") == 0) | (pl.col("fall_datetime").is_not_null())
    )
    _dropped = df.height - df_td.height

    df_td = df_td.with_columns(
        pl.when(pl.col("fall_datetime").is_not_null())
        .then(
            ((pl.col("fall_datetime") - pl.col("admission_date")).dt.total_seconds() / 86400)
            .floor()
            .cast(pl.Int32)
        )
        .otherwise(None)
        .alias("fall_day_index")
    )

    mo.md(
        f"**Time-dependent cohort**: {df_td.height:,} encounters "
        f"(dropped {_dropped:,} fall encounters missing `fall_datetime`)."
    )
    return (df_td,)


@app.cell
def _():
    # Default horizon: days 0..14 since admission.
    max_hospital_day = 14
    # Bootstrap replicates per day (cluster bootstrap by MRN).
    # Reduce to ~500 for iteration speed if needed.
    n_bootstrap = 2000
    return max_hospital_day, n_bootstrap


@app.cell
def _(Path, df_td, has_scores_extract, max_hospital_day, mo, pl, scores_path):
    scores_daily_wide = None

    if has_scores_extract:
        scores = (
            pl.read_parquet(scores_path)
            if scores_path.suffix.lower() == ".parquet"
            else pl.read_csv(scores_path)
        )

        # Normalize column names to lowercase.
        _lower = {c: c.lower() for c in scores.columns if c != c.lower()}
        if _lower:
            scores = scores.rename(_lower)

        cols = set(scores.columns)

        if {"instrument", "hospital_day_index"}.issubset(cols):
            # Pre-aggregated daily-max format.
            if "score_day_max" in cols:
                score_col = "score_day_max"
            elif "score" in cols:
                score_col = "score"
            else:
                raise ValueError(
                    "Daily extract must include `score_day_max` (preferred) or `score`."
                )

            scores_daily = scores.select(
                [
                    pl.col("encounter_csn"),
                    pl.col("hospital_day_index").cast(pl.Int32),
                    pl.col("instrument"),
                    pl.col(score_col).cast(pl.Float64).alias("score_day_max"),
                ]
            )

        else:
            # Time-series format.
            required = {"encounter_csn", "score_datetime", "instrument", "score"}
            missing = sorted(required - cols)
            if missing:
                raise ValueError(
                    f"Time-series extract missing required columns: {missing}. "
                    f"Found columns: {sorted(cols)}"
                )

            scores_ts = scores
            if scores_ts["score_datetime"].dtype in (pl.String, pl.Utf8):
                scores_ts = scores_ts.with_columns(
                    pl.col("score_datetime")
                    .str.to_datetime(strict=False)
                    .alias("score_datetime")
                )

            scores_ts = scores_ts.with_columns(
                pl.col("instrument").cast(pl.Utf8).str.to_lowercase().alias("instrument")
            ).with_columns(
                pl.when(pl.col("instrument").str.contains("morse"))
                .then(pl.lit("morse"))
                .when(
                    pl.col("instrument").str.contains("epic")
                    | pl.col("instrument").str.contains("pmfrs")
                )
                .then(pl.lit("epic"))
                .otherwise(pl.col("instrument"))
                .alias("instrument")
            )

            scores_ts = scores_ts.filter(pl.col("instrument").is_in(["epic", "morse"]))

            # Join to analytic cohort to get admission/discharge timestamps and restrict scope.
            _enc = df_td.select(
                [
                    "encounter_csn",
                    "admission_date",
                    "discharge_date",
                ]
            )
            scores_ts = scores_ts.join(_enc, on="encounter_csn", how="inner")

            scores_ts = scores_ts.filter(
                (pl.col("score_datetime") >= pl.col("admission_date"))
                & (pl.col("score_datetime") <= pl.col("discharge_date"))
            )

            scores_ts = scores_ts.with_columns(
                (
                    (
                        (pl.col("score_datetime") - pl.col("admission_date")).dt.total_seconds()
                        / 86400
                    )
                    .floor()
                    .cast(pl.Int32)
                ).alias("hospital_day_index")
            ).filter(
                (pl.col("hospital_day_index") >= 0)
                & (pl.col("hospital_day_index") <= max_hospital_day)
            )

            scores_daily = scores_ts.group_by(
                ["encounter_csn", "instrument", "hospital_day_index"]
            ).agg(pl.col("score").cast(pl.Float64).max().alias("score_day_max"))

        # Pivot to wide: one row per encounter-day.
        scores_daily_wide = scores_daily.pivot(
            index=["encounter_csn", "hospital_day_index"],
            columns="instrument",
            values="score_day_max",
            aggregate_function="first",
        ).rename({
            "epic": "epic_day_max",
            "morse": "morse_day_max",
        })

        # Ensure both columns exist (even if one instrument absent).
        for _col in ["epic_day_max", "morse_day_max"]:
            if _col not in scores_daily_wide.columns:
                scores_daily_wide = scores_daily_wide.with_columns(pl.lit(None).alias(_col))

        _status = mo.md(
            f"**Loaded score extract**: {scores.height:,} rows → "
            f"daily max table {scores_daily_wide.height:,} encounter-days."
        )
    else:
        _status = mo.md("*Skipping score processing*: no score extract found.")

    _status
    return (scores_daily_wide,)


@app.cell
def _(df_td, max_hospital_day, pl):
    _days = pl.DataFrame({"hospital_day_index": list(range(max_hospital_day + 1))})
    panel = df_td.select(
        [
            "mrn",
            "encounter_csn",
            "los_days",
            "fall_day_index",
            "epic_score_admission",
            "morse_score_admission",
        ]
    ).join(_days, how="cross")

    # Risk set at the *start* of each hospital day.
    panel = panel.filter(
        (pl.col("los_days") > pl.col("hospital_day_index"))
        & (pl.col("fall_day_index").is_null() | (pl.col("fall_day_index") >= pl.col("hospital_day_index")))
    )

    panel = panel.with_columns(
        (pl.col("fall_day_index") == pl.col("hospital_day_index")).cast(pl.Int8).alias("y")
    )
    return (panel,)


@app.cell
def _(panel, pl, scores_daily_wide):
    landmark_df = None

    if scores_daily_wide is not None:
        _panel = panel.join(
            scores_daily_wide,
            on=["encounter_csn", "hospital_day_index"],
            how="left",
        ).sort(["encounter_csn", "hospital_day_index"])

        _panel = _panel.with_columns(
            [
                pl.when(pl.col("hospital_day_index") == 0)
                .then(
                    pl.coalesce(
                        [pl.col("epic_day_max"), pl.col("epic_score_admission")]
                    )
                )
                .otherwise(pl.col("epic_day_max"))
                .alias("epic_day"),
                pl.when(pl.col("hospital_day_index") == 0)
                .then(
                    pl.coalesce(
                        [pl.col("morse_day_max"), pl.col("morse_score_admission")]
                    )
                )
                .otherwise(pl.col("morse_day_max"))
                .alias("morse_day"),
            ]
        ).with_columns(
            [
                pl.col("epic_day")
                .fill_null(strategy="forward")
                .over("encounter_csn")
                .alias("epic_day_ffill"),
                pl.col("morse_day")
                .fill_null(strategy="forward")
                .over("encounter_csn")
                .alias("morse_day_ffill"),
            ]
        ).with_columns(
            [
                pl.col("epic_day_ffill")
                .shift(1)
                .over("encounter_csn")
                .alias("epic_lag1"),
                pl.col("morse_day_ffill")
                .shift(1)
                .over("encounter_csn")
                .alias("morse_lag1"),
            ]
        ).with_columns(
            [
                pl.when(pl.col("hospital_day_index") == 0)
                .then(pl.col("epic_score_admission"))
                .otherwise(pl.col("epic_lag1"))
                .alias("epic_pred"),
                pl.when(pl.col("hospital_day_index") == 0)
                .then(pl.col("morse_score_admission"))
                .otherwise(pl.col("morse_lag1"))
                .alias("morse_pred"),
            ]
        )

        landmark_df = _panel.select(
            [
                pl.col("mrn"),
                pl.col("encounter_csn"),
                pl.col("hospital_day_index").alias("day"),
                pl.col("y"),
                pl.col("epic_pred"),
                pl.col("morse_pred"),
            ]
        )
    return (landmark_df,)


@app.cell
def _(ALPHA, MIN_SUBGROUP_EVENTS, RANDOM_SEED, landmark_df, mo, n_bootstrap, np, pl, roc_auc_score):
    auc_by_day = None
    _status = mo.md("*Skipping AUROC computation*: missing score extract.")

    if landmark_df is not None:

        def _cluster_boot_ci_pair(
            y,
            s_a,
            s_b,
            clusters,
            *,
            n_boot: int,
            seed: int,
            alpha: float,
        ) -> tuple[tuple[float, float], tuple[float, float]]:
            _, inv = np.unique(clusters, return_inverse=True)
            n_clusters = int(inv.max()) + 1
            rng = np.random.default_rng(seed)

            auc_a: list[float] = []
            auc_b: list[float] = []
            for _ in range(n_boot):
                sampled = rng.integers(0, n_clusters, size=n_clusters)
                w_c = np.bincount(sampled, minlength=n_clusters)
                w = w_c[inv]

                if np.sum(w[y == 1]) == 0 or np.sum(w[y == 0]) == 0:
                    continue

                auc_a.append(float(roc_auc_score(y, s_a, sample_weight=w)))
                auc_b.append(float(roc_auc_score(y, s_b, sample_weight=w)))

            if len(auc_a) < max(50, int(0.25 * n_boot)):
                return (float("nan"), float("nan")), (float("nan"), float("nan"))

            lo = 100 * alpha / 2
            hi = 100 * (1 - alpha / 2)
            return (
                (float(np.percentile(auc_a, lo)), float(np.percentile(auc_a, hi))),
                (float(np.percentile(auc_b, lo)), float(np.percentile(auc_b, hi))),
            )

        rows: list[dict] = []
        days = landmark_df["day"].unique().sort().to_list()
        for d in days:
            sub = landmark_df.filter(pl.col("day") == d)
            n = sub.height
            n_events = int(sub["y"].sum())
            n_nonevents = n - n_events
            n_patients = int(sub["mrn"].n_unique())

            if n_events < MIN_SUBGROUP_EVENTS or n_nonevents < MIN_SUBGROUP_EVENTS:
                rows.append(
                    {
                        "day": int(d),
                        "n_rows": int(n),
                        "n_patients": n_patients,
                        "n_events": n_events,
                        "epic_auc": None,
                        "epic_ci_lower": None,
                        "epic_ci_upper": None,
                        "morse_auc": None,
                        "morse_ci_lower": None,
                        "morse_ci_upper": None,
                    }
                )
                continue

            y = sub["y"].to_numpy()
            epic = sub["epic_pred"].to_numpy()
            morse = sub["morse_pred"].to_numpy()
            clusters = sub["mrn"].to_numpy()

            _epic_auc = float(roc_auc_score(y, epic))
            _morse_auc = float(roc_auc_score(y, morse))

            (ep_lo, ep_hi), (mo_lo, mo_hi) = _cluster_boot_ci_pair(
                y,
                epic,
                morse,
                clusters,
                n_boot=n_bootstrap,
                seed=RANDOM_SEED + int(d),
                alpha=ALPHA,
            )

            rows.append(
                {
                    "day": int(d),
                    "n_rows": int(n),
                    "n_patients": n_patients,
                    "n_events": n_events,
                    "epic_auc": _epic_auc,
                    "epic_ci_lower": ep_lo,
                    "epic_ci_upper": ep_hi,
                    "morse_auc": _morse_auc,
                    "morse_ci_lower": mo_lo,
                    "morse_ci_upper": mo_hi,
                }
            )

        auc_by_day = pl.DataFrame(rows).sort("day")
        _status = mo.md(
            "Computed day-specific AUROC with cluster bootstrap CIs (cluster = MRN). "
            f"Bootstrap replicates per day: {n_bootstrap:,}. "
            f"Days with <{MIN_SUBGROUP_EVENTS} events are left blank."
        )

    _status
    return (auc_by_day,)


@app.cell
def _(Path, auc_by_day, mo):
    if auc_by_day is None:
        _status = mo.md("*Skipping save*: AUROC table not computed.")
    else:
        out_dir = Path("outputs/tables")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "etable_time_dependent_auc_by_day.csv"
        auc_by_day.write_csv(out_path)
        _status = mo.md(f"**Saved**: `{out_path}`")

    _status
    return


@app.cell
def _(COLORS, FIG_DOUBLE_COL, auc_by_day, mo, plt, save_figure):
    if auc_by_day is None:
        _status = mo.md("*Skipping figure*: AUROC table not computed.")
    else:
        dfp = auc_by_day.drop_nulls(["epic_auc", "morse_auc"])
        if dfp.height == 0:
            _status = mo.md(
                "No days met minimum event/non-event thresholds for AUROC estimation."
            )
        else:
            fig, ax = plt.subplots(figsize=FIG_DOUBLE_COL)

            x = dfp["day"].to_numpy()

            _epic_auc = dfp["epic_auc"].to_numpy()
            _epic_lo = dfp["epic_ci_lower"].to_numpy()
            _epic_hi = dfp["epic_ci_upper"].to_numpy()

            _morse_auc = dfp["morse_auc"].to_numpy()
            _morse_lo = dfp["morse_ci_lower"].to_numpy()
            _morse_hi = dfp["morse_ci_upper"].to_numpy()

            ax.plot(x, _epic_auc, color=COLORS["epic"], label="Epic PMFRS")
            ax.fill_between(
                x,
                _epic_lo,
                _epic_hi,
                color=COLORS["epic"],
                alpha=0.15,
                linewidth=0,
            )

            ax.plot(x, _morse_auc, color=COLORS["morse"], label="Morse Fall Scale")
            ax.fill_between(
                x,
                _morse_lo,
                _morse_hi,
                color=COLORS["morse"],
                alpha=0.15,
                linewidth=0,
            )

            ax.axhline(0.5, color="0.6", linewidth=0.8, linestyle="--")
            ax.set_xlim(min(x), max(x))
            ax.set_ylim(0.4, 1.0)
            ax.set_xlabel("Hospital day since admission")
            ax.set_ylabel("AUROC")
            ax.set_title("Landmark AUROC by hospital day (next-day prediction)")
            ax.legend(loc="lower right")

            save_figure(fig, "efigure_time_dependent_auc_by_day")
            _status = mo.md(
                "**Saved**: `outputs/figures/efigure_time_dependent_auc_by_day.pdf` and `.png`."
            )

    _status
    return


@app.cell
def _(auc_by_day, mo):
    out = (
        mo.ui.table(auc_by_day)
        if auc_by_day is not None
        else mo.md("*AUROC table not available (missing score extract).*")
    )
    out
    return


if __name__ == "__main__":
    app.run()
