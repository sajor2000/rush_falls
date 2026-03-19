"""
06_decision_curve.py — Decision Curve Analysis (Figure 3)

Decision curve analysis comparing Epic PMFRS vs Morse Fall Scale net benefit
across threshold probabilities 0-10%, the clinically relevant range for
a 1% baseline fall prevalence.

Inputs:  data/processed/analytic.parquet
Outputs: outputs/figures/figure3_dca.pdf
         outputs/figures/figure3_dca.png
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
# Figure 3. Decision Curve Analysis

Net benefit curves for Epic PMFRS, Morse Fall Scale, treat-all, and treat-none
across threshold probabilities 0–10%. DCA quantifies clinical utility by weighing
the benefit of true positives against the harm of false positives at each decision
threshold.

**Reference**: Vickers AJ et al. *Diagn Progn Res* 2019;3:18.
"""
    )
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    return Path, np, pl, plt


@app.cell
def _():
    from utils.constants import (
        EPIC_2TIER_HIGH,
        EPIC_3TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        MFS_HIGH,
        MFS_MODERATE,
    )
    from utils.metrics import extract_dca_threshold_range, logistic_recalibration
    from utils.plotting import COLORS, JAMA_STYLE, save_figure
    return (
        COLORS,
        EPIC_2TIER_HIGH,
        EPIC_3TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        JAMA_STYLE,
        MFS_HIGH,
        MFS_MODERATE,
        extract_dca_threshold_range,
        logistic_recalibration,
        save_figure,
    )


# ── Load analytic dataset ────────────────────────────────────────────
@app.cell
def _(Path, pl):
    _path = Path("data/processed/analytic.parquet")
    df = pl.read_parquet(_path)
    return (df,)


@app.cell
def _(df, mo):
    _n = df.height
    _n_falls = df.filter(df["fall_flag"] == 1).height
    mo.md(
        f"""
        ## Dataset

        - Encounters: {_n:,}
        - Falls: {_n_falls:,} ({_n_falls / _n * 100:.1f}%)
        - Fall prevalence: {_n_falls / _n:.4f}
        """
    )
    return


# ── Logistic recalibration: score → probability ──────────────────────
@app.cell
def _(EPIC_2TIER_HIGH, EPIC_3TIER_HIGH, EPIC_3TIER_MEDIUM, MFS_HIGH, MFS_MODERATE, df, logistic_recalibration):
    _y = df["fall_flag"].to_numpy()
    _epic_score = df["epic_score_admission"].to_numpy()
    _morse_score = df["morse_score_admission"].to_numpy()

    epic_prob, _epic_lr = logistic_recalibration(_epic_score, _y)
    morse_prob, _morse_lr = logistic_recalibration(_morse_score, _y)

    # Probability equivalents for DCA annotations
    morse_prob_at_25 = float(_morse_lr.predict_proba([[float(MFS_MODERATE)]])[0, 1])
    morse_prob_at_45 = float(_morse_lr.predict_proba([[float(MFS_HIGH)]])[0, 1])
    epic_prob_at_35 = float(_epic_lr.predict_proba([[float(EPIC_3TIER_MEDIUM)]])[0, 1])
    epic_prob_at_50 = float(_epic_lr.predict_proba([[float(EPIC_2TIER_HIGH)]])[0, 1])
    epic_prob_at_70 = float(_epic_lr.predict_proba([[float(EPIC_3TIER_HIGH)]])[0, 1])

    y_true = _y
    return epic_prob, epic_prob_at_35, epic_prob_at_50, epic_prob_at_70, morse_prob, morse_prob_at_25, morse_prob_at_45, y_true


# ── Attach probabilities to Polars df, then convert for dcurves ─────
@app.cell
def _(df, epic_prob, mo, morse_prob, np, pl):
    df_with_probs = df.with_columns(
        [
            pl.Series("epic_prob", epic_prob),
            pl.Series("morse_prob", morse_prob),
        ]
    )

    mo.md(
        f"""
        ### Recalibrated Probabilities

        | Statistic | Epic PMFRS | Morse Fall Scale |
        |---|---|---|
        | Mean prob | {epic_prob.mean():.4f} | {morse_prob.mean():.4f} |
        | Median prob | {float(np.median(epic_prob)):.4f} | {float(np.median(morse_prob)):.4f} |
        | Max prob | {epic_prob.max():.4f} | {morse_prob.max():.4f} |
        """
    )
    return (df_with_probs,)


# ── DCA via dcurves ──────────────────────────────────────────────────
@app.cell
def _(df_with_probs, np):
    from dcurves import dca

    # Convert to pandas only at dcurves boundary
    pdf = df_with_probs.select(
        ["fall_flag", "epic_prob", "morse_prob"]
    ).to_pandas()

    df_dca = dca(
        data=pdf,
        outcome="fall_flag",
        modelnames=["epic_prob", "morse_prob"],
        thresholds=np.arange(0, 0.10, 0.001),
    )
    return (df_dca,)


@app.cell
def _(df_dca, mo):
    mo.md(
        f"""
        ### DCA Results Shape

        - Rows: {len(df_dca):,}
        - Columns: {list(df_dca.columns)}
        """
    )
    return


# ── Extract net benefit by model ─────────────────────────────────────
@app.cell
def _(df_dca):
    """
    dcurves returns a tidy long-format DataFrame with columns:
    model, threshold, net_benefit, ... (varies by version).
    Extract each model's series by filtering on the model name.
    """
    _models = df_dca["model"].unique().tolist() if "model" in df_dca.columns else []

    def _get_nb(model_name: str):
        _sub = df_dca[df_dca["model"] == model_name].sort_values("threshold")
        return _sub["threshold"].to_numpy(), _sub["net_benefit"].to_numpy()

    thresh_epic, nb_epic = _get_nb("epic_prob")
    thresh_morse, nb_morse = _get_nb("morse_prob")
    thresh_all, nb_all = _get_nb("all")
    thresh_none, nb_none = _get_nb("none")
    return nb_all, nb_epic, nb_morse, nb_none, thresh_all, thresh_epic, thresh_morse, thresh_none


# ── Figure 3: DCA plot ────────────────────────────────────────────────
@app.cell
def _(
    COLORS,
    JAMA_STYLE,
    epic_prob_at_35,
    epic_prob_at_50,
    epic_prob_at_70,
    morse_prob_at_25,
    morse_prob_at_45,
    nb_all,
    nb_epic,
    nb_morse,
    nb_none,
    plt,
    save_figure,
    thresh_all,
    thresh_epic,
    thresh_morse,
    thresh_none,
):
    with plt.rc_context(JAMA_STYLE):
        fig3, ax3 = plt.subplots(figsize=(7.0, 4.5))

        # Treat-all and treat-none (background reference lines)
        ax3.plot(
            thresh_all * 100,
            nb_all,
            color=COLORS["treat_all"],
            linewidth=1.0,
            linestyle="--",
            label="Treat all",
            zorder=1,
        )
        ax3.plot(
            thresh_none * 100,
            nb_none,
            color=COLORS["treat_none"],
            linewidth=0.75,
            linestyle=":",
            label="Treat none",
            zorder=1,
        )

        # Model curves
        ax3.plot(
            thresh_epic * 100,
            nb_epic,
            color=COLORS["epic"],
            linewidth=1.5,
            linestyle="-",
            label="Epic PMFRS",
            zorder=3,
        )
        ax3.plot(
            thresh_morse * 100,
            nb_morse,
            color=COLORS["morse"],
            linewidth=1.5,
            linestyle="--",
            label="Morse Fall Scale",
            zorder=2,
        )

        # Axes
        ax3.set_xlabel("Threshold probability (%)", fontsize=9)
        ax3.set_ylabel("Net benefit", fontsize=9)
        ax3.set_xlim(0, 10)

        # Y-axis: tight to model curves so low net benefit is visible
        _y_min_models = min(nb_epic.min(), nb_morse.min())
        _y_min = max(_y_min_models - 0.002, -0.01)
        _y_max = max(nb_epic.max(), nb_morse.max(), nb_all.max()) + 0.002
        ax3.set_ylim(_y_min, _y_max)

        # Legend placed below the plot — never blocking data
        ax3.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=4,
            fontsize=8,
            frameon=False,
            columnspacing=1.0,
        )

        # Zero reference
        ax3.axhline(0, color="black", linewidth=0.4, linestyle="-", zorder=0)

        # Standard cutoff annotations: Morse + Epic (skip if outside 0-10% DCA range)
        _cutoff_annotations = [
            (morse_prob_at_25, "MFS \u226525", COLORS["morse"]),
            (morse_prob_at_45, "MFS \u226545", COLORS["morse"]),
            (epic_prob_at_35, "Epic \u226535", COLORS["epic"]),
            (epic_prob_at_50, "Epic \u226550", COLORS["epic"]),
            (epic_prob_at_70, "Epic \u226570", COLORS["epic"]),
        ]
        _placed_x = []
        _y_range = _y_max - _y_min
        for _cutoff_prob, _cutoff_label, _color in _cutoff_annotations:
            _x_pct = _cutoff_prob * 100
            if _x_pct > 10:
                continue
            ax3.axvline(
                _x_pct, color=_color, linewidth=0.6,
                linestyle=":", alpha=0.6, zorder=1,
            )
            # Stagger y-position to avoid label overlap
            _y_pos = _y_max - 0.001
            for _px in _placed_x:
                if abs(_x_pct - _px) < 0.6:
                    _y_pos -= _y_range * 0.12
            ax3.text(
                _x_pct, _y_pos, _cutoff_label, fontsize=8,
                color=_color, ha="center", va="top",
                rotation=90, alpha=0.8,
            )
            _placed_x.append(_x_pct)

        # Tick formatting: percentage labels on x
        ax3.set_xticks([0, 2, 4, 6, 8, 10])
        ax3.set_xticklabels(["0%", "2%", "4%", "6%", "8%", "10%"], fontsize=8)
        ax3.tick_params(axis="both", labelsize=8)

        fig3.subplots_adjust(bottom=0.22)

    save_figure(fig3, "figure3_dca")
    fig3
    return (fig3,)


# ── DCA-derived threshold ranges ──────────────────────────────────────
@app.cell
def _(Path, df_dca, extract_dca_threshold_range, mo):
    import json

    epic_dca_range = extract_dca_threshold_range(df_dca, "epic_prob")
    morse_dca_range = extract_dca_threshold_range(df_dca, "morse_prob")

    _ranges = {"epic": epic_dca_range, "morse": morse_dca_range}

    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _json_path = _out_dir / "dca_threshold_ranges.json"
    _json_path.write_text(json.dumps(_ranges, indent=2))

    # Pre-format to avoid invalid f-string format specs with ternary
    def _fmt(val, decimals=4):
        return f"{val:.{decimals}f}" if val is not None else "—"

    def _fmt_pct(val, decimals=2):
        return f"{val * 100:.{decimals}f}%" if val is not None else "—"

    mo.md(
        f"""
        ### DCA-Derived Threshold Ranges

        Range where model net benefit > treat-all AND > treat-none:

        | Model | Lower (prob) | Upper (prob) | Lower (%) | Upper (%) |
        |---|---|---|---|---|
        | Epic PMFRS | {_fmt(epic_dca_range['lower'])} | {_fmt(epic_dca_range['upper'])} | {_fmt_pct(epic_dca_range['lower'])} | {_fmt_pct(epic_dca_range['upper'])} |
        | Morse Fall Scale | {_fmt(morse_dca_range['lower'])} | {_fmt(morse_dca_range['upper'])} | {_fmt_pct(morse_dca_range['lower'])} | {_fmt_pct(morse_dca_range['upper'])} |

        **Saved**: `{_json_path}`
        """
    )
    return


# ── Net benefit table at key thresholds ─────────────────────────────
@app.cell
def _(df_dca, mo, pl):
    _key_thresh = [0.01, 0.02, 0.03, 0.05]

    _rows = []
    for _t in _key_thresh:
        _epic_row = df_dca[
            (df_dca["model"] == "epic_prob")
            & (abs(df_dca["threshold"] - _t) < 0.0005)
        ]
        _morse_row = df_dca[
            (df_dca["model"] == "morse_prob")
            & (abs(df_dca["threshold"] - _t) < 0.0005)
        ]
        _all_row = df_dca[
            (df_dca["model"] == "all")
            & (abs(df_dca["threshold"] - _t) < 0.0005)
        ]

        _nb_epic = float(_epic_row["net_benefit"].iloc[0]) if len(_epic_row) > 0 else float("nan")
        _nb_morse = float(_morse_row["net_benefit"].iloc[0]) if len(_morse_row) > 0 else float("nan")
        _nb_all = float(_all_row["net_benefit"].iloc[0]) if len(_all_row) > 0 else float("nan")

        _rows.append(
            {
                "Threshold (%)": f"{_t * 100:.0f}%",
                "Epic PMFRS": round(_nb_epic, 5),
                "Morse Fall Scale": round(_nb_morse, 5),
                "Treat all": round(_nb_all, 5),
                "Treat none": 0.0,
            }
        )

    dca_tbl = pl.DataFrame(_rows)
    mo.md("## Net Benefit at Key Thresholds")
    return (dca_tbl,)


@app.cell
def _(dca_tbl, mo):
    mo.ui.table(dca_tbl)
    return


@app.cell
def _(Path, mo, dca_tbl):
    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _csv_path = _out_dir / "figure3_dca_net_benefit.csv"
    dca_tbl.write_csv(_csv_path)
    mo.md(f"**Saved**: `{_csv_path}` ({dca_tbl.height} rows)")
    return


if __name__ == "__main__":
    app.run()
