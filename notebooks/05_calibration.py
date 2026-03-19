import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from pathlib import Path

    return Path, matplotlib, mo, np, pl, plt


@app.cell
def _(mo):
    mo.md(
        """
        # 05 — Calibration Analysis

        **Purpose**: Assess calibration of Epic PMFRS and Morse Fall Scale after logistic
        recalibration to predicted probabilities. Generates eFigures 1 and 2.

        **Method**:
        1. Logistic recalibration: fit `LogisticRegression(score → y_true)` for each model
        2. Extract `predict_proba` as calibrated probabilities
        3. Compute CITL, calibration slope, and ICI
        4. LOWESS-smoothed calibration curve (frac=0.3) with reference diagonal
        5. Spike/rug plot for predicted probability distribution

        **Outputs**:
        - eFigure 1: Calibration plot — Epic PMFRS
        - eFigure 2: Calibration plot — Morse Fall Scale
        """
    )
    return


# ── Imports: utilities ─────────────────────────────────────────────────────
@app.cell
def _():
    from utils.metrics import calibration_metrics, logistic_recalibration
    from utils.plotting import COLORS, JAMA_STYLE, FIG_SINGLE_COL, save_figure

    return COLORS, FIG_SINGLE_COL, JAMA_STYLE, calibration_metrics, logistic_recalibration, save_figure


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
        """
    )
    return


# ── 2. Extract arrays at sklearn boundary ─────────────────────────────────
@app.cell
def _(df):
    y_true = df["fall_flag"].to_numpy()
    epic_scores = df["epic_score_admission"].to_numpy()
    morse_scores = df["morse_score_admission"].to_numpy()
    return epic_scores, morse_scores, y_true


# ── 3. Logistic recalibration: score → probability ────────────────────────
@app.cell
def _(epic_scores, logistic_recalibration, morse_scores, y_true):
    epic_prob, _ = logistic_recalibration(epic_scores, y_true)
    morse_prob, _ = logistic_recalibration(morse_scores, y_true)
    return epic_prob, morse_prob


# ── 4. Calibration metrics ────────────────────────────────────────────────
@app.cell
def _(calibration_metrics, epic_prob, morse_prob, y_true):
    epic_cal = calibration_metrics(y_true, epic_prob, lowess_frac=0.3)
    morse_cal = calibration_metrics(y_true, morse_prob, lowess_frac=0.3)
    return epic_cal, morse_cal


@app.cell
def _(epic_cal, mo, morse_cal):
    mo.md(
        f"""
        ## Calibration Metrics Summary

        | Metric | Epic PMFRS | Morse Fall Scale |
        |---|---|---|
        | CITL | {epic_cal['citl']:.3f} | {morse_cal['citl']:.3f} |
        | Calibration intercept | {epic_cal['calibration_intercept']:.3f} | {morse_cal['calibration_intercept']:.3f} |
        | Calibration slope | {epic_cal['calibration_slope']:.3f} | {morse_cal['calibration_slope']:.3f} |
        | ICI | {epic_cal['ici']:.4f} | {morse_cal['ici']:.4f} |

        **CITL** = calibration-in-the-large (logit scale); ideal = 0.
        **ICI** = integrated calibration index; ideal = 0.
        **Calibration slope**: ideal = 1.0.
        """
    )
    return


# ── 5. Calibration plot helper ────────────────────────────────────────────
@app.cell
def _(matplotlib, np, plt):
    import statsmodels.api as sm

    def make_calibration_plot(
        y_true_arr,
        y_prob_arr,
        cal_metrics: dict,
        model_label: str,
        line_color: str,
        jama_style: dict,
        fig_size: tuple,
    ):
        """Single-column calibration plot per JAMA specs.

        Returns the Figure (caller is responsible for save/display).
        """
        # LOWESS smoothed calibration curve
        _lowess_result = sm.nonparametric.lowess(
            y_true_arr,
            y_prob_arr,
            frac=0.3,
            it=3,
            return_sorted=True,
        )

        # Data-driven axis limits — zoom to the probability range
        _p_max = float(np.max(np.concatenate([y_prob_arr, _lowess_result[:, 1]])))
        _axis_max = np.ceil(_p_max * 20) / 20  # round up to nearest 0.05
        _axis_max = max(_axis_max, 0.05)        # minimum so axes aren't degenerate
        _axis_max = min(_axis_max, 1.0)          # cap at 1.0

        # Scale rug positions proportionally to axis range
        _rug_y_event = -0.03 * _axis_max
        _rug_y_nonevent = -0.06 * _axis_max
        _rug_height = max(0.018 * _axis_max, 0.003)

        with matplotlib.rc_context(jama_style):
            _fig, _ax = plt.subplots(figsize=fig_size)

            # Reference diagonal covers 0 to _axis_max (not 0 to 1)
            _ref_x = np.linspace(0, _axis_max, 100)
            _ax.plot(
                _ref_x,
                _ref_x,
                color="#888888",
                linewidth=0.8,
                linestyle="--",
                zorder=1,
                label="Perfect calibration",
            )

            # LOWESS calibration curve
            _ax.plot(
                _lowess_result[:, 0],
                _lowess_result[:, 1],
                color=line_color,
                linewidth=1.2,
                zorder=3,
                label=model_label,
            )

            # Spike / rug plot at bottom — show distribution of predicted probabilities
            # Events (y=1) above the axis, non-events below
            _events_mask = y_true_arr == 1
            _nonevents_mask = ~_events_mask

            # Non-events (subsample to 5000 max for render speed)
            _rng = np.random.RandomState(42)
            _non_idx = np.where(_nonevents_mask)[0]
            if len(_non_idx) > 5000:
                _non_idx = _rng.choice(_non_idx, size=5000, replace=False)
            _ax.vlines(
                y_prob_arr[_non_idx],
                ymin=_rug_y_nonevent - _rug_height / 2,
                ymax=_rug_y_nonevent + _rug_height / 2,
                color="#777777",
                linewidth=0.3,
                alpha=0.4,
                zorder=2,
            )

            # Events (all — typically <1000)
            _ax.vlines(
                y_prob_arr[_events_mask],
                ymin=_rug_y_event - _rug_height / 2,
                ymax=_rug_y_event + _rug_height / 2,
                color=line_color,
                linewidth=0.5,
                alpha=0.7,
                zorder=2,
            )

            # Annotation box: calibration metrics (top-right, above data)
            _anno = (
                f"CITL = {cal_metrics['citl']:.2f}\n"
                f"Slope = {cal_metrics['calibration_slope']:.2f}\n"
                f"ICI = {cal_metrics['ici']:.4f}"
            )
            _ax.text(
                0.97,
                0.95,
                _anno,
                transform=_ax.transAxes,
                fontsize=8,
                va="top",
                ha="right",
                linespacing=1.4,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="#cccccc",
                    linewidth=0.5,
                ),
            )

            # Legend: below the chart to avoid overlapping LOWESS curve
            _ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=2,
                fontsize=8,
                frameon=False,
                handlelength=1.5,
            )

            _ax.set_xlabel("Predicted probability", fontsize=9)
            _ax.set_ylabel("Observed event rate", fontsize=9)
            _ax.set_title(
                f"Calibration: {model_label}",
                fontsize=10,
                fontweight="bold",
                pad=6,
            )
            _ax.set_xlim(-0.002, _axis_max * 1.05)
            _ax.set_ylim(_rug_y_nonevent - _rug_height, _axis_max * 1.05)
            _ax.tick_params(labelsize=8)

            plt.subplots_adjust(bottom=0.18)

            # Rug labels (8pt minimum)
            _ax.text(
                1.01,
                _rug_y_event,
                "Falls",
                transform=_ax.get_yaxis_transform(),
                fontsize=8,
                va="center",
                ha="left",
                color=line_color,
            )
            _ax.text(
                1.01,
                _rug_y_nonevent,
                "Non-falls",
                transform=_ax.get_yaxis_transform(),
                fontsize=8,
                va="center",
                ha="left",
                color="#777777",
            )

        return _fig

    return make_calibration_plot, sm


# ── 6. eFigure 1: Epic PMFRS calibration ─────────────────────────────────
@app.cell
def _(
    COLORS,
    FIG_SINGLE_COL,
    JAMA_STYLE,
    make_calibration_plot,
    epic_cal,
    epic_prob,
    save_figure,
    y_true,
):
    _efig1 = make_calibration_plot(
        y_true_arr=y_true,
        y_prob_arr=epic_prob,
        cal_metrics=epic_cal,
        model_label="Epic PMFRS",
        line_color=COLORS["epic"],
        jama_style=JAMA_STYLE,
        fig_size=FIG_SINGLE_COL,
    )
    save_figure(_efig1, "efigure1_calibration_epic")
    efigure1 = _efig1
    return (efigure1,)


@app.cell
def _(mo):
    mo.md(
        "**eFigure 1 saved** to `outputs/figures/efigure1_calibration_epic.pdf` and `.png`."
    )
    return


# ── 7. eFigure 2: Morse Fall Scale calibration ────────────────────────────
@app.cell
def _(
    COLORS,
    FIG_SINGLE_COL,
    JAMA_STYLE,
    make_calibration_plot,
    morse_cal,
    morse_prob,
    save_figure,
    y_true,
):
    _efig2 = make_calibration_plot(
        y_true_arr=y_true,
        y_prob_arr=morse_prob,
        cal_metrics=morse_cal,
        model_label="Morse Fall Scale",
        line_color=COLORS["morse"],
        jama_style=JAMA_STYLE,
        fig_size=FIG_SINGLE_COL,
    )
    save_figure(_efig2, "efigure2_calibration_morse")
    efigure2 = _efig2
    return (efigure2,)


@app.cell
def _(mo):
    mo.md(
        "**eFigure 2 saved** to `outputs/figures/efigure2_calibration_morse.pdf` and `.png`."
    )
    return


# ── 8. Calibration summary table ─────────────────────────────────────────
@app.cell
def _(epic_cal, mo, morse_cal, pl):
    _cal_rows = [
        {
            "Model": "Epic PMFRS",
            "CITL": round(epic_cal["citl"], 3),
            "Calibration intercept": round(epic_cal["calibration_intercept"], 3),
            "Calibration slope": round(epic_cal["calibration_slope"], 3),
            "ICI": round(epic_cal["ici"], 4),
        },
        {
            "Model": "Morse Fall Scale",
            "CITL": round(morse_cal["citl"], 3),
            "Calibration intercept": round(morse_cal["calibration_intercept"], 3),
            "Calibration slope": round(morse_cal["calibration_slope"], 3),
            "ICI": round(morse_cal["ici"], 4),
        },
    ]
    cal_summary_df = pl.DataFrame(_cal_rows)
    mo.md("## Calibration summary table")
    mo.ui.table(cal_summary_df)
    return (cal_summary_df,)


@app.cell
def _(Path, mo, cal_summary_df):
    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _csv_path = _out_dir / "calibration_summary.csv"
    cal_summary_df.write_csv(_csv_path)
    mo.md(f"**Saved**: `{_csv_path}` ({cal_summary_df.height} rows)")
    return


if __name__ == "__main__":
    app.run()
