import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


# ── Cell 1: Title + plain-English intro ────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        # 03b — Why We Convert Scores to Probabilities

        Neither the Epic Predictive Model Fall Risk Score (PMFRS) nor the Morse
        Fall Scale (MFS) outputs a fall *probability*. They are ordinal risk
        indices on completely different scales:

        | Instrument | Range | What the number means |
        |---|---|---|
        | **Epic PMFRS** | 0–100 (continuous float) | Output of an ordinal logistic regression trained on a **3-level target**: (0) no intervention & not high-risk & no fall, (1) high-risk or minor intervention, (2) major intervention or fall. A score of 45 does **not** mean "45% chance of falling." |
        | **Morse Fall Scale** | 0–125 (discrete integer) | Sum of 6 nurse-assessed items — e.g., 25 points for fall history + 20 for IV therapy = 45. The total is an additive checklist, not a probability. |

        **Recalibration** answers the question: *"In our hospital, what is the
        actual probability of falling during this admission for a patient with
        a given score?"*

        This notebook walks through the recalibration step that all downstream
        analyses (notebooks 04–11) rely on. It is designed so that someone
        without biostatistics training can follow along.
        """
    )
    return


# ── Cell 2: Core imports ──────────────────────────────────────────────────
@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    return Path, matplotlib, mo, np, pl, plt


# ── Cell 3: Utility imports ──────────────────────────────────────────────
@app.cell
def _():
    from utils.constants import (
        EPIC_2TIER_HIGH,
        EPIC_3TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        MFS_HIGH,
        MFS_MODERATE,
    )
    from utils.metrics import (
        calibration_metrics,
        logistic_recalibration,
        youden_threshold,
    )
    from utils.plotting import COLORS, FIG_DOUBLE_COL, JAMA_STYLE, save_figure

    return (
        COLORS,
        EPIC_2TIER_HIGH,
        EPIC_3TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        FIG_DOUBLE_COL,
        JAMA_STYLE,
        MFS_HIGH,
        MFS_MODERATE,
        calibration_metrics,
        logistic_recalibration,
        save_figure,
        youden_threshold,
    )


# ── Cell 4: Load data ────────────────────────────────────────────────────
@app.cell
def _(Path, mo, pl):
    df = pl.read_parquet(Path("data/processed/analytic.parquet"))
    _n = df.height
    _n_falls = df.filter(pl.col("fall_flag") == 1).height
    mo.md(
        f"""
        ## Dataset

        - **Encounters**: {_n:,}
        - **Falls**: {_n_falls:,} ({_n_falls / _n * 100:.1f}%)
        """
    )
    return (df,)


@app.cell
def _(df):
    y_true = df["fall_flag"].to_numpy()
    epic_scores = df["epic_score_admission"].to_numpy()
    morse_scores = df["morse_score_admission"].to_numpy()
    return epic_scores, morse_scores, y_true


# ── Cell 5: Raw score distributions ──────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## What do the raw scores look like?

        Epic PMFRS produces a **continuous float** with 61,908 unique values
        (range 0.1–96.7). Morse is a **discrete integer** with only 24 unique
        values (range 0–125). The two instruments live on completely different
        scales.

        The histograms below show how scores are distributed for fallers
        (colored) versus non-fallers (grey). Note that both distributions
        overlap heavily — neither score cleanly separates the two groups.
        """
    )
    return


@app.cell
def _(COLORS, FIG_DOUBLE_COL, JAMA_STYLE, epic_scores, matplotlib, morse_scores, np, plt, y_true):
    _events = y_true == 1
    with matplotlib.rc_context(JAMA_STYLE):
        _fig, (_ax1, _ax2) = plt.subplots(
            1, 2, figsize=(FIG_DOUBLE_COL[0], 3.2)
        )

        # ── Epic histogram ──
        _ax1.hist(
            epic_scores[~_events],
            bins=80,
            color="#cccccc",
            edgecolor="none",
            alpha=0.7,
            label="Non-fallers",
            density=True,
        )
        _ax1.hist(
            epic_scores[_events],
            bins=80,
            color=COLORS["epic"],
            edgecolor="none",
            alpha=0.7,
            label="Fallers",
            density=True,
        )
        _epic_med = float(np.median(epic_scores))
        _epic_p95 = float(np.percentile(epic_scores, 95))
        _ymax1 = _ax1.get_ylim()[1]
        _ax1.axvline(_epic_med, color="black", linewidth=0.8, linestyle="--")
        _ax1.text(
            _epic_med + 1,
            _ymax1 * 0.92,
            f"Median {_epic_med:.1f}",
            fontsize=8,
            va="top",
        )
        _ax1.axvline(_epic_p95, color="black", linewidth=0.8, linestyle=":")
        _ax1.text(
            _epic_p95 + 1,
            _ymax1 * 0.78,
            f"95th pct {_epic_p95:.1f}",
            fontsize=8,
            va="top",
        )
        _ax1.set_xlabel("Epic PMFRS score", fontsize=9)
        _ax1.set_ylabel("Density", fontsize=9)
        _ax1.set_title("Epic PMFRS", fontsize=10, fontweight="bold", pad=6)
        _ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            fontsize=8,
            frameon=False,
        )

        # ── Morse histogram ──
        _morse_unique = np.sort(np.unique(morse_scores))
        _bin_edges = np.append(_morse_unique - 2.5, _morse_unique[-1] + 2.5)

        _ax2.hist(
            morse_scores[~_events],
            bins=_bin_edges,
            color="#cccccc",
            edgecolor="none",
            alpha=0.7,
            label="Non-fallers",
            density=True,
        )
        _ax2.hist(
            morse_scores[_events],
            bins=_bin_edges,
            color=COLORS["morse"],
            edgecolor="none",
            alpha=0.7,
            label="Fallers",
            density=True,
        )
        _morse_med = float(np.median(morse_scores))
        _morse_p95 = float(np.percentile(morse_scores, 95))
        _ymax2 = _ax2.get_ylim()[1]
        _ax2.axvline(_morse_med, color="black", linewidth=0.8, linestyle="--")
        _ax2.text(
            _morse_med + 2,
            _ymax2 * 0.92,
            f"Median {_morse_med:.0f}",
            fontsize=8,
            va="top",
        )
        _ax2.axvline(_morse_p95, color="black", linewidth=0.8, linestyle=":")
        _ax2.text(
            _morse_p95 + 2,
            _ymax2 * 0.78,
            f"95th pct {_morse_p95:.0f}",
            fontsize=8,
            va="top",
        )
        _ax2.set_xlabel("Morse Fall Scale score", fontsize=9)
        _ax2.set_ylabel("Density", fontsize=9)
        _ax2.set_title("Morse Fall Scale", fontsize=10, fontweight="bold", pad=6)
        _ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            fontsize=8,
            frameon=False,
        )

        _fig.subplots_adjust(wspace=0.35, bottom=0.22)

    _fig
    return


# ── Cell 6: Summary statistics table ─────────────────────────────────────
@app.cell
def _(mo):
    mo.md("## Summary statistics by fall status")
    return


@app.cell
def _(df, mo, np, pl):
    def _score_stats(col: str, label: str) -> list[dict]:
        rows = []
        for flag, flag_label in [(0, "Non-fallers"), (1, "Fallers"), (None, "All")]:
            if flag is not None:
                _sub = df.filter(pl.col("fall_flag") == flag)[col].drop_nulls()
            else:
                _sub = df[col].drop_nulls()
            _arr = _sub.to_numpy()
            rows.append(
                {
                    "Score": label,
                    "Group": flag_label,
                    "N": len(_arr),
                    "Mean": round(float(np.mean(_arr)), 1),
                    "SD": round(float(np.std(_arr, ddof=1)), 1),
                    "Median": round(float(np.median(_arr)), 1),
                    "IQR": f"{np.percentile(_arr, 25):.1f}–{np.percentile(_arr, 75):.1f}",
                    "Min": round(float(np.min(_arr)), 1),
                    "Max": round(float(np.max(_arr)), 1),
                    "Unique values": int(_sub.n_unique()),
                }
            )
        return rows

    _rows = _score_stats("epic_score_admission", "Epic PMFRS")
    _rows += _score_stats("morse_score_admission", "Morse Fall Scale")
    _stats_df = pl.DataFrame(_rows)
    mo.ui.table(_stats_df)
    return


# ── Cell 7: Why scores are not probabilities ─────────────────────────────
@app.cell
def _(df, epic_scores, mo, np, pl):
    _prev = df.filter(pl.col("fall_flag") == 1).height / df.height * 100
    _epic_median = float(np.median(epic_scores))
    _pct_ge50 = float(np.sum(epic_scores >= 50) / len(epic_scores) * 100)
    mo.md(
        f"""
        ## Why raw scores are not probabilities

        If the Epic score (0–100) were a probability of falling, then:

        - The **median** should be close to the prevalence (~{_prev:.1f}%), not {_epic_median:.1f}.
        - A score of 50 would mean "50% chance of falling" — but only about
          {_pct_ge50:.1f}% of encounters score ≥50 at admission.

        The Epic model was trained on a **3-level ordinal target** that includes
        nursing interventions and high-risk designations — not just falls. So a
        high score means "this patient looks like the patients who fell **or**
        required major interventions in the training data." It is not
        calibrated to our hospital's fall rate.

        The Morse score is even further from a probability: it is a simple
        **additive checklist** (history of falling = 25, IV therapy = 20,
        weak gait = 10, etc.). A score of 45 means the patient has certain risk
        factors — not that 45% of similar patients fall.

        **Key point**: Both scores correctly **rank** patients by risk (higher
        score = riskier). This ranking ability is what AUROC measures, and it
        does **not** require recalibration. But several downstream analyses
        need actual fall probabilities:

        - **Decision curve analysis (DCA)** — compares net benefit at
          probability thresholds
        - **Net reclassification improvement (NRI)** — measures whether one
          model reclassifies patients into more appropriate risk categories
        - **Calibration assessment** — checks whether predicted probabilities
          match observed event rates
        - **Value-optimizing thresholds** — finds the cutoff that maximizes net
          monetary benefit given costs of falls vs. interventions
        """
    )
    return


# ── Cell 8: Fit recalibration models ─────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## Logistic recalibration

        We fit a simple one-variable logistic regression for each score:

        > **P(fall) = 1 / (1 + exp(-(a + b x score)))**

        This finds the best-fitting S-curve that maps each score to a fall
        probability using our hospital's data. The technique is called
        **logistic recalibration** (Steyerberg 2019, Van Calster 2019).
        """
    )
    return


@app.cell
def _(epic_scores, logistic_recalibration, morse_scores, y_true):
    epic_prob, epic_lr = logistic_recalibration(epic_scores, y_true)
    morse_prob, morse_lr = logistic_recalibration(morse_scores, y_true)
    return epic_lr, epic_prob, morse_lr, morse_prob


# ── Cell 9: Model coefficients ───────────────────────────────────────────
@app.cell
def _(df, epic_lr, mo, morse_lr, np, pl):
    _epic_a = epic_lr.intercept_[0]
    _epic_b = epic_lr.coef_[0][0]
    _morse_a = morse_lr.intercept_[0]
    _morse_b = morse_lr.coef_[0][0]
    _prev = df.filter(pl.col("fall_flag") == 1).height / df.height * 100

    mo.md(
        f"""
        ## Recalibration coefficients

        | Model | Intercept (*a*) | Slope (*b*) | Odds ratio per point |
        |---|---|---|---|
        | Epic PMFRS | {_epic_a:.4f} | {_epic_b:.4f} | {np.exp(_epic_b):.4f} |
        | Morse Fall Scale | {_morse_a:.4f} | {_morse_b:.4f} | {np.exp(_morse_b):.4f} |

        **How to read this:**

        - The **intercept** (*a*) captures the baseline log-odds of falling
          when the score is 0. A large negative value means very low baseline
          risk — as expected, since falls are rare (~{_prev:.1f}%).
        - The **slope** (*b*) is how much the log-odds increase for each
          1-point rise in the score. Positive *b* means higher score → higher
          risk.
        - The **odds ratio** (exp(*b*)) is the multiplicative increase in odds
          per point. For example, an OR of {np.exp(_epic_b):.4f} for Epic means
          each additional Epic point multiplies the odds of falling by
          {np.exp(_epic_b):.4f}. Over 10 points, the odds multiply by
          {np.exp(_epic_b * 10):.2f}.
        """
    )
    return


# ── Cell 10: Score-to-probability mapping curves (KEY FIGURE) ────────────
@app.cell
def _(
    COLORS,
    EPIC_2TIER_HIGH,
    EPIC_3TIER_HIGH,
    EPIC_3TIER_MEDIUM,
    FIG_DOUBLE_COL,
    JAMA_STYLE,
    MFS_HIGH,
    MFS_MODERATE,
    epic_lr,
    epic_scores,
    matplotlib,
    morse_lr,
    morse_scores,
    np,
    plt,
    save_figure,
):
    with matplotlib.rc_context(JAMA_STYLE):
        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE_COL)

        # ── Epic panel ──
        _epic_x = np.linspace(0, 100, 500)
        _epic_y = epic_lr.predict_proba(_epic_x.reshape(-1, 1))[:, 1]

        _ax1.plot(_epic_x, _epic_y * 100, color=COLORS["epic"], linewidth=1.2)

        # Rug plot of actual score distribution
        _rng = np.random.RandomState(42)
        _rug_idx = _rng.choice(len(epic_scores), size=min(5000, len(epic_scores)), replace=False)
        _ax1.plot(
            epic_scores[_rug_idx],
            np.full(len(_rug_idx), -0.15),
            "|",
            color="#999999",
            markersize=2,
            alpha=0.3,
        )

        # Annotate clinical cutoffs (stagger y-offsets to avoid overlap)
        for _cutoff, _label, _ls, _yoff in [
            (EPIC_3TIER_MEDIUM, "35 (3-tier med.)", "--", -30),
            (EPIC_2TIER_HIGH, "50 (2-tier high)", "-.", 12),
            (EPIC_3TIER_HIGH, "70 (3-tier high)", ":", -30),
        ]:
            _p = float(epic_lr.predict_proba([[_cutoff]])[0, 1]) * 100
            _ax1.axvline(_cutoff, color="#888888", linewidth=0.6, linestyle=_ls, alpha=0.7)
            _ax1.plot(
                [0, _cutoff],
                [_p, _p],
                color="#888888",
                linewidth=0.5,
                linestyle=_ls,
                alpha=0.5,
            )
            _ax1.plot(_cutoff, _p, "o", color=COLORS["epic"], markersize=3, zorder=5)
            _ax1.annotate(
                f"{_label}\n{_p:.1f}%",
                xy=(_cutoff, _p),
                xytext=(5, _yoff),
                textcoords="offset points",
                fontsize=8,
                ha="left",
                va="bottom",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", linewidth=0.5),
            )

        _ax1.set_xlabel("Epic PMFRS score", fontsize=9)
        _ax1.set_ylabel("Predicted fall probability, %", fontsize=9)
        _ax1.set_title("Epic PMFRS", fontsize=10, fontweight="bold", pad=6)
        _ax1.set_xlim(-2, 102)
        _ax1.set_ylim(-0.5, max(_epic_y * 100) * 1.15)

        # ── Morse panel ──
        _morse_x = np.linspace(0, 125, 500)
        _morse_y = morse_lr.predict_proba(_morse_x.reshape(-1, 1))[:, 1]

        _ax2.plot(_morse_x, _morse_y * 100, color=COLORS["morse"], linewidth=1.2)

        # Rug plot
        _rug_idx_m = _rng.choice(len(morse_scores), size=min(5000, len(morse_scores)), replace=False)
        _ax2.plot(
            morse_scores[_rug_idx_m],
            np.full(len(_rug_idx_m), -0.08),
            "|",
            color="#999999",
            markersize=2,
            alpha=0.3,
        )

        # Annotate Morse cutoffs
        for _cutoff, _label, _ls, _yoff in [
            (MFS_MODERATE, "25 (moderate)", "--", -28),
            (MFS_HIGH, "45 (high)", "-.", 12),
        ]:
            _p = float(morse_lr.predict_proba([[_cutoff]])[0, 1]) * 100
            _ax2.axvline(_cutoff, color="#888888", linewidth=0.6, linestyle=_ls, alpha=0.7)
            _ax2.plot(
                [0, _cutoff],
                [_p, _p],
                color="#888888",
                linewidth=0.5,
                linestyle=_ls,
                alpha=0.5,
            )
            _ax2.plot(_cutoff, _p, "o", color=COLORS["morse"], markersize=3, zorder=5)
            _ax2.annotate(
                f"{_label}\n{_p:.1f}%",
                xy=(_cutoff, _p),
                xytext=(5, _yoff),
                textcoords="offset points",
                fontsize=8,
                ha="left",
                va="bottom",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", linewidth=0.5),
            )

        _ax2.set_xlabel("Morse Fall Scale score", fontsize=9)
        _ax2.set_ylabel("Predicted fall probability, %", fontsize=9)
        _ax2.set_title("Morse Fall Scale", fontsize=10, fontweight="bold", pad=6)
        _ax2.set_xlim(-3, 128)
        _ax2.set_ylim(-0.25, max(_morse_y * 100) * 1.15)

        _fig.suptitle(
            "Score-to-probability mapping via logistic recalibration",
            fontsize=10,
            fontweight="bold",
            y=1.02,
        )
        _fig.subplots_adjust(wspace=0.35, bottom=0.13)

    save_figure(_fig, "recalibration_mapping")
    _fig
    return


# ── Cell 11: Probability equivalents at clinical cutoffs ─────────────────
@app.cell
def _(
    EPIC_2TIER_HIGH,
    EPIC_3TIER_HIGH,
    EPIC_3TIER_MEDIUM,
    MFS_HIGH,
    MFS_MODERATE,
    epic_lr,
    epic_scores,
    mo,
    morse_lr,
    morse_scores,
    pl,
    y_true,
    youden_threshold,
):
    def _prob_at(lr, score):
        return float(lr.predict_proba([[score]])[0, 1])

    _epic_youden_score = youden_threshold(y_true, epic_scores)
    _morse_youden_score = youden_threshold(y_true, morse_scores)

    _cutoff_rows = [
        {
            "Model": "Morse",
            "Cutoff": f">={MFS_MODERATE} (moderate risk)",
            "Score": MFS_MODERATE,
            "P(fall), %": round(_prob_at(morse_lr, MFS_MODERATE) * 100, 2),
        },
        {
            "Model": "Morse",
            "Cutoff": f">={MFS_HIGH} (high risk)",
            "Score": MFS_HIGH,
            "P(fall), %": round(_prob_at(morse_lr, MFS_HIGH) * 100, 2),
        },
        {
            "Model": "Morse",
            "Cutoff": "Youden optimal",
            "Score": round(_morse_youden_score, 1),
            "P(fall), %": round(_prob_at(morse_lr, _morse_youden_score) * 100, 2),
        },
        {
            "Model": "Epic",
            "Cutoff": f">={EPIC_3TIER_MEDIUM} (3-tier medium)",
            "Score": EPIC_3TIER_MEDIUM,
            "P(fall), %": round(_prob_at(epic_lr, EPIC_3TIER_MEDIUM) * 100, 2),
        },
        {
            "Model": "Epic",
            "Cutoff": f">={EPIC_2TIER_HIGH} (2-tier high)",
            "Score": EPIC_2TIER_HIGH,
            "P(fall), %": round(_prob_at(epic_lr, EPIC_2TIER_HIGH) * 100, 2),
        },
        {
            "Model": "Epic",
            "Cutoff": f">={EPIC_3TIER_HIGH} (3-tier high)",
            "Score": EPIC_3TIER_HIGH,
            "P(fall), %": round(_prob_at(epic_lr, EPIC_3TIER_HIGH) * 100, 2),
        },
        {
            "Model": "Epic",
            "Cutoff": "Youden optimal",
            "Score": round(_epic_youden_score, 1),
            "P(fall), %": round(_prob_at(epic_lr, _epic_youden_score) * 100, 2),
        },
    ]

    _cutoff_df = pl.DataFrame(_cutoff_rows)
    mo.md(
        f"""
        ## Probability equivalents at standard cutoffs

        The table below translates familiar clinical cutoffs into predicted
        fall probabilities in our cohort. For example, a Morse score of {MFS_HIGH}
        (the "high risk" threshold) corresponds to approximately
        {_prob_at(morse_lr, MFS_HIGH) * 100:.1f}% predicted probability of falling.
        """
    )
    mo.ui.table(_cutoff_df)
    return


# ── Cell 12: Recalibrated probability distributions ─────────────────────
@app.cell
def _(df, mo, pl):
    _prev = df.filter(pl.col("fall_flag") == 1).height / df.height * 100
    mo.md(
        f"""
        ## Recalibrated probability distributions

        After recalibration, both models produce predicted probabilities on
        the same scale. This makes direct comparison possible.
        The vertical dashed line marks the cohort prevalence ({_prev:.1f}%).
        """
    )
    return


@app.cell
def _(COLORS, FIG_DOUBLE_COL, JAMA_STYLE, df, epic_prob, matplotlib, morse_prob, np, pl, plt, y_true):
    _events = y_true == 1
    _prev = float(df.filter(pl.col("fall_flag") == 1).height / df.height)
    _pmax = max(float(np.max(epic_prob)), float(np.max(morse_prob)))

    with matplotlib.rc_context(JAMA_STYLE):
        _fig, (_ax1, _ax2) = plt.subplots(
            1, 2, figsize=(FIG_DOUBLE_COL[0], 3.2)
        )

        # ── Epic probabilities ──
        _ax1.hist(
            epic_prob[~_events] * 100,
            bins=60,
            color="#cccccc",
            edgecolor="none",
            alpha=0.7,
            label="Non-fallers",
            density=True,
        )
        _ax1.hist(
            epic_prob[_events] * 100,
            bins=60,
            color=COLORS["epic"],
            edgecolor="none",
            alpha=0.7,
            label="Fallers",
            density=True,
        )
        _ax1.axvline(_prev * 100, color="black", linewidth=0.8, linestyle="--")
        _ax1.text(
            _prev * 100 + 0.2,
            _ax1.get_ylim()[1] * 0.9,
            f"Prevalence\n{_prev * 100:.1f}%",
            fontsize=8,
            va="top",
        )
        _ax1.set_xlabel("Predicted fall probability, %", fontsize=9)
        _ax1.set_ylabel("Density", fontsize=9)
        _ax1.set_title("Epic PMFRS (recalibrated)", fontsize=10, fontweight="bold", pad=6)
        _ax1.set_xlim(0, _pmax * 100 * 1.10)
        _ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            fontsize=8,
            frameon=False,
        )

        # ── Morse probabilities ──
        _ax2.hist(
            morse_prob[~_events] * 100,
            bins=60,
            color="#cccccc",
            edgecolor="none",
            alpha=0.7,
            label="Non-fallers",
            density=True,
        )
        _ax2.hist(
            morse_prob[_events] * 100,
            bins=60,
            color=COLORS["morse"],
            edgecolor="none",
            alpha=0.7,
            label="Fallers",
            density=True,
        )
        _ax2.axvline(_prev * 100, color="black", linewidth=0.8, linestyle="--")
        _ax2.text(
            _prev * 100 + 0.2,
            _ax2.get_ylim()[1] * 0.9,
            f"Prevalence\n{_prev * 100:.1f}%",
            fontsize=8,
            va="top",
        )
        _ax2.set_xlabel("Predicted fall probability, %", fontsize=9)
        _ax2.set_ylabel("Density", fontsize=9)
        _ax2.set_title("Morse Fall Scale (recalibrated)", fontsize=10, fontweight="bold", pad=6)
        _ax2.set_xlim(0, _pmax * 100 * 1.10)
        _ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            fontsize=8,
            frameon=False,
        )

        _fig.subplots_adjust(wspace=0.35, bottom=0.22)

    _fig
    return


# ── Cell 13: AUC preservation check ─────────────────────────────────────
@app.cell
def _(epic_prob, epic_scores, mo, morse_prob, morse_scores, y_true):
    from sklearn.metrics import roc_auc_score as _roc_auc_score

    _auc_epic_raw = _roc_auc_score(y_true, epic_scores)
    _auc_epic_cal = _roc_auc_score(y_true, epic_prob)
    _auc_morse_raw = _roc_auc_score(y_true, morse_scores)
    _auc_morse_cal = _roc_auc_score(y_true, morse_prob)

    mo.md(
        f"""
        ## AUC is unchanged by recalibration

        | Model | AUC (raw score) | AUC (recalibrated) | Difference |
        |---|---|---|---|
        | Epic PMFRS | {_auc_epic_raw:.6f} | {_auc_epic_cal:.6f} | {abs(_auc_epic_raw - _auc_epic_cal):.2e} |
        | Morse Fall Scale | {_auc_morse_raw:.6f} | {_auc_morse_cal:.6f} | {abs(_auc_morse_raw - _auc_morse_cal):.2e} |

        Logistic recalibration is a **monotonic transformation** — it preserves
        the ranking of patients. A patient ranked higher-risk before
        recalibration is still ranked higher after. This is why AUC (which only
        depends on ranking) does not change.

        The recalibration changes the **scale** (from arbitrary points to
        probabilities), not the **ordering**. Any tiny difference above is
        floating-point arithmetic, not a real change.
        """
    )
    return


# ── Cell 14: Calibration metrics ─────────────────────────────────────────
@app.cell
def _(calibration_metrics, epic_prob, mo, morse_prob, y_true):
    epic_cal = calibration_metrics(y_true, epic_prob, lowess_frac=0.3)
    morse_cal = calibration_metrics(y_true, morse_prob, lowess_frac=0.3)

    mo.md(
        f"""
        ## Calibration metrics after recalibration

        | Metric | Epic PMFRS | Morse Fall Scale | Ideal |
        |---|---|---|---|
        | CITL | {epic_cal['citl']:.3f} | {morse_cal['citl']:.3f} | 0 |
        | Calibration slope | {epic_cal['calibration_slope']:.3f} | {morse_cal['calibration_slope']:.3f} | 1 |
        | ICI | {epic_cal['ici']:.4f} | {morse_cal['ici']:.4f} | 0 |

        - **CITL** (calibration-in-the-large): measures the average difference
          between predicted and observed risk on the logit scale. Zero is
          perfect.
        - **Calibration slope**: measures whether predicted probabilities are
          too spread out (slope > 1) or too compressed (slope < 1). A slope of
          1 is perfect.
        - **ICI** (integrated calibration index): the average absolute
          difference between the LOWESS-smoothed observed rate and predicted
          probability. Smaller is better.

        The calibration notebook (05) explores these metrics in more detail.
        """
    )
    return


# ── Cell 15: Key takeaways ──────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## Key takeaways

        1. **Neither score is a probability.** Epic PMFRS is an ordinal logistic
           regression output trained on a 3-level target (no-intervention /
           intervention / fall). Morse is a 6-item additive checklist. Both are
           ordinal risk indices, not calibrated probabilities.

        2. **Logistic recalibration** fits a one-variable logistic regression
           (`score -> P(fall)`) using our hospital's data, converting each raw
           score to a predicted fall probability.

        3. **Discrimination (AUC) is unchanged.** Recalibration is a monotonic
           transformation that preserves patient ranking. The same AUC, the
           same DeLong test, the same bootstrap CIs — nothing about
           discrimination changes.

        4. **Probabilities are required for downstream analyses.** DCA, NRI/IDI,
           calibration plots, and value-optimizing thresholds all need actual
           predicted probabilities, not arbitrary score units.

        5. **All downstream notebooks (04–11) use this same recalibration** via
           `logistic_recalibration()` in `utils/metrics.py`. This notebook
           documents the shared step so every collaborator understands what it
           does and why.
        """
    )
    return


if __name__ == "__main__":
    app.run()
