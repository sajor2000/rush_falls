import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    return FancyArrowPatch, FancyBboxPatch, Path, mo, pl, plt


@app.cell
def _(mo):
    mo.md("""
    # eFigure 4 — CONSORT-Style Cohort Flow Diagram

    Reconstructs the exclusion sequence step-by-step from the raw Excel export,
    then reads the post-exclusion analytic cohort to verify counts. Produces a
    publication-quality flow diagram per JAMA Network Open figure specifications.

    **Inputs**
    - `data/raw/output_table_v4.xlsx` — raw export (pre-exclusion)
    - `data/processed/analytic.parquet` — complete-case analytic cohort

    **Output**
    - `outputs/figures/efigure4_cohort_flow.pdf` (primary, vector)
    - `outputs/figures/efigure4_cohort_flow.png` (300 DPI backup)
    """)
    return


@app.cell
def _():
    from utils.constants import COLUMN_RENAME_MAP
    from utils.plotting import JAMA_STYLE, save_figure

    return COLUMN_RENAME_MAP, JAMA_STYLE, save_figure


@app.cell
def _(COLUMN_RENAME_MAP, Path, pl):
    _raw_path = Path("data/raw/output_table_v4.xlsx")
    _df_raw = pl.read_excel(_raw_path)

    # Rename columns using the project map (only existing columns)
    _rename = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in _df_raw.columns}
    _df = _df_raw.rename(_rename)
    _remaining = {
        c: c.lower()
        for c in _df.columns
        if c not in _rename.values() and c != c.lower()
    }
    if _remaining:
        _df = _df.rename(_remaining)

    # Parse datetime columns (tolerant — some may already be datetime or date)
    for _col in ["admission_date", "discharge_date", "fall_datetime"]:
        if _col in _df.columns:
            _dtype = _df[_col].dtype
            if _dtype == pl.Date:
                _df = _df.with_columns(pl.col(_col).cast(pl.Datetime).alias(_col))
            elif _dtype == pl.String or _dtype == pl.Utf8:
                _df = _df.with_columns(
                    pl.col(_col).str.to_datetime(strict=False).alias(_col)
                )

    # Derive fall_flag and los_days (mirrors 01_data_discovery.py)
    _df = _df.with_columns(
        [
            pl.when(
                pl.col("fall_datetime").is_not_null()
                | pl.col("unit_fall_occurred").is_not_null()
            )
            .then(1)
            .otherwise(0)
            .alias("fall_flag"),
            (
                (pl.col("discharge_date") - pl.col("admission_date")).dt.total_seconds()
                / 86400
            ).alias("los_days"),
        ]
    )

    # ── Step-by-step exclusion tracking ──────────────────────────────
    n_raw = _df.height
    n_raw_falls = int(_df["fall_flag"].sum())

    # Step 1: exclude age < 18
    _excl_age = _df.filter(pl.col("age") < 18)
    n_excl_age = _excl_age.height
    n_excl_age_falls = int(_excl_age["fall_flag"].sum())
    _df = _df.filter(pl.col("age") >= 18)
    n_after_age = _df.height
    n_after_age_falls = int(_df["fall_flag"].sum())

    # Step 2: exclude missing discharge_department
    _excl_dept = _df.filter(pl.col("discharge_department").is_null())
    n_excl_dept = _excl_dept.height
    n_excl_dept_falls = int(_excl_dept["fall_flag"].sum())
    _df_eligible = _df.filter(pl.col("discharge_department").is_not_null())
    n_eligible = _df_eligible.height
    n_eligible_falls = int(_df_eligible["fall_flag"].sum())

    # Step 4: characterise missing scores (branch — not excluded yet)
    _miss_epic_mask = pl.col("epic_score_admission").is_null()
    _miss_morse_mask = pl.col("morse_score_admission").is_null()

    n_miss_epic = _df_eligible.filter(_miss_epic_mask).height
    n_miss_epic_falls = int(
        _df_eligible.filter(_miss_epic_mask)["fall_flag"].sum()
    )
    n_miss_morse = _df_eligible.filter(_miss_morse_mask).height
    n_miss_morse_falls = int(
        _df_eligible.filter(_miss_morse_mask)["fall_flag"].sum()
    )

    # Step 5: complete-case cohort (both scores present)
    _df_analytic = _df_eligible.filter(
        pl.col("epic_score_admission").is_not_null()
        & pl.col("morse_score_admission").is_not_null()
    )
    n_analytic = _df_analytic.height
    n_analytic_falls = int(_df_analytic["fall_flag"].sum())
    n_analytic_nofall = n_analytic - n_analytic_falls

    # Pack all counts into a single dict for downstream cells
    flow_counts = {
        "n_raw": n_raw,
        "n_raw_falls": n_raw_falls,
        "n_excl_age": n_excl_age,
        "n_excl_age_falls": n_excl_age_falls,
        "n_after_age": n_after_age,
        "n_after_age_falls": n_after_age_falls,
        "n_excl_dept": n_excl_dept,
        "n_excl_dept_falls": n_excl_dept_falls,
        "n_eligible": n_eligible,
        "n_eligible_falls": n_eligible_falls,
        "n_miss_epic": n_miss_epic,
        "n_miss_epic_falls": n_miss_epic_falls,
        "n_miss_morse": n_miss_morse,
        "n_miss_morse_falls": n_miss_morse_falls,
        "n_analytic": n_analytic,
        "n_analytic_falls": n_analytic_falls,
        "n_analytic_nofall": n_analytic_nofall,
    }
    return (flow_counts,)


@app.cell
def _(Path, flow_counts, mo, pl):
    _parquet = Path("data/processed/analytic.parquet")
    _check_lines = []

    if _parquet.exists():
        _df_saved = pl.read_parquet(_parquet)
        _saved_n = _df_saved.height
        _saved_falls = int(_df_saved["fall_flag"].sum())
        _match_n = _saved_n == flow_counts["n_analytic"]
        _match_f = _saved_falls == flow_counts["n_analytic_falls"]
        _check_lines.append(
            f"{'PASS' if _match_n else 'WARN'}: "
            f"Row count — reconstructed {flow_counts['n_analytic']:,} "
            f"vs saved {_saved_n:,}"
        )
        _check_lines.append(
            f"{'PASS' if _match_f else 'WARN'}: "
            f"Fall count — reconstructed {flow_counts['n_analytic_falls']:,} "
            f"vs saved {_saved_falls:,}"
        )
    else:
        _check_lines.append(
            "INFO: analytic.parquet not found — run 01_data_discovery.py first"
        )

    mo.md(
        "## Cross-Check: Reconstructed vs Saved Parquet\n\n"
        + "\n".join(f"- {ln}" for ln in _check_lines)
    )
    return


@app.cell
def _(flow_counts, mo, pl):
    _rows = [
        {"Step": "All encounters (raw)", "N": flow_counts["n_raw"], "Falls": flow_counts["n_raw_falls"]},
        {"Step": "Excluded: age < 18", "N": flow_counts["n_excl_age"], "Falls": flow_counts["n_excl_age_falls"]},
        {"Step": "After age exclusion", "N": flow_counts["n_after_age"], "Falls": flow_counts["n_after_age_falls"]},
        {"Step": "Excluded: missing discharge dept", "N": flow_counts["n_excl_dept"], "Falls": flow_counts["n_excl_dept_falls"]},
        {"Step": "Eligible encounters", "N": flow_counts["n_eligible"], "Falls": flow_counts["n_eligible_falls"]},
        {"Step": "Missing Epic score (branch)", "N": flow_counts["n_miss_epic"], "Falls": flow_counts["n_miss_epic_falls"]},
        {"Step": "Missing Morse score (branch)", "N": flow_counts["n_miss_morse"], "Falls": flow_counts["n_miss_morse_falls"]},
        {"Step": "Analytic cohort (complete case)", "N": flow_counts["n_analytic"], "Falls": flow_counts["n_analytic_falls"]},
        {"Step": "  — Fall encounters", "N": flow_counts["n_analytic_falls"], "Falls": flow_counts["n_analytic_falls"]},
        {"Step": "  — No-fall encounters", "N": flow_counts["n_analytic_nofall"], "Falls": 0},
    ]
    summary_table = pl.DataFrame(_rows)
    mo.md("## Cohort Flow Summary")
    return (summary_table,)


@app.cell
def _(mo, summary_table):
    mo.ui.table(summary_table)
    return


@app.cell
def _(
    FancyArrowPatch,
    FancyBboxPatch,
    JAMA_STYLE,
    flow_counts,
    plt,
    save_figure,
):
    with plt.rc_context(JAMA_STYLE):
        # Figure: double-column width (7.0 in), tall for 8 node levels
        _fig, _ax = plt.subplots(figsize=(7.0, 10.0))
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1)
        _ax.axis("off")

        # ── Color palette ──────────────────────────────────────────────
        _BOX_FACE = "#FFFFFF"
        _BOX_EDGE = "#333333"
        _ARROW_CLR = "#333333"
        _EXCL_FACE = "#F5F5F5"
        _SPLIT_FALL = "#FFF0F0"
        _SPLIT_NO = "#EBF0F7"

        # ── Layout constants ──────────────────────────────────────────
        _MX = 0.37          # Main-flow box center x
        _MW = 0.38          # Main-flow box width
        _EX = 0.81          # Exclusion box center x
        _EW = 0.30          # Exclusion box width
        _BH = 0.068         # Standard box height
        _SH = 0.055         # Small exclusion box height

        # Y levels (1 = top, 0 = bottom) — 7 nodes
        _YL = {
            "raw":      0.935,
            "age":      0.820,
            "dept":     0.700,
            "eligible": 0.570,
            "missing":  0.420,
            "analytic": 0.260,
            "split":    0.090,
        }

        # ── Helpers ───────────────────────────────────────────────────
        def _box(xc, yc, w, h, text,
                 face=_BOX_FACE, edge=_BOX_EDGE,
                 fs=9, fw="normal", lw=0.75):
            _patch = FancyBboxPatch(
                (xc - w / 2, yc - h / 2), w, h,
                boxstyle="round,pad=0.008",
                facecolor=face, edgecolor=edge, linewidth=lw,
                transform=_ax.transAxes, zorder=2,
            )
            _ax.add_patch(_patch)
            _ax.text(
                xc, yc, text,
                transform=_ax.transAxes,
                ha="center", va="center",
                fontsize=fs, fontweight=fw,
                linespacing=1.35, zorder=3,
            )

        def _arrow(xs, ys, xe, ye, cs="arc3,rad=0.0", lw=0.75):
            _ax.add_patch(FancyArrowPatch(
                (xs, ys), (xe, ye),
                arrowstyle="-|>", mutation_scale=9,
                color=_ARROW_CLR, linewidth=lw,
                connectionstyle=cs,
                transform=_ax.transAxes, zorder=1,
            ))

        def _excl_arrow(y):
            _arrow(_MX + _MW / 2, y, _EX - _EW / 2, y)

        def _vert(y_top_level, y_bot_level, h_top=_BH, h_bot=_BH):
            _arrow(_MX, _YL[y_top_level] - h_top / 2 - 0.003,
                   _MX, _YL[y_bot_level] + h_bot / 2 + 0.003)

        # ── Node 1: All encounters ─────────────────────────────────────
        _box(
            _MX, _YL["raw"], _MW, _BH,
            f"All encounters in study period\n"
            f"(N\u202f=\u202f{flow_counts['n_raw']:,}; "
            f"{flow_counts['n_raw_falls']:,} falls)",
            fs=9, fw="bold",
        )

        # ── Node 2: After age exclusion ────────────────────────────────
        _vert("raw", "age")
        _box(
            _MX, _YL["age"], _MW, _BH,
            f"Age \u226518 years\n"
            f"(n\u202f=\u202f{flow_counts['n_after_age']:,}; "
            f"{flow_counts['n_after_age_falls']:,} falls)",
            fs=9,
        )
        _box(
            _EX, _YL["age"], _EW, _SH,
            f"Excluded: age <18\n"
            f"(n\u202f=\u202f{flow_counts['n_excl_age']:,})",
            face=_EXCL_FACE, fs=8,
        )
        _excl_arrow(_YL["age"])

        # ── Node 3: After discharge-dept filter ────────────────────────
        _vert("age", "dept")
        _box(
            _MX, _YL["dept"], _MW, _BH,
            f"Discharge department recorded\n"
            f"(n\u202f=\u202f{flow_counts['n_eligible']:,}; "
            f"{flow_counts['n_eligible_falls']:,} falls)",
            fs=9,
        )
        _box(
            _EX, _YL["dept"], _EW, _SH,
            f"Excluded: missing discharge\n"
            f"department (n\u202f=\u202f{flow_counts['n_excl_dept']:,})",
            face=_EXCL_FACE, fs=8,
        )
        _excl_arrow(_YL["dept"])

        # ── Node 5: Eligible cohort ────────────────────────────────────
        _vert("dept", "eligible")
        _box(
            _MX, _YL["eligible"], _MW, _BH + 0.008,
            f"Eligible encounters\n"
            f"(N\u202f=\u202f{flow_counts['n_eligible']:,}; "
            f"{flow_counts['n_eligible_falls']:,} falls [{flow_counts['n_eligible_falls'] / flow_counts['n_eligible'] * 100:.1f}%])",
            fs=9, fw="bold", edge="#2166AC", lw=1.2,
        )

        # ── Node 6a/6b: Missing score branches ────────────────────────
        _MISS_LX = 0.195
        _MISS_RX = 0.565
        _MISS_W = 0.295
        _miss_bh = _BH + 0.005

        _box(
            _MISS_LX, _YL["missing"], _MISS_W, _miss_bh,
            f"Missing Epic PMFRS\nadmission score\n"
            f"(n\u202f=\u202f{flow_counts['n_miss_epic']:,}; "
            f"{flow_counts['n_miss_epic_falls']:,} falls)",
            face=_EXCL_FACE, fs=8,
        )
        _box(
            _MISS_RX, _YL["missing"], _MISS_W, _miss_bh,
            f"Missing Morse Fall Scale\nadmission score\n"
            f"(n\u202f=\u202f{flow_counts['n_miss_morse']:,}; "
            f"{flow_counts['n_miss_morse_falls']:,} fall{'s' if flow_counts['n_miss_morse_falls'] != 1 else ''})",
            face=_EXCL_FACE, fs=8,
        )

        # Arrows: eligible → missing boxes (diverge from bottom of node 5)
        _elig_bot = _YL["eligible"] - (_BH + 0.008) / 2 - 0.004
        _miss_top = _YL["missing"] + _miss_bh / 2 + 0.004
        _arrow(_MX - 0.06, _elig_bot, _MISS_LX, _miss_top, cs="arc3,rad=0.12")
        _arrow(_MX + 0.06, _elig_bot, _MISS_RX, _miss_top, cs="arc3,rad=-0.12")

        # Label above missing boxes
        _ax.text(
            _MX, (_YL["eligible"] + _YL["missing"]) / 2 + 0.025,
            "Excluded for missing scores (complete-case analysis)",
            transform=_ax.transAxes,
            ha="center", va="center", fontsize=8,
            fontstyle="italic", color="#555555",
        )

        # ── Node 7: Analytic (complete-case) cohort ────────────────────
        _analytic_bh = _BH + 0.012
        _box(
            _MX, _YL["analytic"], _MW + 0.02, _analytic_bh,
            f"Analytic cohort (complete case)\n"
            f"(N\u202f=\u202f{flow_counts['n_analytic']:,}; "
            f"{flow_counts['n_analytic_falls']:,} falls "
            f"[{flow_counts['n_analytic_falls'] / flow_counts['n_analytic'] * 100:.1f}%])",
            fs=10, fw="bold", edge="#2166AC", lw=1.4,
        )

        # Arrows: missing boxes → analytic cohort (converge)
        _miss_bot = _YL["missing"] - _miss_bh / 2 - 0.004
        _analy_top = _YL["analytic"] + _analytic_bh / 2 + 0.004
        _arrow(_MISS_LX, _miss_bot, _MX - 0.06, _analy_top, cs="arc3,rad=-0.12")
        _arrow(_MISS_RX, _miss_bot, _MX + 0.06, _analy_top, cs="arc3,rad=0.12")

        _ax.text(
            _MX, (_YL["missing"] + _YL["analytic"]) / 2 - 0.01,
            "Both scores present",
            transform=_ax.transAxes,
            ha="center", va="center", fontsize=8,
            fontstyle="italic", color="#555555",
        )

        # ── Node 8a/8b: Fall vs No-fall split ─────────────────────────
        _FALL_X = 0.195
        _NOFALL_X = 0.565
        _SPLIT_W = 0.295

        _box(
            _FALL_X, _YL["split"], _SPLIT_W, _BH,
            f"Fall\n(n\u202f=\u202f{flow_counts['n_analytic_falls']:,})",
            face=_SPLIT_FALL, fs=9, fw="bold",
            edge="#B2182B", lw=1.0,
        )
        _box(
            _NOFALL_X, _YL["split"], _SPLIT_W, _BH,
            f"No fall\n(n\u202f=\u202f{flow_counts['n_analytic_nofall']:,})",
            face=_SPLIT_NO, fs=9, fw="bold",
            edge="#2166AC", lw=1.0,
        )

        # Arrows: analytic → split nodes
        _analy_bot = _YL["analytic"] - _analytic_bh / 2 - 0.004
        _split_top = _YL["split"] + _BH / 2 + 0.004
        _arrow(_MX - 0.06, _analy_bot, _FALL_X, _split_top, cs="arc3,rad=0.12")
        _arrow(_MX + 0.06, _analy_bot, _NOFALL_X, _split_top, cs="arc3,rad=-0.12")

        # ── Figure title below (JAMA convention) ───────────────────────
        _fig.text(
            0.5, -0.04,
            "eFigure 4. CONSORT-style cohort flow diagram",
            ha="center", va="top", fontsize=10, fontweight="bold",
        )

        # ── Save figure ───────────────────────────────────────────────
        save_figure(_fig, "efigure4_cohort_flow", formats=("pdf", "png"), bbox_inches=None, pad_inches=0.15)

    efig4 = _fig
    return (efig4,)


@app.cell
def _(efig4, mo):
    mo.md(
        "**Saved**: `outputs/figures/efigure4_cohort_flow.pdf` and `.png`"
    )
    efig4
    return


@app.cell
def _(Path, mo, summary_table):
    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _csv_path = _out_dir / "efigure4_cohort_flow_counts.csv"
    summary_table.write_csv(_csv_path)
    mo.md(f"**Saved**: `{_csv_path}` ({summary_table.height} rows)")
    return


if __name__ == "__main__":
    app.run()
