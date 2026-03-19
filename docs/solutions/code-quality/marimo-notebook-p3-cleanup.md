---
title: "P3 Code Cleanup: Remove Unused Variables, Deduplicate Computation, Fix Legend Placement"
date: 2026-03-18
category: code-quality
tags:
  - marimo
  - refactor
  - dead-code
  - matplotlib
  - jama-style
  - reactive-graph
severity: P3
components:
  - notebooks/07_threshold_analysis.py
  - notebooks/04_primary_analysis.py
  - notebooks/10_sensitivity_analyses.py
symptoms:
  - Unused probability-space ROC arrays computed but never consumed downstream
  - _compute_flag_stats() helper recomputed values already in threshold_summary DataFrame
  - eFigure 5 legends placed inside plot area risking overlap with Morse histogram mode at 125
  - GT cell returned (GT, gt_table2, loc, style) but only gt_table2 used downstream
  - json module returned from cell but never consumed by any downstream cell
root_cause: >
  Incremental notebook development left behind stale return values and helper
  logic after refactors; legend placement defaulted to interior position
  incompatible with multimodal Morse score distribution.
resolution_type: cleanup/refactor
---

# P3 Code Review Cleanup: Marimo Notebook Hygiene

Five P3 "nice-to-have" findings from code review of the Wong et al. refinements
(flag rate, lead time, score distributions). All issues involved dead code, redundant
computation, or JAMA figure spec violations in marimo notebooks.

## Problem

Code review identified these issues across 3 notebooks:

1. **Dead reactive exports** -- variables computed and returned from cells but never consumed downstream, inflating the marimo reactive graph
2. **Redundant computation** -- a helper function recomputed values already stored in an upstream DataFrame
3. **Legend overlap risk** -- `loc="upper right"` inside plot area where Morse histogram has a mode spike at score 125

## Fixes Applied

### Fix 1: Remove unused probability-space ROC arrays

**File**: `notebooks/07_threshold_analysis.py` (ROC curve cell, ~line 191)

The cell computed 4 extra arrays (`fpr_epic_prob`, `tpr_epic_prob`, `fpr_morse_prob`,
`tpr_morse_prob`) that no downstream cell consumed. Also removed `epic_prob` and
`morse_prob` from the cell's function signature since they were only needed for
the deleted computation.

```python
# Before: 8 arrays returned, 4 unused
@app.cell
def _(epic_prob, epic_score, morse_prob, morse_score, roc_curve, y_true):
    fpr_epic, tpr_epic, _ = roc_curve(y_true, epic_score)
    fpr_morse, tpr_morse, _ = roc_curve(y_true, morse_score)
    fpr_epic_prob, tpr_epic_prob, _ = roc_curve(y_true, epic_prob)
    fpr_morse_prob, tpr_morse_prob, _ = roc_curve(y_true, morse_prob)
    return (fpr_epic, fpr_epic_prob, fpr_morse, fpr_morse_prob,
            tpr_epic, tpr_epic_prob, tpr_morse, tpr_morse_prob)

# After: 4 arrays returned, all consumed by eFigure 3
@app.cell
def _(epic_score, morse_score, roc_curve, y_true):
    fpr_epic, tpr_epic, _ = roc_curve(y_true, epic_score)
    fpr_morse, tpr_morse, _ = roc_curve(y_true, morse_score)
    return fpr_epic, fpr_morse, tpr_epic, tpr_morse
```

### Fix 2: Replace `_compute_flag_stats()` with filter on `threshold_summary`

**File**: `notebooks/07_threshold_analysis.py` (flag rate cell, ~line 530)

A ~40-line cell with a helper function recomputed flag rates from scratch using raw
score arrays and threshold constants. The same values already existed in the
`threshold_summary` DataFrame produced by an upstream cell.

```python
# Before: 40 lines, 7 dependencies (epic_score, morse_score, y_true, np, pl, 5 constants)
def _compute_flag_stats(scores, y, threshold, model, label):
    _total = len(scores)
    _flagged = int(np.sum(scores >= threshold))
    ...

# After: 15 lines, 2 dependencies (pl, threshold_summary)
@app.cell
def _(mo, pl, threshold_summary):
    _standard_cutoffs = [
        "Epic >= 35 (3-tier med)",
        "Epic >= 50 (2-tier high)",
        "Epic >= 70 (3-tier high)",
        "MFS >= 25",
        "MFS >= 45",
    ]
    flag_rate_df = threshold_summary.filter(
        pl.col("Method").is_in(_standard_cutoffs)
    ).select(["Model", "Method", "Threshold", "Flag rate, %", "Sensitivity", "TP", "FP"])
    mo.md("## Flag Rate Summary")
    return (flag_rate_df,)
```

Eliminates a second source of truth that could silently diverge from `threshold_summary`.

### Fix 3: Move eFigure 5 legends outside plot area

**File**: `notebooks/07_threshold_analysis.py` (eFigure 5 cell)

Per CLAUDE.md: "legends must NOT block data -- place outside the plot area." The Morse
panel has a mode at score 125, so `loc="upper right"` risks overlap.

```python
# Before: per-axis legends inside plot area
ax_epic.legend(fontsize=8, loc="upper right", frameon=False)
ax_morse.legend(fontsize=8, loc="upper right", frameon=False)

# After: shared figure-level legend below both panels
_handles, _labels = ax_epic.get_legend_handles_labels()
fig_dist.legend(
    _handles, _labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=2, fontsize=8, frameon=False,
)
```

### Fix 4: Trim unused GT return tuple

**File**: `notebooks/04_primary_analysis.py` (line 602)

```python
# Before
return (GT, gt_table2, loc, style)

# After
return (gt_table2,)
```

`GT`, `loc`, `style` are great_tables imports consumed locally -- no downstream cell
needs them. Returning them created false edges in the reactive graph.

### Fix 5: Remove unused `json` return

**File**: `notebooks/10_sensitivity_analyses.py` (line 534)

```python
# Before
return json,

# After
return
```

The `json` stdlib module was imported locally to call `json.dumps()` but was returned
as if it were a cell output. Bare `return` signals a side-effect-only cell.

## Root Cause

Incremental development pattern: cells were written with broad return tuples during
exploration, then the consuming cells were refactored or deleted without pruning the
upstream returns. The flag rate helper was likely written before `threshold_summary`
existed and never consolidated.

## Prevention

### Marimo cell hygiene checklist

1. **Return minimality**: every variable in a cell's `return` tuple must be consumed
   as a named parameter in at least one downstream cell's function signature
2. **Single-computation rule**: no value computed in more than one cell -- if a metric
   exists in an upstream DataFrame, consume it via filter/select, don't recompute
3. **Legend placement**: every `ax.legend()` call uses `bbox_to_anchor` to place the
   legend outside the plot area; verify visually after `save_figure()`
4. **Import discipline**: stdlib modules imported mid-notebook for local use should
   not appear in the return tuple

### Tooling

- `uv run ruff check notebooks/` -- catches `F841` (unused variable) and `F401` (unused import)
- `uv run marimo check notebooks/` -- validates reactive graph integrity
- `make quality` -- runs lint + marimo check + typecheck as a gate before commit

## Verification

All 3 files pass:
- `python3 -c "import ast; ast.parse(open(f).read())"` -- valid Python
- `uv run ruff check` -- zero new violations
- `uv run marimo check` -- no new structural errors (pre-existing warnings only)

## Cross-References

- [Study plan](../../docs/plans/2026-03-18-feat-pmfrs-vs-morse-validation-study-plan.md)
- [JAMA figure specs](../../CLAUDE.md#jama-figure-specifications) -- legend placement rules
- [`utils/plotting.py`](../../utils/plotting.py) -- `JAMA_STYLE`, `save_figure()`, `COLORS`
- [`utils/constants.py`](../../utils/constants.py) -- threshold constants used in flag rate filter
