---
title: "Add missing Epic thresholds to DCA/reclassification and retain fitted LR models"
date: 2026-03-18
category: logic-errors
tags:
  - decision-curve-analysis
  - reclassification
  - epic-pmfrs
  - morse-fall-scale
  - logistic-regression
  - threshold-analysis
  - bootstrap
  - marimo
  - f-string-bug
  - float-tolerance
severity: moderate
modules_affected:
  - utils/metrics.py
  - notebooks/04_primary_analysis.py
  - notebooks/06_decision_curve.py
  - notebooks/07_threshold_analysis.py
  - notebooks/08_reclassification.py
symptoms:
  - DCA figure missing Epic standard threshold annotations (35/50/70)
  - Reclassification Table 3 missing Epic threshold rows entirely
  - No DCA-derived optimal threshold range extraction
  - No probability equivalents table mapping ordinal cutoffs to calibrated probabilities
  - Fitted logistic regression models discarded preventing reuse
root_cause_type: incomplete-implementation
---

# Missing Epic Threshold Annotations and Rush-Specific Optimal Cutoffs

## Problem Statement

Epic's recommended thresholds (3-tier: 35/70, 2-tier: 50) were absent from two of four threshold-relevant notebooks (`06_decision_curve.py` and `08_reclassification.py`). The DCA figure showed only Morse cutoff annotations. Reclassification Table 3 had only 2 categorical NRI rows (Morse Youden + MFS >=45), with no Epic thresholds at all. No DCA-derived optimal threshold range was extracted. No probability equivalents table existed to show the ordinal-to-probability mapping that makes these thresholds interpretable on the probability scale.

## Root Cause

Two interlocking problems:

1. **Discarded fitted models**: `logistic_recalibration()` returns `(probability_array, fitted_lr_model)`, but notebooks assigned the model to `_` (discarded). Without the fitted `LogisticRegression` objects, there was no way to call `model.predict_proba([[cutoff]])` to convert ordinal score thresholds into probability-scale equivalents.

2. **Missing probability equivalents**: Notebooks 06 (DCA) and 08 (NRI) operate on the probability scale (0-1), not ordinal scores. Epic's standard cutoffs (35, 50, 70 on 0-100) and Morse's (25, 45 on 0-125) need probability translation before they can appear as DCA annotations or NRI classification thresholds.

## Solution

### `utils/metrics.py` -- new `extract_dca_threshold_range()`

Added a utility function that takes dcurves DCA output and finds the threshold range where a model's net benefit exceeds both treat-all and treat-none. Rounds threshold indices to 8 decimals before joining to handle float imprecision.

```python
def extract_dca_threshold_range(df_dca, model_name, ...):
    def _extract_nb(model, name):
        s = df_dca[df_dca["model"] == model].copy()
        s["threshold"] = s["threshold"].round(8)  # float tolerance
        return s.set_index("threshold")["net_benefit"].rename(name)
    # ... join, filter nb_model > nb_all AND > nb_none, return min/max
```

### `notebooks/04_primary_analysis.py` -- retained models + probability equivalents

- Cell 2b now captures fitted models: `epic_prob, epic_lr = logistic_recalibration(...)`
- New cell 6b computes and displays probability equivalents at all 7 cutoffs (Epic 35/50/70, Morse 25/45, both Youden)

### `notebooks/06_decision_curve.py` -- Epic annotations + DCA extraction

- Recalibration cell retains Epic fitted model, computes all 5 probability equivalents
- DCA figure expanded from 2 Morse-only annotations to 5 (2 Morse + 3 Epic) with color-coded lines and staggered labels
- New cell extracts DCA-derived threshold ranges and exports to JSON

### `notebooks/08_reclassification.py` -- expanded categorical NRI

- Added Epic Youden threshold alongside Morse Youden
- Expanded categorical NRI from 2 rows to 6 (Morse Youden, Epic Youden, MFS >=25, MFS >=45, Epic >=35, Epic >=50)
- Bootstrap loop expanded with 4 additional `compute_categorical_nri` calls per iteration
- Bootstrap CIs now use `ALPHA` constant instead of hardcoded 2.5/97.5

## Key Patterns

### Logistic recalibration to probability equivalent

```python
prob, lr = logistic_recalibration(scores, y_true)
prob_at_cutoff = float(lr.predict_proba([[float(CUTOFF)]])[0, 1])
```

Used identically in notebooks 04, 06, and 08. Each notebook fits independently (marimo notebook independence).

### Marimo cell-local vs exported convention

- Models needed downstream: `epic_lr` (no underscore, returned from cell)
- Models used only within same cell: `_epic_lr` (underscore, cell-local)

### DCA annotation stagger

```python
_placed_x = []
for _cutoff_prob, _cutoff_label, _color in _cutoff_annotations:
    _y_pos = _y_max - 0.001
    for _px in _placed_x:
        if abs(_x_pct - _px) < 0.6:
            _y_pos -= _y_range * 0.12
    _placed_x.append(_x_pct)
```

## Bugs Caught in Review

### 1. Runtime f-string bug (Critical)

Ternary inside f-string format spec:
```python
# BUG: Python parses : as format spec separator, ternary becomes invalid format spec
{val:.4f if val is not None else '---'}

# FIX: Pre-format with helper
def _fmt(val, decimals=4):
    return f"{val:.{decimals}f}" if val is not None else "---"
```
Passes `ast.parse` and `marimo check` but crashes at runtime.

### 2. Float tolerance in index joins

`dcurves.dca()` output has float imprecision. Without rounding, `pd.concat(..., axis=1)` drops rows silently. Fix: `.round(8)` before `set_index`.

### 3. Dead code and naming

- `epic_prob_at_70` computed but never used in notebook 08 (Epic >=70 excluded intentionally)
- `_lr_epic` vs `epic_lr` naming inconsistency across notebooks
- Hardcoded `2.5`/`97.5` percentiles instead of `ALPHA`-derived values
- `__import__('numpy')` hack instead of proper marimo cell dependency
- Unused returns (`dca`, `pdf`, `average_precision_score`) polluting reactivity graph
- Non-underscore cell-local variables in display-only cells

## Prevention Strategies

### Missing thresholds
- Add a threshold coverage checklist to `CLAUDE.md` annotating which thresholds each notebook must include
- Consider exporting named threshold collections from `utils/constants.py` (e.g., `ALL_PUBLISHED_THRESHOLDS`)

### Discarded model objects
- Never assign to `_` unless the value is truly disposable. If there is any downstream use for the object, assign a meaningful name.
- Convention in `CLAUDE.md`: "When calling `logistic_recalibration()`, always retain the fitted model"

### f-string format spec bugs
- Never put ternary expressions after `:` inside f-strings
- Pre-compute formatted strings or use helper functions for conditional formatting
- Ensure `make run-all` exercises every notebook end-to-end (catches runtime-only bugs)

### Float join issues
- Never join on raw float columns -- round both sides first
- After merge, assert expected row count to catch silent drops

### Marimo conventions
- Prefix all cell-local variables with `_`
- Never use `__import__()` -- declare dependencies in cell signature
- Only return variables that downstream cells actually consume
- Run `marimo check` in pre-commit and CI

## Related Documents

- `docs/plans/2026-03-18-feat-pmfrs-vs-morse-validation-study-plan.md` -- Master study plan defining all notebook requirements
- `docs/solutions/code-quality/marimo-notebook-p3-cleanup.md` -- Prior cleanup pass on notebooks 04, 07, 10
- `docs/clustered_auroc_methodology.md` -- Statistical methodology for clustering
- `CLAUDE.md` -- Threshold Selection Methods section, Figures and Tables Plan, Critical Methodological Rules
- `utils/constants.py` -- All threshold constants with warning about Epic monitoring vs screening gap
- `utils/metrics.py` -- Shared statistical functions consumed by notebooks 04, 06, 07, 08
- `notebooks/12_report.py` -- Aggregate report consuming outputs from all modified notebooks
