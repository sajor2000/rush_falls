---
title: "eTable gender naming unification and NB01/NB13 fall event characterization cells"
date_solved: 2026-03-18
severity: medium
time_to_solve: "2-3 hours"
symptoms:
  - "Inconsistent eTable gender naming across notebooks and DOCX generator (etable4_gender, etable5_gender, eTable 8)"
  - "Missing TRIPOD+AI Item 5b study period summary in NB01"
  - "No faller demographics, fall location, department fall rate, temporal pattern, or admission-to-fall timing analyses in NB01 or NB13"
root_cause: "Incremental development across multiple notebooks caused naming drift for the gender fairness eTable; NB01 and NB13 lacked descriptive epidemiology cells required by the study plan and TRIPOD+AI reporting standards"
components:
  - "notebooks/09_fairness_audit.py"
  - "scripts/generate_docx_tables.py"
  - "notebooks/12_report.py"
  - "notebooks/01_data_discovery.py"
  - "notebooks/13_master_report.py"
tags:
  - naming-consistency
  - data-discovery
  - exploratory-analysis
  - TRIPOD-AI
  - marimo-notebooks
  - JAMA-figures
  - fairness-audit
  - master-report
---

## Problem Description

Two distinct issues were addressed in a single implementation pass.

### Issue 1: Inconsistent eTable naming for gender fairness audit

The gender fairness audit table was referenced by **three different names** across the codebase:

| File | Variable name | CSV filename | Display label |
|---|---|---|---|
| `notebooks/09_fairness_audit.py` | `etable4_gender_df` | `etable4_gender.csv` | "eTable 4 (Bonus)" |
| `notebooks/13_master_report.py` | `etable4_gender_df` | `etable5_gender.csv` | "eTable 5." |
| `scripts/generate_docx_tables.py` | — | `etable4_gender.csv` | "eTable 8" |
| `notebooks/12_report.py` | — | `etable5_gender.csv` | — |

The DOCX generator's "eTable 8" was the canonical number (it follows eTables 1-3 for fairness, eTable 4 for sensitivity, eTables 5-7 for calibration/thresholds/DCA). The other references drifted because each file was written or edited independently.

### Issue 2: Missing fall event characterization (TRIPOD+AI compliance gap)

NB01 (data discovery) and NB13 (master report) lacked descriptive analyses of fall events:

- **No study date range** — required by TRIPOD+AI Item 5b
- **No faller demographics** beyond what Table 1 covers
- **No fall location analysis** (`unit_fall_occurred`)
- **No temporal patterns** (hour-of-day, day-of-week)
- **No fall rate by department**
- **No admission-to-fall timing** distribution

These are standard descriptive epidemiology elements for any clinical validation study.

## Root Cause

**Issue 1:** No single source of truth for supplemental table numbering. Each file author used whatever label seemed correct at the time. The DOCX generator (written last) used the correct numbering, but earlier notebooks were never updated to match.

**Issue 2:** The original notebook pipeline focused on the primary statistical comparison (AUROC, calibration, reclassification). Descriptive epidemiology of fall events was listed in CLAUDE.md as "Known Gaps Requiring Clarification" rather than implemented as analysis cells.

## Solution

### Issue 1 — eTable naming unification

Unified to `etable8_gender` everywhere using `replace_all`:

1. **`notebooks/09_fairness_audit.py`**: `etable4_gender_df` -> `etable8_gender_df` (all occurrences), `etable4_gender.csv` -> `etable8_gender.csv`, section headers and GT title updated to "eTable 8"

2. **`scripts/generate_docx_tables.py`**: `etable4_gender.csv` -> `etable8_gender.csv` (line 387)

3. **`notebooks/12_report.py`**: `etable5_gender.csv` -> `etable8_gender.csv` in `expected_tables` inventory

4. **`notebooks/13_master_report.py`**: `etable4_gender_df` -> `etable8_gender_df`, `etable5_gender.csv` -> `etable8_gender.csv`, tab label "eTable 5." -> "eTable 8.", status text "eTables 1-5" -> "eTables 1-3, 8"

### Issue 2 — Fall event characterization

**NB01 (`notebooks/01_data_discovery.py`)** — Added imports cell + 6 new analysis cells (sections 9-14), renumbered parquet write from section 9 to 15:

| Cell | Content | Output |
|---|---|---|
| Imports | matplotlib, JAMA_STYLE, save_figure | — |
| 9. Study Period | Min/max admission, discharge, fall dates; study duration | Markdown table |
| 10. Faller Demographics | Age mean/SD/median/IQR, gender/race/ethnicity value counts | `mo.ui.table()` |
| 11. Fall Locations | Top 15 units by fall count, horizontal bar chart | `nb01_fall_locations.{pdf,png}` |
| 12. Department Fall Rates | Top 15 departments by volume, fall rate bar chart | `nb01_department_fall_rates.{pdf,png}` |
| 13. Temporal Patterns | 2-panel: hour-of-day histogram + day-of-week bars | `nb01_fall_temporal.{pdf,png}` |
| 14. Admission-to-Fall Timing | Hours-to-fall histogram with median annotation + LOS boxplot | `nb01_admission_to_fall.{pdf,png}`, `nb01_los_comparison.{pdf,png}` |

**NB13 (`notebooks/13_master_report.py`)** — Added 8 new cells under "Section 3b: Fall Event Characterization":

| Cell | Content | Output |
|---|---|---|
| Narrative | Section header + statistical details accordion | — |
| Study Period | `mo.hstack()` stat cards: date range, duration, earliest/latest fall | — |
| Fall Locations | Top 10 units table + bar chart | `fall_locations.{pdf,png}`, `fall_location_summary.csv` |
| Department Rates | Top 10 departments table + bar chart | `department_fall_rates.{pdf,png}`, `department_fall_rates.csv` |
| Temporal Patterns | 2-panel hour/day figure | `fall_temporal.{pdf,png}`, `fall_temporal_hourly.csv`, `fall_temporal_daily.csv` |
| Admission-to-Fall | Histogram with median[IQR] annotation | `admission_to_fall_hours.{pdf,png}` |
| Callout | `mo.callout(kind="info")` with interpretive summary | — |
| Inventory | Updated with 4 new figures + 4 new CSVs | — |

## Key Code Patterns

1. **`replace_all=True` in Edit tool** for global find-and-replace across entire files
2. **Polars for all data manipulation** — `.group_by().agg()`, `.filter()`, `.sort()`, `.with_columns()` — converting to numpy only at the matplotlib boundary
3. **Marimo cell structure** — `@app.cell` with explicit `return (var,)` tuples and proper dependency chaining
4. **JAMA figure compliance** — `mpl.rcParams.update(JAMA_STYLE)`, Arial font, 8pt minimum, no top/right spines, `save_figure()` for dual PDF+PNG export
5. **Temporal extraction** — Polars `.dt.hour()`, `.dt.weekday()`, `.dt.total_seconds()` for temporal pattern analysis

## Verification

1. **Syntax**: `python3 -c "import ast; ast.parse(open(f).read())"` — all 5 files pass
2. **Ruff**: `uv run ruff check` — only pre-existing import sorting warnings in NB09 (not introduced by changes)
3. **Marimo**: `uv run marimo check` — only pre-existing markdown-indentation warnings
4. **Stale references**: `grep -r "etable4_gender\|etable5_gender"` — zero matches across entire codebase

## Prevention Strategies

### For naming drift

1. **Centralize artifact names in `utils/constants.py`** — Define an `OUTPUT_REGISTRY` dict mapping every table/figure to its canonical filename. Every notebook imports from the registry instead of hardcoding strings.

2. **Automated cross-reference validation** — Add a `make quality` target that greps all notebooks and `generate_docx_tables.py` for filename strings, then verifies they all agree with the registry.

3. **Review diffs across all files together** — When a PR touches any artifact name, verify that every file referencing that artifact was updated in the same PR.

### For TRIPOD+AI completeness

1. **Traceability matrix** — Create a structured mapping of every TRIPOD+AI item to a specific notebook and cell. Any row with "TBD" is a known gap.

2. **Completeness check in NB13** — The master report should programmatically verify that every TRIPOD+AI item has a corresponding non-null output.

3. **Phase-gate before manuscript assembly** — Run `make tripod-check` before `generate_docx_tables.py` to catch missing reporting elements.

### Notebook addition checklist

When adding new notebooks or new analysis cells:

```
[ ] 1. Create file in notebooks/
[ ] 2. CLAUDE.md — Code Organization tree
[ ] 3. CLAUDE.md — Execution Order
[ ] 4. CLAUDE.md — Figures and Tables Plan (if produces output)
[ ] 5. Makefile (verify wildcard coverage)
[ ] 6. NB13 inventory (add new figures/tables)
[ ] 7. Remove stale numeric ranges in cross-references
[ ] 8. Run make run-all
```

## Related Documentation

- `docs/solutions/code-quality/marimo-notebook-p3-cleanup.md` — Cell hygiene, reactive graph validation, legend placement fixes
- `docs/solutions/integration-issues/node-script-rewrite-python-docx.md` — DOCX generator architecture, python-docx XML patterns
- `docs/solutions/logic-errors/missing-epic-threshold-annotations-and-optimal-cutoffs.md` — Marimo patterns (never discard model objects), float tolerance in joins
- `docs/solutions/maintenance/stale-messaging-audit-notebooks-docs.md` — Dynamic computation pattern, notebook registry updates, directory naming consistency
- `docs/clustered_auroc_methodology.md` — GEE and cluster bootstrap statistical foundations
