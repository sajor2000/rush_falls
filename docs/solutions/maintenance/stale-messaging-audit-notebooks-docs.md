---
title: "Stale hardcoded statistics, wrong cross-references, and inconsistent paths across marimo notebooks and CLAUDE.md"
date: 2026-03-18
category: maintenance
component: Documentation/prose consistency across marimo notebooks and project documentation
tags:
  - marimo-notebooks
  - stale-data
  - hardcoded-statistics
  - cross-references
  - path-inconsistency
  - CLAUDE.md
  - prose-audit
severity: moderate
symptoms:
  - Hardcoded event rate "1.0%" in mo.md() cells instead of post-exclusion analytic cohort rate of 1.27%
  - Incorrect notebook cross-references ("notebooks 04-08" omitting notebooks 09-11 that also consume recalibration output)
  - Inconsistent output directory naming between Makefile ("output/") and all other code ("outputs/")
  - CLAUDE.md code organization and execution order sections missing 03b_recalibration.py and 12_report.py
  - Prejudged interpretive language in 12_report.py assuming specific result direction before analysis completion
  - Grammar issues in explanatory prose within notebook markdown cells
related_files:
  - notebooks/12_report.py
  - notebooks/03b_recalibration.py
  - notebooks/04_primary_analysis.py
  - notebooks/08_reclassification.py
  - Makefile
  - CLAUDE.md
---

## Root Cause

Iterative development across 13+ marimo notebooks and multiple sessions caused prose in `CLAUDE.md`, `Makefile`, and notebook markdown cells to fall out of sync with the actual codebase. Three categories of drift:

1. **Stale hardcoded numbers**: The analytic cohort evolved from 85,394 raw encounters (1.0% fall rate) to 63,324 post-exclusion encounters (1.27% fall rate), but early-written prose still referenced the old figures.
2. **Incorrect cross-references and paths**: Notebook cross-references cited "notebooks (04-08)" when downstream consumers actually span 04-11. The `Makefile` and `CLAUDE.md` used `output/` (singular) while the actual directory is `outputs/`.
3. **Missing registry entries**: Notebooks `03b_recalibration.py` and `12_report.py` were added during later sessions but never registered in `CLAUDE.md`'s execution order, code organization tree, or figures/tables plan.

## Solution

Fifteen targeted edits across four files, grouped by category.

### 1. Replace hardcoded statistics with dynamic computation

**`notebooks/12_report.py` line 49** -- prevalence was baked in as a string literal:

```python
# BEFORE
Among **{n_enc:,}** inpatient encounters ({n_falls:,} falls; 1.0% event rate),

# AFTER
Among **{n_enc:,}** inpatient encounters ({n_falls:,} falls; {n_falls/n_enc*100:.1f}% event rate),
```

**`notebooks/12_report.py` lines 59-60** -- interpretive claim prejudged the results:

```python
# BEFORE
Both models demonstrated poor-to-fair discrimination, with the Morse Fall Scale
statistically superior to the Epic PMFRS at admission (p {delong_p}).

# AFTER
The paired DeLong test yielded p {delong_p}, indicating a statistically
significant difference in discrimination between models at admission.
```

### 2. Fix cross-references to downstream notebooks

**`notebooks/03b_recalibration.py`** (two locations, lines 28 and 805):

```python
# BEFORE
"notebooks (04-08)"

# AFTER
"notebooks (04-11)"
```

**`notebooks/03b_recalibration.py` line 774** -- grammar fix:

```python
# BEFORE
"These metrics describe the *calibration* notebooks (05) in more detail"

# AFTER
"The calibration notebook (05) explores these metrics in more detail"
```

**`notebooks/03b_recalibration.py` line 293** -- stale statistic:

```python
# BEFORE
"fewer than 0.5% of encounters ever reach 50"

# AFTER
"only about 1% of encounters score >=50 at admission"
```

### 3. Fix directory paths (`output/` to `outputs/`)

**`Makefile` lines 5-6 and 72-76**:

```makefile
# BEFORE
HTML_DIR := output/html
PDF_DIR  := output/pdf
...
output/md/

# AFTER
HTML_DIR := outputs/html
PDF_DIR  := outputs/pdf
...
outputs/md/
```

**`CLAUDE.md` marimo export example**:

```bash
# BEFORE
uv run marimo export html notebooks/01_data_discovery.py -o output/html/01_data_discovery.html

# AFTER
uv run marimo export html notebooks/01_data_discovery.py -o outputs/html/01_data_discovery.html
```

### 4. Register missing notebooks in CLAUDE.md

**Execution order** -- added 03b after Phase 2 and a new Phase 5:

```bash
# Phase 2: Descriptives (added)
make edit NB=notebooks/03b_recalibration.py

# Phase 5: Aggregation (new)
make edit NB=notebooks/12_report.py
```

**Figures/tables plan** -- added two rows:

| Element | Title | Notebook |
|---|---|---|
| Notebook 03b | Recalibration explainer (no numbered figure; generates `recalibration_mapping`) | `03b_recalibration.py` |
| Notebook 12 | Aggregate report -- collects key results from all notebooks | `12_report.py` |

**Code organization tree** -- added entries under `notebooks/`:

```
notebooks/03b_recalibration.py    # Recalibration explainer + mapping figure
notebooks/12_report.py            # Aggregate report -- collects all results
```

**Threshold methods section** -- added note explaining why Epic >= 70 is omitted from categorical NRI (only 0.4% of encounters score >= 70 at admission, insufficient events for meaningful reclassification).

### Verification

After all edits, three grep sweeps confirmed no remaining instances of the stale patterns:

- No hardcoded `"1.0%"` prevalence in any notebook
- No singular `"output/"` path in Python files or Makefile targets
- No stale `"04-08"` cross-references

Ruff lint passed clean on both modified notebooks.

### Pass 2: Codebase-wide sweep (additional fixes)

A second sweep across all notebooks found and fixed additional instances of the same problem classes, plus issues surfaced by multi-agent code review.

#### Hardcoded statistics made dynamic

| File | Issue | Fix |
|---|---|---|
| `03b_recalibration.py` (3 cells) | Prevalence hardcoded as `~1.3%` and `1.27%` in static `mo.md()` cells | Changed cell signatures to accept `df, pl` (and `epic_scores, np` where needed); prevalence now computed as `df.filter(pl.col("fall_flag") == 1).height / df.height * 100` |
| `03b_recalibration.py` cell 7 | Epic median `5.5` and `>=50` percentage hardcoded | Computed from `np.median(epic_scores)` and `np.sum(epic_scores >= 50) / len(epic_scores) * 100` |
| `04_primary_analysis.py` | `97%` hardcoded in Epic threshold note | Computed inline, then extracted to `_pct_below_35` variable for readability |
| `08_reclassification.py` (3 locations) | `0.4%` hardcoded for Epic >= 70 encounters | Computed as `pct_ge70 = float(np.sum(epic_scores >= 70) / len(epic_scores) * 100)` |
| `12_report.py` interpretation cell | `At 1% fall prevalence` and `DeLong p < 0.001` hardcoded | Made dynamic from `key_results` JSON; removed prejudged interpretive language |

#### Review-surfaced bugs fixed

| File | Issue | Fix |
|---|---|---|
| `12_report.py` Key Findings cell | `TypeError` crash when JSON missing — `n_falls/n_enc*100` on `"N/A"` strings | Changed defaults from `"N/A"` to `0`; added `_event_rate` guard |
| `12_report.py` interpretation cell | Referenced `n_enc`/`n_falls` not exported from Key Findings cell | Added to return tuple; added to interpretation cell's function signature |
| `08_reclassification.py` | `_pct_ge70` duplicated across two cells (sync risk) | Exported `pct_ge70` from Youden cell; great-tables cell consumes it via dependency |

#### Code quality fixes

| File | Issue | Fix |
|---|---|---|
| `12_report.py` | Import block unsorted (ruff I001) | Alphabetized imports |
| `04_primary_analysis.py` | Import block unsorted (ruff I001) | Alphabetized imports |
| `04_primary_analysis.py` | `zip()` without `strict=` (ruff B905) | Added `strict=True` |
| `03b_recalibration.py` | `:.2f` vs `:.1f` inconsistency for prevalence | Standardized to `:.1f` |
| `03b_recalibration.py` | Space before `%` in f-strings (`{val:.1f} %`) | Removed space to match project convention |
| `08_reclassification.py` | `epic_score`/`morse_score` (singular) vs plural convention | Renamed to `epic_scores`/`morse_scores` to match other notebooks |
| `Makefile` clean target | Dead `output/` (singular) directory reference | Removed; only `outputs/` remains |

#### Verification (pass 2)

All six classes of hardcoded values eliminated:
- `~1.3%` / `1.27%` prevalence: 0 remaining
- `poor-to-fair` / `statistically superior`: 0 remaining
- `0.4%` Epic >= 70: 0 remaining
- `97%` Epic < 35: 0 remaining
- `At 1% fall prevalence`: 0 remaining
- `DeLong p < 0.001`: 0 remaining

Ruff lint: **All checks passed** on all 4 modified notebooks (03b, 04, 08, 12).

## Prevention

### 1. Strategies to Prevent Documentation Drift

**Replace hardcoded statistics with computed values.** Never embed computed numbers (cohort counts, prevalence rates, score ranges) directly in notebook prose. Instead, compute values in code and interpolate them with f-strings inside `mo.md()` rather than writing literal numbers. The fix to `12_report.py` (replacing `"1.0%"` with `{n_falls/n_enc*100:.1f}%`) demonstrates the pattern.

**Use parameterized cross-references instead of brittle range descriptions.** Instead of writing "notebooks 04 through 08", reference notebooks by function (e.g., "all downstream analysis notebooks") or by a group name. When the set of notebooks changes, a named group is either still correct or obviously wrong, whereas a numeric range silently becomes stale.

**Avoid premature interpretive claims in infrastructure documents.** Phrases like "Both models demonstrated poor-to-fair discrimination" in a report template are analysis conclusions that belong in the final manuscript, not in code that runs before the analysis. Rule: report notebooks should present numbers; interpretation belongs in the manuscript or in a clearly labeled narrative cell that is conditional on the actual values.

**Enforce consistent directory naming at introduction time.** When a directory convention is established (e.g., `outputs/` with an 's'), grep the entire repo for the old name before the first commit. A pre-commit hook or Makefile target can validate path consistency.

### 2. Best Practices for Multi-Notebook Research Projects

- **Commit notebook additions and CLAUDE.md updates in the same commit** so they cannot diverge.
- **Run `make run-all` before updating documentation** so numbers are current.
- **Treat CLAUDE.md like an API contract**: it describes *what to do* and *how to do it*, not *what the results mean*.
- **Separate stable specifications from volatile observations**: variable dictionaries, tool specs, and method rules rarely change; cohort counts and score profiles change every time the pipeline runs.

### 3. Checklist: Adding a New Notebook

When adding a notebook (e.g., `notebooks/12_new_analysis.py`), update every location below before committing:

```
[ ] 1. Create the notebook file in notebooks/ following the marimo template
      and naming convention (##_descriptive_name.py).

[ ] 2. CLAUDE.md -- "Code Organization" tree
      Add the file to the directory listing under notebooks/ with a
      one-line comment describing its purpose.

[ ] 3. CLAUDE.md -- "Execution Order"
      Add the `make edit NB=notebooks/##_name.py` line in the correct
      phase, maintaining the sequential numbering and phase grouping.

[ ] 4. CLAUDE.md -- "Figures and Tables Plan"
      If the notebook produces any figure or table, add a row to
      either the "Main Manuscript" or "Supplement" table.

[ ] 5. Makefile
      Verify the notebook is captured by the wildcard pattern in
      `run-all` and `export-html` targets.

[ ] 6. Update or remove any numeric ranges in CLAUDE.md or notebook
      prose that reference notebook sequences (e.g., "notebooks 04-07")
      that are now stale due to the insertion.

[ ] 7. Run `make run-all` to confirm the new notebook executes
      successfully in the full pipeline sequence.
```

## Cross-References

- `docs/solutions/logic-errors/missing-epic-threshold-annotations-and-optimal-cutoffs.md` -- Prior fix for missing Epic threshold annotations across notebooks
- `docs/solutions/code-quality/marimo-notebook-p3-cleanup.md` -- Prior cleanup of dead reactive exports and JAMA legend placement
- `docs/solutions/integration-issues/node-script-rewrite-python-docx.md` -- Documents similar `.gitignore` drift prevention
- `CLAUDE.md` -- Threshold Selection Methods, Code Organization, Execution Order, Figures and Tables Plan sections
- `utils/constants.py` -- Centralized threshold constants (lines 137-149)
