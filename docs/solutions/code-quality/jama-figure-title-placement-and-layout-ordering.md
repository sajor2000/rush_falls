---
title: "Figure titles placed above plots instead of below per JAMA Network Open convention"
date: 2026-03-19
tags:
  - figures
  - JAMA-style
  - matplotlib
  - layout
  - caption-placement
severity: moderate
status: resolved
modules:
  - notebooks/02_cohort_flow.py
  - notebooks/03b_recalibration.py
  - notebooks/04_primary_analysis.py
  - notebooks/05_calibration.py
  - notebooks/06_decision_curve.py
  - notebooks/07_threshold_analysis.py
  - notebooks/11_dot_plot.py
  - notebooks/13_master_report.py
symptoms:
  - "Figure titles rendered above plots via suptitle() or ax.set_title() instead of below per JAMA Network Open submission requirements"
  - "Panel sub-titles (A, B, C, D) correctly placed above subplots but main figure captions also above"
  - "In 4 NB13 exploratory figures, fig.text() called before tight_layout(), risking layout conflicts"
root_cause: "matplotlib's default title APIs (suptitle, set_title) place text above the figure; JAMA Network Open requires figure captions below the figure, necessitating manual repositioning via fig.text() at negative y-coordinates and reordering tight_layout() calls to precede fig.text()"
---

## Problem

JAMA Network Open manuscript formatting requires figure captions (titles) to appear **below** figures, not above them. All matplotlib figures across the project's 8 Marimo notebooks used `fig.suptitle()` or `ax.set_title()` to place the main figure caption above the plot area. This applied to all 3 main manuscript figures, all 5 supplementary eFigures, the recalibration mapping figure, and 4 exploratory figures in NB13.

A secondary issue was discovered during code review: in 4 NB13-only exploratory figures, `fig.text()` was called before `tight_layout()`. Since `tight_layout()` does not account for `fig.text()` annotations, this ordering could cause layout conflicts.

## Solution

### Root Cause

matplotlib's `fig.suptitle()` and `ax.set_title()` place text above the figure or axes by default. JAMA convention places descriptive captions below the figure. The fix requires switching to `fig.text()` with explicit negative y-coordinates.

### Fix Steps

1. **Remove above-figure title calls**: Delete `fig.suptitle(...)` and figure-level `ax.set_title(...)` calls.

2. **Add below-figure captions using `fig.text()`**:
   ```python
   fig.text(0.5, y_below, "Figure N. Caption text.",
            ha="center", va="top", fontsize=10, fontweight="bold")
   ```

3. **Preserve panel sub-titles**: Keep `ax.set_title()` for multi-panel subplot labels (e.g., "Epic PMFRS", "Morse Fall Scale") — these are panel identifiers, not figure-level captions.

4. **Update shared helper functions**: For helpers like `make_calibration_plot()`, remove the `set_title()` from the helper. The caller adds `fig.text()` after invoking the helper but before `save_figure()`.

5. **Enforce correct call ordering**: Always `tight_layout()` BEFORE `fig.text()`, then `save_figure()` last:
   ```python
   fig.tight_layout()                    # Step 1: finalize layout
   fig.text(0.5, -0.06, "Figure 1...",  # Step 2: add below-figure caption
            ha="center", va="top", fontsize=10, fontweight="bold")
   save_figure(fig, "figure1")           # Step 3: save (bbox_inches="tight" captures fig.text)
   ```

### Y-Offset Reference Table

The negative `y` value must be tuned per figure to avoid colliding with below-plot legends:

| Figure Type | Legend Position | `y` Value |
|---|---|---|
| Simple single-panel, flow diagrams | N/A | `-0.02` |
| Multi-panel with legend at `(0.5, -0.01)` | Below axes | `-0.06` |
| Two-panel with legend at `(0.5, -0.02)` | Below axes | `-0.08` |
| Two-panel with legend at `(0.5, -0.06)`, `bottom=0.28` | Below axes + padding | `-0.16` |
| Dot plot with legend at `(0.5, -0.14)` | Low below axes | `-0.20` |
| Calibration with legend at `(0.5, -0.15)`, `bottom=0.18` | Lowest | `-0.22` |

### Code Example

**Before** (title above — incorrect for JAMA):
```python
fig.suptitle("Figure 1. Discrimination performance.",
             fontsize=10, fontweight="bold", y=1.01)
fig.tight_layout()
save_figure(fig, "figure1")
```

**After** (title below — JAMA-compliant):
```python
fig.tight_layout()
fig.text(0.5, -0.06, "Figure 1. Discrimination performance.",
         ha="center", va="top", fontsize=10, fontweight="bold")
save_figure(fig, "figure1")
```

### Scope of Changes

21 total `fig.text()` calls added across 8 notebooks:

| Notebook | Figures Changed |
|---|---|
| `02_cohort_flow.py` | eFigure 4 |
| `03b_recalibration.py` | Recalibration mapping |
| `04_primary_analysis.py` | Figure 1 |
| `05_calibration.py` | eFigures 1 and 2 (+ helper function update) |
| `06_decision_curve.py` | Figure 3 |
| `07_threshold_analysis.py` | eFigures 3 and 5 |
| `11_dot_plot.py` | Figure 2 |
| `13_master_report.py` | All above + 4 exploratory figures |

## Prevention

### Checklist for new figures

- [ ] Title placed BELOW figure via `fig.text()` at `y < 0`, not above via `suptitle()` or `set_title()`
- [ ] `tight_layout()` or `subplots_adjust()` called BEFORE `fig.text()`
- [ ] `save_figure()` called AFTER `fig.text()` (bbox_inches="tight" captures it)
- [ ] Panel sub-titles ("A", "B", etc.) use `ax.set_title()` — these stay above their subplots
- [ ] Legends placed outside data area with `bbox_to_anchor`
- [ ] Y-offset tuned to clear any below-plot legend
- [ ] Font: Arial, 8pt minimum, 10pt bold for captions
- [ ] Colors: `#2166AC` (Epic), `#B2182B` (Morse)
- [ ] Visual inspection of exported PNG confirms no overlap

### Common mistakes

- **`tight_layout()` after `fig.text()`**: `tight_layout()` ignores `fig.text()` annotations. Always call it first.
- **Forgetting to update NB13**: The master report duplicates all figure code. Changes in individual notebooks must be mirrored in NB13.
- **Using `fig.suptitle()` out of habit**: Default matplotlib patterns put titles above. Always use `fig.text()` for below-figure captions.

### The NB13 sync invariant

Every figure-producing code block in notebooks 02–11 has a corresponding duplicate in `13_master_report.py`. When figure code is modified in any source notebook, the identical modification must be applied to the corresponding block in NB13. Search NB13 by figure filename (e.g., `"figure1_"` or `"efigure3_"`) to find the matching block.

## Related Documentation

- **`docs/solutions/code-quality/marimo-notebook-p3-cleanup.md`** — Established the canonical legend placement pattern (`bbox_to_anchor` outside plot area)
- **`docs/solutions/logic-errors/missing-epic-threshold-annotations-and-optimal-cutoffs.md`** — DCA annotation stagger algorithm for avoiding text collisions
- **`docs/solutions/logic-errors/etable-naming-inconsistency-and-missing-tripod-cells.md`** — Documents JAMA figure compliance pattern in NB01/NB13 exploratory figures
- **`CLAUDE.md` § JAMA Figure Specifications** — Canonical spec: Arial, 8pt min, no top/right spines, legend outside data area
- **`utils/plotting.py`** — `save_figure()` uses `bbox_inches="tight"` and `pad_inches=0.1`, ensuring below-figure `fig.text()` is captured in exports
