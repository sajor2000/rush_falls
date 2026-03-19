---
title: "feat: Epic PMFRS vs Morse Fall Scale Validation Study"
type: feat
date: 2026-03-18
---

# Epic PMFRS vs Morse Fall Scale Validation Study

## Overview

Single-center retrospective validation of the Epic Predictive Model Fall Risk Score (PMFRS) versus the Morse Fall Scale (MFS) for predicting inpatient falls at Rush University Medical Center. Encounter-level analysis following TRIPOD+AI (2024) reporting standards. Target journal: JAMA Network Open. Blueprint: Wong et al. JAMA Netw Open 2026;9(2):e260181.

**Study type**: TRIPOD Type 3 (external temporal validation) — both models were developed at external institutions/by external vendors.

**Key numbers**: 85,394 encounters, 861 falls (1.0%), 778 unique patients. Complete case after excluding encounters missing either score: ~84,800.

---

## Problem Statement / Motivation

Epic's Predictive Model Fall Risk Score is deployed in Epic EHRs nationwide but lacks independent validation against the established Morse Fall Scale in a head-to-head encounter-level comparison. Clinicians need evidence on whether the algorithmic PMFRS adds discriminative and clinical value over the nurse-assessed MFS to justify workflow integration. JAMA Network Open published the Wong et al. ESM v2 validation as a template for rigorous proprietary model evaluation — this study follows the same framework for fall prediction.

---

## Proposed Solution

Build an 11-notebook Marimo analysis pipeline (Polars + sklearn + statsmodels) that produces all tables, figures, and supplementary materials for a TRIPOD+AI–compliant manuscript. Each notebook is independently runnable (`marimo run`) for reproducibility.

---

## Technical Approach

### Architecture

```
rushfalla/
├── CLAUDE.md                           # Project context
├── pyproject.toml                      # Dependencies (uv/pip)
├── notebooks/
│   ├── 01_data_discovery.py            # Data ingestion, QC, standardization
│   ├── 02_cohort_flow.py              # CONSORT diagram (eFigure 4)
│   ├── 03_table1.py                   # Demographics (Table 1)
│   ├── 04_primary_analysis.py         # AUROC, DeLong, metrics (Table 2, Figure 1)
│   ├── 05_calibration.py             # Calibration plots (eFigures 1-2)
│   ├── 06_decision_curve.py          # DCA (Figure 3)
│   ├── 07_threshold_analysis.py      # Threshold comparison (eFigure 3)
│   ├── 08_reclassification.py        # NRI/IDI (Table 3)
│   ├── 09_fairness_audit.py          # Stratified performance (eTables 1-3)
│   ├── 10_sensitivity_analyses.py    # All score timings (eTable 4)
│   └── 11_dot_plot.py                # AUROC comparison dot plot (Figure 2)
├── utils/
│   ├── metrics.py                     # AUROC, bootstrap, DeLong, NRI/IDI
│   ├── plotting.py                    # JAMA_STYLE, save_figure(), COLORS
│   └── constants.py                   # Thresholds, column maps, labels
├── data/
│   ├── raw/                           # output_table_v4.xlsx (not committed)
│   └── processed/                     # analytic.parquet (Polars output)
└── outputs/
    ├── figures/                        # PDF + PNG per figure
    ├── tables/                         # CSV per table
    └── html/                           # Exported marimo notebooks
```

### Technology Stack

| Component | Tool | Version | Notes |
|---|---|---|---|
| Data manipulation | Polars | ~1.32.x | Lazy eval, `.to_numpy()` at sklearn boundary |
| Notebooks | Marimo | ~0.20.x | Reactive, Git-friendly `.py` files |
| Discrimination | scikit-learn | latest | `roc_auc_score`, `roc_curve`, `precision_recall_curve` |
| DeLong test | Custom (Sun & Xu 2014) | — | Yandex fast DeLong in `utils/metrics.py` |
| DCA | dcurves | latest | **Requires pandas** — convert at boundary |
| GEE | statsmodels | ~0.14.x | **Requires pandas** — convert at boundary |
| Calibration | statsmodels + custom | — | LOWESS, logistic recalibration |
| Tables | great-tables | latest | Polars-native, export PNG/HTML |
| Figures | matplotlib | latest | JAMA style, Arial, vector PDF |
| Bootstrap | Custom | — | Stratified, BCa, 2000 replicates |

### Cross-Framework Data Flow

```
[xlsx] → Polars (all manipulation)
           ├─→ .to_numpy()  → sklearn (AUROC, metrics)
           ├─→ .to_pandas() → dcurves (DCA)
           ├─→ .to_pandas() → statsmodels (GEE, calibration)
           ├─→ GT(polars_df) → great-tables (publication tables)
           └─→ .to_numpy()  → matplotlib (figures)
```

---

## Implementation Phases

### Phase 1: Data Preparation (Notebooks 01-02)

#### 01_data_discovery.py — Data Ingestion, QC, Standardization

**Inputs**: `data/raw/output_table_v4.xlsx`
**Outputs**: `data/processed/analytic.parquet`

Tasks:
- [ ] Read xlsx with Polars (`pl.read_excel()` or via openpyxl → Polars)
- [ ] Standardize all column names per CLAUDE.md mapping (26 columns)
- [ ] Compute derived variables:
  - `fall_flag`: 1 if `fall_datetime` is not null OR `unit_fall_occurred` is not null, else 0
  - `los_days`: `(discharge_date - admission_date).dt.total_seconds() / 86400`
- [ ] **Data quality checks** (critical before any analysis):
  - Verify one row per unique `encounter_csn` (no duplicates)
  - Verify all `fall_datetime` values fall between `admission_date` and `discharge_date`
  - Confirm age range (expect >= 18; flag any pediatric encounters)
  - Profile `accommodation_code` values — define which are "inpatient" vs excluded
  - Verify `epic_score_max >= epic_score_admission` for all encounters (sanity check)
  - Check for encounters where `fall_datetime` < first score timestamp (score timing issue)
- [ ] **Missing data characterization**:
  - Compare demographics of encounters missing Epic scores (n=229) vs complete
  - Compare demographics of encounters missing Morse scores (n=416) vs complete
  - Test whether missingness is associated with fall_flag, age, race, unit (MCAR vs MAR)
  - Document in a missingness profile table for the supplement
- [ ] **Apply exclusions** (sequentially, counting at each step):
  1. Exclude non-adult encounters (age < 18) — count excluded
  2. Exclude observation/outpatient (`accommodation_code` not in inpatient set) — count excluded
  3. Exclude encounters missing discharge_department (per CLAUDE.md) — count excluded
  4. Flag (but do not exclude) encounters missing Epic or Morse admission scores
- [ ] **Define analytic cohorts**:
  - Primary cohort: encounters with BOTH `epic_score_admission` and `morse_score_admission` non-null
  - Epic-only cohort: encounters with `epic_score_admission` non-null
  - Morse-only cohort: encounters with `morse_score_admission` non-null
- [ ] Write `data/processed/analytic.parquet` with all derived variables
- [ ] Display summary statistics with `mo.ui.table()` in Marimo

```python
# Key Polars patterns
df = pl.read_excel("data/raw/output_table_v4.xlsx")
df = df.rename({
    "MRN": "mrn", "EncounterEpicCsn": "encounter_csn",
    "age_at_admission": "age", "Ethnicity": "ethnicity",
    # ... full mapping from CLAUDE.md
})
df = df.with_columns([
    pl.when(pl.col("unit_fall_occurred").is_not_null())
      .then(1).otherwise(0).alias("fall_flag"),
    ((pl.col("discharge_date") - pl.col("admission_date"))
      .dt.total_seconds() / 86400).alias("los_days"),
])
```

#### 02_cohort_flow.py — CONSORT Diagram (eFigure 4)

**Inputs**: `data/processed/analytic.parquet`
**Outputs**: `outputs/figures/efigure4_cohort_flow.pdf`, `.png`

Tasks:
- [ ] Read parquet and compute counts at each exclusion step
- [ ] Render CONSORT-style flow diagram using matplotlib patches/arrows
- [ ] Include these nodes (sequential):
  1. All encounters in study period: N = ?
  2. Excluded: age < 18 (n = ?)
  3. Excluded: observation/outpatient (n = ?)
  4. Excluded: missing discharge department (n = ?)
  5. Remaining: analytic cohort (N = ~85,394)
  6. Split: missing Epic score (n = 229, 0 fallers) / missing Morse score (n = 416, 1 faller)
  7. Complete case cohort: N = ~84,800, events = ~860
  8. Final split: fall (n = ~860) vs no-fall (n = ~83,940)
- [ ] JAMA style: Arial font, minimum 8pt text, no overlap
- [ ] Save via `save_figure()` (PDF + PNG)

### Phase 2: Descriptives (Notebook 03)

#### 03_table1.py — Patient Characteristics (Table 1)

**Inputs**: `data/processed/analytic.parquet`
**Outputs**: `outputs/tables/table1.csv`, great-tables HTML/PNG

Tasks:
- [ ] Stratify by `fall_flag` (0 vs 1)
- [ ] Report per JAMA Table 1 conventions:
  - **Continuous**: Age (mean ± SD), LOS (median [IQR])
  - **Categorical**: Gender, race, ethnicity — n (%)
  - **Scores**: Epic admission (mean ± SD, median [IQR]), Morse admission (mean ± SD, median [IQR])
  - **Scores**: Epic max, mean; Morse max, mean — all with mean ± SD
- [ ] Compute standardized mean differences (SMD) for each variable
- [ ] Add p-values: t-test or Wilcoxon for continuous, chi-square for categorical
- [ ] Render with great-tables (`GT(polars_df)`) for publication quality
- [ ] Export as CSV for manuscript submission

```python
from great_tables import GT, md, html
gt_table = (
    GT(table1_df)
    .tab_header(title="Table 1. Patient Characteristics by Fall Status")
    .tab_spanner(label="Fall (n=861)", columns=["fall_n", "fall_pct"])
    .tab_spanner(label="No Fall (n=84,533)", columns=["nofall_n", "nofall_pct"])
    .fmt_number(columns=["smd"], decimals=3)
)
```

### Phase 3: Primary Analysis (Notebooks 04-06)

#### 04_primary_analysis.py — Discrimination (Table 2, Figure 1)

**Inputs**: `data/processed/analytic.parquet`
**Outputs**: `outputs/figures/figure1_discrimination.pdf`, `outputs/tables/table2.csv`

Tasks:
- [ ] **Primary metric: AUROC** with DeLong 95% CIs for each model
  - Use custom fast DeLong implementation (Sun & Xu 2014) in `utils/metrics.py`
  - `y_true = df["fall_flag"].to_numpy()`
  - `epic_scores = df["epic_score_admission"].to_numpy()`
  - `morse_scores = df["morse_score_admission"].to_numpy()`
- [ ] **Paired AUROC comparison** via DeLong test (H0: AUROC_epic == AUROC_morse)
  - Report AUROC difference with 95% CI and p-value
- [ ] **Co-primary metric: AUPRC** (average precision)
  - At 1% prevalence, chance AUPRC = 0.01 — report both models' AUPRC
  - Use `sklearn.metrics.average_precision_score()`
- [ ] **GEE-adjusted comparison** for within-patient clustering:
  - Convert to pandas at the statsmodels boundary
  - Fit GEE: `fall_flag ~ score`, groups=`mrn`, family=Binomial, cov_struct=Exchangeable
  - Extract cluster-robust C-statistic or use cluster-bootstrap DeLong
  - Report both naive and cluster-adjusted AUROC CIs
- [ ] **Classification metrics at multiple thresholds** (Table 2):

| Threshold Method | Epic PMFRS | Morse Fall Scale |
|---|---|---|
| Youden index | Sens, Spec, PPV, NPV, NNE | Sens, Spec, PPV, NPV, NNE |
| Fixed sensitivity = 60% | ... | ... |
| Fixed sensitivity = 80% | ... | ... |
| Standard cutoffs (MFS 25/45, Epic low/med/high) | ... | ... |
| Value-optimizing (Parsons NMB) | ... | ... |
| DCA-derived | ... | ... |

- [ ] **Bootstrap CIs for all metrics**: 2000 stratified resamples, BCa intervals
  - Stratify by `fall_flag` to preserve 1% event ratio
  - Set seed = 42, report seed for reproducibility
- [ ] **Figure 1**: Multi-panel plot (sensitivity, specificity, PPV, NPV across all thresholds)
  - Blue (#2166AC) for Epic, Red (#B2182B) for Morse
  - 95% CI shading from bootstrap
  - Full-width figure (7.0 × 8.0 inches), 4 panels
  - JAMA style: Arial, 8pt body, no top/right spines

```python
# DeLong paired test
from utils.metrics import delong_roc_test, delong_roc_variance
p_value = delong_roc_test(y_true, epic_scores, morse_scores)
auc_epic, var_epic = delong_roc_variance(y_true, epic_scores)
auc_morse, var_morse = delong_roc_variance(y_true, morse_scores)
```

#### 05_calibration.py — Calibration Plots (eFigures 1-2)

**Inputs**: `data/processed/analytic.parquet`
**Outputs**: `outputs/figures/efigure1_calibration_epic.pdf`, `efigure2_calibration_morse.pdf`

**Critical methodological decision**: Epic PMFRS and Morse are ordinal scores, NOT predicted probabilities. Calibration requires mapping scores to probabilities.

Tasks:
- [ ] **Score-to-probability mapping**: Two approaches (report both):
  1. **If Epic provides a risk probability** alongside the ordinal score → use directly
  2. **If only ordinal scores available** → Fit logistic recalibration: `logit(P(fall)) = α + β × score`
     - Note: this is inherently well-calibrated by construction (tautological)
     - Alternative: present calibration as **observed fall rate per score category** (deciles or natural categories)
- [ ] **Calibration-in-the-large (CITL)**:
  - `CITL = logit(observed prevalence) - mean(logit(predicted probabilities))`
  - Target: 0. Positive = model underestimates; Negative = model overestimates
- [ ] **Calibration slope** (Cox's method):
  - Regress `fall_flag` on `logit(predicted_prob)` via logistic regression
  - Target: slope = 1, intercept = 0
- [ ] **Integrated Calibration Index (ICI)** with LOWESS:
  - Use `statsmodels.nonparametric.lowess(y_true, y_pred, frac=0.3)`
  - Increase `frac` to 0.3-0.5 for rare events (default 0.2/3 is too local)
- [ ] **Calibration plots** (one per model):
  - LOWESS curve (observed vs predicted)
  - 45-degree reference line
  - **Spike/rug plot** of predicted probability distribution at bottom (critical for showing clustering near zero)
  - Report CITL, slope, ICI on each plot
  - JAMA style, single-column width (3.5 × 2.8 inches)
- [ ] Bootstrap CIs for CITL and calibration slope (2000 replicates)

**Note**: Do NOT use Hosmer-Lemeshow test — it is sensitive to binning and underpowered for rare events (Van Calster et al., BMC Medicine 2019).

#### 06_decision_curve.py — DCA (Figure 3)

**Inputs**: `data/processed/analytic.parquet`
**Outputs**: `outputs/figures/figure3_dca.pdf`

Tasks:
- [ ] Convert scores to predicted probabilities (same mapping as 05_calibration.py)
- [ ] **Convert Polars → Pandas** before calling `dcurves.dca()` (dcurves requires pandas)
- [ ] Run DCA with threshold range 0% to 10% (clinically relevant for 1% event rate):

```python
from dcurves import dca
import numpy as np

df_dca = dca(
    data=pdf,  # pandas DataFrame
    outcome="fall_flag",
    modelnames=["epic_prob", "morse_prob"],
    thresholds=np.arange(0, 0.10, 0.001),
)
```

- [ ] **Custom matplotlib figure** (not `plot_graphs()` — need JAMA styling):
  - Plot net benefit vs threshold for: Epic, Morse, treat-all, treat-none
  - Blue for Epic, Red for Morse, gray for treat-all/treat-none
  - Y-axis: net benefit (will be compressed at 1% prevalence)
  - X-axis: threshold probability (0-10%)
  - Full-width figure (7.0 × 4.5 inches)
- [ ] **Supplementary**: Net benefit table at key thresholds (1%, 2%, 3%, 5%)
- [ ] **Interpretation guidance** in `mo.md()`:
  - At 1% prevalence, treat-all (universal precautions) is a reasonable default
  - The model must show net benefit ABOVE treat-all at clinically plausible thresholds
  - Consider standardized net benefit (NB / prevalence) for clearer visualization

### Phase 4: Secondary Analyses (Notebooks 07-11)

#### 07_threshold_analysis.py — Threshold Comparison (eFigure 3)

**Inputs**: `data/processed/analytic.parquet`
**Outputs**: `outputs/figures/efigure3_threshold_overlay.pdf`

Tasks:
- [ ] Compute optimal thresholds for each method:
  1. **Youden index**: J = max(sensitivity + specificity - 1)
  2. **Closest-to-(0,1)**: min(sqrt((1-sens)² + (1-spec)²))
  3. **Fixed sensitivity at 60%** (per Wong et al.)
  4. **Fixed sensitivity at 80%** (safety-critical target)
  5. **Value-optimizing (Parsons NMB)**: Monte Carlo with cost parameters
     - Cost of fall: $14,000 (sampled: Normal(14000, 3000))
     - Cost of intervention: $200 (sampled: Normal(200, 50))
     - Intervention effectiveness: 30% RR reduction (sampled: Beta(40, 60))
     - QALY loss per fall: 0.0036 (sampled: Normal(0.0036, 0.0005))
     - WTP: $100,000/QALY
     - 1000 Monte Carlo iterations
  6. **DCA-derived**: threshold range where net benefit > treat-all AND > treat-none
  7. **Standard cutoffs**: MFS 25 (moderate), 45 (high); Epic low/medium/high categories
- [ ] Overlay all thresholds on ROC curve for each model
- [ ] Report sensitivity, specificity, PPV, NPV, NNE (Number Needed to Evaluate = 1/PPV) at each threshold
- [ ] Bootstrap CIs for value-optimizing threshold

#### 08_reclassification.py — NRI/IDI (Table 3)

**Inputs**: `data/processed/analytic.parquet`
**Outputs**: `outputs/tables/table3.csv`

Tasks:
- [ ] **Category-free (continuous) NRI**: Implement manually in `utils/metrics.py`
  - Report event NRI and non-event NRI **separately** (Pepe et al. 2015 — combined NRI is misleading)
  - At 1% event rate, non-event NRI will dominate the total
- [ ] **Category-based NRI**: Using clinically meaningful thresholds
  - Need pre-defined risk categories: low / moderate / high for each model
  - If categories unavailable: use Youden threshold as the single category boundary
- [ ] **Integrated Discrimination Improvement (IDI)**:
  - IDI = (mean predicted prob for events: new - ref) - (mean predicted prob for non-events: new - ref)
  - Report IDI_events and IDI_nonevents separately
- [ ] **Bootstrap CIs**: 2000 stratified replicates for all NRI/IDI components
- [ ] **Reclassification table**: Cross-tabulate risk categories (old model vs new model) for events and non-events separately

**Caveat**: NRI is supplementary, not primary. AUROC (DeLong) and DCA are the primary analyses. NRI/IDI add clinical interpretability. Continuous NRI inflates type I error (Pencina et al.) — report but interpret cautiously.

#### 09_fairness_audit.py — Stratified Performance (eTables 1-3)

**Inputs**: `data/processed/analytic.parquet`
**Outputs**: `outputs/tables/etable1_age.csv`, `etable2_race.csv`, `etable3_unit.csv`

Tasks:
- [ ] **Stratification variables**:
  - Age groups: <65, 65-79, ≥80
  - Race/ethnicity: White, Black, Hispanic/Latino, Asian, Other/Unknown
  - Unit type: derived from `admitting_department` (Medical, Surgical, ICU, Other)
  - Gender: Female, Male (add to fairness audit — gender was missing from original spec)
- [ ] **Minimum subgroup size**: Require ≥20 fall events per stratum for AUROC reporting
  - For strata with <20 events: report n, event rate, but flag AUROC as unreliable
  - Consider collapsing small categories (e.g., Asian + Other/Unknown)
- [ ] **Metrics per subgroup** (both Epic and Morse):
  - AUROC with 95% bootstrap CI
  - Sensitivity and specificity at the primary threshold (Youden)
  - Calibration-in-the-large (does the model systematically over/under-predict for this group?)
  - False positive rate (equalized odds assessment)
- [ ] **Heterogeneity test**: Compare AUROCs across subgroups (chi-square test of AUROC homogeneity)
- [ ] **Acknowledge limitations**:
  - No intersectional analysis (age × race combinations have too few events)
  - No socioeconomic variables available (insurance, zip code)
  - No cognitive/functional status variables

#### 10_sensitivity_analyses.py — Score Timing Variations (eTable 4)

**Inputs**: `data/processed/analytic.parquet`
**Outputs**: `outputs/tables/etable4_sensitivity.csv`

Tasks:
- [ ] **Score timing analyses** (repeat primary AUROC + DeLong for each):

| Analysis | Epic Score | Morse Score | Non-faller Comparator | Bias Risk |
|---|---|---|---|---|
| Primary (admission) | `epic_score_admission` | `morse_score_admission` | Same (admission) | None |
| Pre-fall / pre-discharge | `epic_score_before_fall` | `morse_score_before_fall` | **Last score before discharge** | Moderate |
| Max score | `epic_score_max` | `morse_score_max` | Max score during stay | High (post-fall scores?) |
| Mean score | `epic_score_mean` | `morse_score_mean` | Mean score during stay | Moderate (post-fall?) |

- [ ] **Critical concern for max/mean**: Verify whether max and mean scores include post-fall assessments for fallers. If yes, this is serious look-ahead bias — document clearly and interpret cautiously.
- [ ] **First encounter per patient**: Restrict to first encounter per `mrn` to eliminate within-patient correlation entirely (sensitivity analysis for GEE)
- [ ] **LOS stratification**: Short stay (<3 days) vs long stay (≥3 days) — admission scores may be more predictive for short stays
- [ ] **Temporal trend**: If study spans multiple years, report AUROC by year to assess model drift
- [ ] Present all results in a single summary eTable with AUROC, 95% CI, DeLong p-value per comparison

#### 11_dot_plot.py — Model Comparison Dot Plot (Figure 2)

**Inputs**: Results from notebooks 04 and 10
**Outputs**: `outputs/figures/figure2_dot_plot.pdf`

Tasks:
- [ ] Dot plot with AUROC point estimates and 95% CI error bars
- [ ] Y-axis: score timing strategy (admission, pre-fall, max, mean)
- [ ] X-axis: AUROC
- [ ] Two dots per row: blue for Epic, red for Morse
- [ ] Vertical reference line at AUROC = 0.5 (chance)
- [ ] Full-width figure (7.0 × 4.5 inches)
- [ ] JAMA style

---

## Shared Utilities

### utils/metrics.py

- [ ] `fast_delong()` — Sun & Xu (2014) fast DeLong algorithm for paired AUROC comparison
- [ ] `delong_roc_test(y_true, pred_a, pred_b)` → p-value
- [ ] `delong_roc_variance(y_true, pred)` → (AUC, variance)
- [ ] `stratified_bootstrap(y_true, pred_a, pred_b, n_boot=2000, seed=42)` → CIs for all metrics
- [ ] `compute_nri_idi(y_true, prob_ref, prob_new, threshold)` → event NRI, non-event NRI, IDI
- [ ] `value_optimizing_threshold(y_true, y_pred, cost_params)` → optimal threshold via NMB
- [ ] `calibration_metrics(y_true, y_pred)` → CITL, slope, ICI

### utils/plotting.py

- [ ] `JAMA_STYLE` dict (matplotlib rcParams)
- [ ] `COLORS` dict (epic=#2166AC, morse=#B2182B, etc.)
- [ ] `FIG_SINGLE_COL`, `FIG_DOUBLE_COL`, `FIG_MULTI_PANEL` size constants
- [ ] `save_figure(fig, name, formats=("pdf", "png"))` → saves to `outputs/figures/`

### utils/constants.py

- [ ] Column name mappings (raw → standardized)
- [ ] Score threshold definitions (Youden, fixed sens, standard cutoffs)
- [ ] Accommodation code inclusion list
- [ ] Department-to-unit-type mapping

---

## Critical Gaps Identified by Spec Analysis (Must Address)

### Gap 1: Treatment Paradox (MOST IMPORTANT)

**Problem**: High-risk patients likely received fall prevention interventions (bed alarms, sitters, non-slip socks), which reduces their fall rate and attenuates the observed AUROC for both tools. This is the single most important limitation of any retrospective fall risk validation.

**Mitigation**:
- [ ] Discuss treatment paradox prominently in Discussion/Limitations
- [ ] Note the direction of bias: AUROC estimates are conservative (true discrimination is likely higher)
- [ ] If data available: stratify by whether fall prevention interventions were documented
- [ ] Report this as a limitation per TRIPOD+AI Item 25

### Gap 2: Score-to-Probability Conversion for Calibration and DCA

**Problem**: Both PMFRS and MFS produce ordinal/integer scores, not predicted probabilities. DCA and calibration require probabilities.

**Approach**:
- [ ] Check if Epic provides a probability output alongside the PMFRS score
- [ ] If not: fit logistic recalibration (`logit(P) = α + β × score`) on the validation data
  - This is circular for calibration but standard for DCA (Vickers recommends it)
- [ ] For calibration: present as observed fall rate per score decile/category instead of continuous calibration curves
- [ ] Document this limitation explicitly

### Gap 3: Pre-Fall Score Comparator for Non-Fallers

**Problem**: `epic_score_before_fall` and `morse_score_before_fall` exist only for fallers. AUROC needs both cases and controls.

**Decision**: For the pre-fall/pre-discharge sensitivity analysis:
- [ ] Fallers: use `epic_score_before_fall` / `morse_score_before_fall`
- [ ] Non-fallers: use last recorded score before discharge (need to verify these columns exist or derive from source data)
- [ ] If last-before-discharge is not in the dataset: this sensitivity analysis may not be feasible — acknowledge and skip

### Gap 4: Look-Ahead Bias in Max/Mean Scores

**Problem**: For fallers, max and mean scores may include assessments performed AFTER the fall event, inflating scores.

**Mitigation**:
- [ ] Verify whether max/mean are restricted to pre-fall assessments (check with data source)
- [ ] If post-fall scores are included: clearly label as having look-ahead bias in the manuscript
- [ ] Include in sensitivity analysis interpretation but not as primary findings

### Gap 5: AUPRC as Co-Primary Metric

**Problem**: At 1% prevalence, AUROC can be misleadingly high. AUPRC is more informative for rare events.

**Action**: Add `sklearn.metrics.average_precision_score()` to the primary analysis alongside AUROC. Report both. Chance AUPRC ≈ 0.01 (prevalence).

### Gap 6: Missing Data Mechanism

**Problem**: 229 encounters missing Epic (0 fallers), 416 missing Morse (1 faller). Low rates, but TRIPOD+AI requires characterizing the missing data mechanism.

**Action**:
- [ ] In 01_data_discovery.py: compare demographics of complete vs missing encounters
- [ ] Determine if MCAR, MAR, or MNAR
- [ ] If MAR: consider multiple imputation sensitivity analysis (but likely not necessary given <0.5% missing)
- [ ] Report missing data handling in Methods per TRIPOD+AI Item 11

### Gap 7: Multiple Comparisons

**Problem**: Many statistical tests across primary, sensitivity, subgroup, and threshold analyses.

**Action**:
- [ ] Declare the primary analysis (admission scores, AUROC DeLong) with alpha = 0.05 two-sided
- [ ] Label ALL other analyses as exploratory/secondary — no multiplicity adjustment
- [ ] State this explicitly in the Statistical Analysis section
- [ ] JAMA Network Open typically accepts this approach for validation studies

---

## Acceptance Criteria

### Functional Requirements

- [ ] All 11 Marimo notebooks run successfully as `marimo run notebooks/XX_*.py`
- [ ] Analytic parquet file produced with correct column names and derived variables
- [ ] All 3 main figures exported as vector PDF + 300 DPI PNG
- [ ] All 3 main tables exported as CSV
- [ ] All eFigures and eTables produced
- [ ] TRIPOD+AI checklist items mapped to manuscript sections

### Non-Functional Requirements

- [ ] All figures follow JAMA specifications: Arial font, ≥8pt text, no overlap, colorblind-friendly
- [ ] Bootstrap CIs: 2000 stratified replicates, seed=42, BCa where feasible
- [ ] Each figure file < 1 MB
- [ ] Code uses Polars for ALL data manipulation (no pandas except at dcurves/statsmodels boundary)
- [ ] Notebooks are pure Python files (Marimo format), Git-friendly

### Quality Gates

- [ ] Data QC checks pass (no duplicate encounters, valid date ranges, age ≥ 18)
- [ ] AUROC CIs are plausible (not crossing 0.5 for either model)
- [ ] Calibration plots include spike/rug plot showing score distribution
- [ ] DCA shows clinically interpretable threshold range (0-10%)
- [ ] Fairness audit has ≥20 events per reported subgroup AUROC
- [ ] All code reproducible with fixed random seed

---

## Statistical Specifications (Formalized)

| Parameter | Value | Justification |
|---|---|---|
| Alpha (primary) | 0.05, two-sided | Standard for validation studies |
| Bootstrap replicates | 2000 | Adequate for 1% prevalence; 5000 for final submission |
| Bootstrap method | Stratified by `fall_flag`, BCa intervals | Preserve event ratio; bias-corrected |
| Random seed | 42 | Reproducibility |
| GEE family | Binomial (logit link) | Binary outcome |
| GEE correlation | Exchangeable | Within-patient clustering |
| GEE SE | Robust (sandwich) — statsmodels default | Valid regardless of correlation misspecification |
| Min subgroup events | 20 | Below this, AUROC is unreliable |
| DCA threshold range | 0-10% | Clinically relevant for 1% prevalence |
| LOWESS bandwidth | frac=0.3 | Increased for rare events |
| NRI reporting | Event and non-event NRI separately | Combined NRI is misleading (Pepe 2015) |

---

## Dependencies & Prerequisites

### Data Dependencies
- [ ] `output_table_v4.xlsx` in `data/raw/` (exists)
- [ ] Clarify: does the dataset contain Epic probability output or only ordinal scores?
- [ ] Clarify: are max/mean scores restricted to pre-fall assessments for fallers?
- [ ] Clarify: what is the exact definition of "admission score" (first score, score within X hours)?
- [ ] Clarify: study date range (required for TRIPOD+AI Item 5b)

### Python Dependencies
```
marimo[recommended]>=0.20.0
polars>=1.30.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
statsmodels>=0.14.0
scipy>=1.12.0
great-tables>=0.12.0
dcurves>=1.0.0
openpyxl>=3.1.0
pyarrow>=15.0.0
```

---

## Risk Analysis & Mitigation

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Epic PMFRS is proprietary — cannot describe internals | Moderate | High | Frame as limitation per TRIPOD+AI; cite Epic documentation |
| Treatment paradox biases AUROC downward | High | Certain | Discuss in limitations; note conservative estimates |
| Max/mean scores include post-fall assessments | High | Unknown | Verify with data source; label bias if present |
| Subgroup cells too small for AUROC | Moderate | Likely | Min 20 events rule; collapse small categories |
| dcurves requires pandas → Polars conversion friction | Low | Certain | `.to_pandas()` at boundary; document pattern |
| Score-to-probability mapping is circular for calibration | Moderate | Certain | Present as observed rate per score category; discuss limitation |
| 1% event rate makes PPV very low | Low | Certain | Expected; contextualize in Discussion; report NNE |

---

## Future Considerations

- **Prospective validation**: If this retrospective study shows promise, design a prospective implementation study with randomized intervention assignment to address treatment paradox
- **Multi-center expansion**: Partner with other Epic sites for external geographic validation (TRIPOD Type 3)
- **Model updating**: If PMFRS underperforms, consider recalibration or combination models
- **Real-time alerting**: If validated, integrate threshold-based alerts into Epic BestPractice Advisories

---

## References

### Blueprint
- Wong A, et al. JAMA Netw Open. 2026;9(2):e260181.

### Reporting
- Collins GS, et al. BMJ. 2024;385:e078378. (TRIPOD+AI)

### Statistical Methods
- Riley RD, et al. Stat Med. 2021;40(19):4230-4251. (Sample size for validation)
- DeLong ER, et al. Biometrics. 1988;44(3):837-845. (Paired AUROC comparison)
- Sun X, Xu W. Biometrics. 2014;70(3):514-524. (Fast DeLong algorithm)
- Parsons R, et al. JAMIA. 2023;30(6):1103-1113. (Value-optimizing thresholds)
- Van Calster B, et al. BMC Medicine. 2019;17:230. (Calibration for rare events)
- Vickers AJ, et al. Diagn Progn Res. 2019;3:18. (DCA guide)
- Pepe MS, et al. Stat Med. 2015;34(1):110-128. (NRI limitations)

### Fall Prediction
- Choi Y, et al. Am J Health Syst Pharm. 2018;75(17):1293-1303.
- Shim S, et al. Clin Exp Emerg Med. 2022;9(4):345-353.
- Lindberg DS, et al. Int J Med Inform. 2020;143:104272.
- Parsons R, et al. J Med Internet Res. 2024;26:e59634.
- Jiang H, et al. Int Nurs Rev. 2025;72(4):e70110.

### Decision Curve Analysis
- Van Calster B, et al. Eur Urol. 2018;74(6):796-804.
