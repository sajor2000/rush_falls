# CLAUDE.md — Epic PMFRS vs Morse Fall Scale Validation Study

## Project Overview

Single-center retrospective validation of the Epic Predictive Model Fall Risk Score (PMFRS) versus the Morse Fall Scale (MFS) for predicting inpatient falls at Rush University Medical Center. Encounter-level analysis guided by TRIPOD+AI (2024) reporting standards.

- **Blueprint paper**: Wong et al. JAMA Netw Open 2026;9(2):e260181 (Epic Sepsis Model v2 multicenter validation)
- **Target journal**: JAMA Network Open
- **Study type**: TRIPOD Type 3 (external temporal validation)
- **Plan file**: `docs/plans/2026-03-18-feat-pmfrs-vs-morse-validation-study-plan.md`

---

## Analytic Dataset Variables

All columns standardized to lowercase. Mapping from raw export (`output_table_v4.xlsx`):

| Raw Column | Standardized | Type | Description |
|---|---|---|---|
| MRN | `mrn` | str | Patient identifier (for clustering) |
| EncounterEpicCsn | `encounter_csn` | str | Encounter identifier (unit of analysis) |
| age_at_admission | `age` | int | Age in years at admission |
| Ethnicity | `ethnicity` | cat | Hispanic/Latino, Not Hispanic/Latino, Unknown |
| race | `race` | cat | White, Black, Asian, Other, Unknown |
| Gender | `gender` | cat | Female, Male |
| accommodation_code | `accommodation_code` | cat | IP, OBS, etc. |
| admitting_department | `admitting_department` | cat | Clinical unit at admission |
| discharge_department | `discharge_department` | cat | Clinical unit at discharge |
| Unit_fall_occurred | `unit_fall_occurred` | cat | Unit where fall happened (null = no fall) |
| admission_date | `admission_date` | datetime | Admission timestamp |
| discharge_date | `discharge_date` | datetime | Discharge timestamp |
| Primary_diagnosis | `primary_diagnosis` | str | Primary ICD/problem |
| FallDateTime | `fall_datetime` | datetime | Timestamp of fall event |
| Epic_score_before_fall | `epic_score_before_fall` | float | Most recent Epic score prior to fall |
| Morse_score_before_fall | `morse_score_before_fall` | float | Most recent Morse score prior to fall |
| epic_score_at_admission | `epic_score_admission` | float | First Epic score after admission |
| morse_score_at_admission | `morse_score_admission` | float | First Morse score after admission |
| Epic_score_max | `epic_score_max` | float | Max Epic score during encounter |
| Epic_score_min | `epic_score_min` | float | Min Epic score during encounter |
| Epic_score_mean | `epic_score_mean` | float | Mean Epic score during encounter |
| Epic_score_median | `epic_score_median` | float | Median Epic score during encounter |
| morse_max | `morse_score_max` | float | Max Morse score during encounter |
| morse_min | `morse_score_min` | float | Min Morse score during encounter |
| morse_mean | `morse_score_mean` | float | Mean Morse score during encounter |
| morse_median | `morse_score_median` | float | Median Morse score during encounter |

**Derived variables** (computed in `01_data_discovery.py`):

| Variable | Definition |
|---|---|
| `fall_flag` | Binary. 1 if `fall_datetime` is not null OR `unit_fall_occurred` is not null, else 0 |
| `los_days` | `(discharge_date - admission_date)` in fractional days |

---

## Cohort Summary

- Total encounters: 85,394
- Fall encounters: 861 (1.0%) from 778 unique patients
- Non-fall encounters: 84,533
- Minimum age: 18
- Missing Epic admission scores: 229 (0.26%), 0 fallers
- Missing Morse admission scores: 416 (0.49%), 1 faller
- All encounters are from inpatient-admitted patients regardless of billing accommodation code (confirmed by data team)
- Complete case cohort (both scores present): exact counts computed dynamically in pipeline
- **Post-exclusion analytic cohort**: computed dynamically after excluding age <18, missing discharge dept, and missing scores

**Epic Score Profile** (admission, analytic cohort):
- dtype: Float64 (continuous, **not** integer despite model brief stating "integer score 0–100")
- Range: 0.117 – 96.706 | 61,908 unique values
- Distribution: extreme right skew — median 5.5, mean 8.7, 75th pct 10.6, 95th pct 27.7, 99th pct 54.3
- 97.0% of encounters score < 35 (Epic "low" 3-tier) at admission
- Fallers: mean 10.4, median 6.2 | Non-fallers: mean 8.7, median 5.5

**Morse Score Profile** (admission, analytic cohort):
- dtype: Int64 (discrete integer), 24 unique values in [0, 125]
- Distribution: multimodal — mode 35 (25.5%), second mode 125 (19.3%), median 50
- Top values: 35 (25.5%), 125 (19.3%), 60 (15.0%), 45 (9.7%)
- Low 0–24: 4,714 (7.4%), 28 falls (0.59%) | Moderate 25–44: 17,028 (26.9%), 102 falls (0.60%) | High ≥45: 41,582 (65.7%), 673 falls (1.62%)

**Critical observation**: Epic's recommended thresholds (3-tier: 35/70; 2-tier: 50) are calibrated for continuous monitoring over a full encounter, NOT admission-time screening. At admission, only 3.0% of encounters exceed Epic "low" (≥35), and the standard cutoffs capture only 4.9% of falls. The Morse ≥45 cutoff captures 83.8% of falls at admission. This asymmetry is a central finding — the two instruments operate on fundamentally different time scales.

---

## Epic PMFRS Model Specifications

Source: Epic Systems. Cognitive Computing Model Brief: Inpatient Risk of Falls. Last updated August 13, 2025.

**Model Architecture**:
- Ordinal logistic regression (not black-box ML)
- Variable selection: LASSO penalized logistic regression (binary target 0 vs 1/2, λ = e⁻⁵, no tuning)
- Final model: ordinal logistic regression fitted via MLE on LASSO-selected variables at each training site
- Output: score 0–100. The model brief describes this as an "integer score", but our dataset contains **continuous float values with 5 decimal places** (e.g. 5.52874, 31.92624). The data team likely exported the raw model output before bedside integer rounding. Treat as a continuous score in [0, 100] for all analyses.
- Does NOT include patient history of falls (deliberate — would dominate LASSO due to correlation with ordinal target that includes interventions)
- Does NOT account for active fall interventions (treatment paradox acknowledged)

**Training Ordinal Target** (used to develop model, not our validation target):

| Target | Description | Prevalence (combined) |
|---|---|---|
| 0 | No intervention, not high risk, no fall | 59.5% |
| 1 | High risk by fall risk program OR minor intervention, no fall | 23.2% |
| 2 | Major intervention AND/OR patient fell | 17.3% |

**Training Data**:
- 3 organizations, ~925,000 encounters, ~2.75M observations (2012–2017)
- 70/30 train/test split
- Site A: 353,268 encounters; Site B: 450,203; Site C: 130,877

**Updated Validation** (August 2024, live customer sites, Jan 2022 – Jul 2023):
- 3 sites (Midwest, Northeast, South), ~1.5M Inpatient/Observation/ED encounters, ~40M scores
- Real production scores (not retroactive), patients aged ≥13
- Fall prevalence: 0.38%–0.54% (95% Wilson CIs)
- Site 1: 206,332 enc, 0.38% falls; Site 2: 322,131 enc, 0.40%; Site 3: 1,050,104 enc, 0.54%

**Epic Model Sensitivity & Flag Rate** (encounter-level, max score before fall):

| Site | Threshold | Sensitivity (95% CI) | Flag Rate (95% CI) |
|---|---|---|---|
| Site 1 | 15 | 86.23% [83.63, 88.47] | 23.86% [23.68, 24.05] |
| Site 2 | 21 | 85.81% [83.72, 87.67] | 31.72% [31.56, 31.88] |
| Site 3 | 7 | 82.12% [81.11, 83.14] | 34.97% [34.88, 35.06] |

**Morse Fall Scale Comparison from Epic Validation** (Sites 1 & 2 only, threshold 45):

| Site | Sensitivity (95% CI) | Flag Rate (95% CI) |
|---|---|---|
| Site 1 | 87.56% [85.05, 89.71] | 23.21% [23.02, 23.41] |
| Site 2 | 80.54% [78.02, 82.83] | 31.04% [31.56, 31.88] |

**Epic Recommended Thresholds**:

| Risk (3-category) | Score Range |
|---|---|
| Low | 0–34 |
| Medium | 35–69 |
| High | 70–100 |

| Risk (2-category) | Score Range |
|---|---|
| Low | 0–49 |
| High | 50–100 |

**WARNING — Threshold applicability**: Epic's validation used **max score before fall** (encounter-level, continuous monitoring). These thresholds are NOT designed for single-timepoint admission screening. In our dataset at admission: 97% score < 35 ("low"), and only 0.4% score ≥ 70 ("high"). Including these thresholds in the analysis is important to demonstrate this gap, but they should not be interpreted as comparable to Morse cutoffs applied at admission.

**Training C-statistics** (original validation, observation-level):

| Site | Target 0 vs (1 or 2) | Target (0 or 1) vs 2 |
|---|---|---|
| Site A | 0.957 (0.957, 0.958) | 0.937 (0.936, 0.939) |
| Site B | 0.828 (0.826, 0.830) | 0.830 (0.825, 0.835) |
| Site C | 0.917 (0.916, 0.918) | 0.938 (0.937, 0.939) |

Note: These C-statistics are observation-level with the original ordinal target (including interventions and high-risk assessments). Not directly comparable to our encounter-level, fall-only validation.

**Model Input Variables** (from Epic model brief):

*Demographics*: patient age, clinically relevant sex, duration of encounter, drug use, tobacco use

*Vital Signs/Assessments* (most recent within 72h, WDL or exception):
Braden Score (default 19 if missing), cardiac assessment WDL, diastolic BP, GI assessment WDL, Glasgow Coma Scale (default 15 if missing), musculoskeletal assessment WDL, peripheral vascular assessment WDL, respiration rate, skin assessment WDL

*Lab Results* (most recent within 72h, normalized via N_x = (x - L_x) / (U_x - L_x)):
Albumin, ALT, bilirubin, calcium, calcium is abnormal, chloride, creatinine, HDL is abnormal, HbA1c is abnormal, magnesium, potassium, total protein is abnormal

*Medications* (Medi-Span pharmacy classes, count of administrations within 24h):
Analgesic narcotics, anticoagulants, anticonvulsants, antidepressants, antihypertensives, antirheumatics, distinct medication classes, misc psychotherapeutics, toxoids, ulcer drugs

*Procedure Orders*: imaging order (binary, any point during encounter)

*LDAs*: peripheral IV (binary, active LDA)

*Missing Value Handling*: binary/count variables → 0; continuous → site median; extremes Winsorized at 0.1th/99.9th percentile across training sites

*Financial Class*: removed from model starting May 2024 (was: Medicare, Medicaid, Private, Other/missing)

**Fall Interventions** (used in training target definition):

- Minor: bed lowered, call light within reach, crib/bed brakes on, family support, lighting adjusted, mobility aid footwear, nonskid shoes, padded rails, patient educated, responsible adult informed, scheduled toileting, sensory aid, side rails up, visual checks Q time period
- Major: bed/chair alarm, chemotherapy agent precaution, direct assist device, door always open, elopement procedures, environmental modification/remove furniture, gait belt for transfers, high risk for fall sign, LEO/security present, mattress on floor, muscle-strengthening agent used, placed in bed designed to prevent falls, room moved near station, seizure precaution, sitter/attendant at bed, supervised/assisted with all activity/mobility, video monitoring

---

## Morse Fall Scale (MFS) Specifications

**Overview**: Nurse-administered bedside fall risk assessment tool developed by Janice M. Morse in 1989. Most widely used fall risk screening instrument in acute care hospitals worldwide. Consists of 6 clinician-assessed items summed to a total score (range 0–125).

**Original Citation**: Morse JM, Morse RM, Tylko SJ. Development of a scale to identify the fall-prone patient. *Canadian Journal on Aging*. 1989;8(4):366-377.

**Key Validation Reference** (by the scale author): Morse JM. The safety of safety research: the case of patient fall research. *Can J Nurs Res*. 2006;38(2):73-88. PMID: 16871851.

**Meta-analysis References**:
- Haines TP et al. Design-related bias in hospital fall risk screening tool predictive accuracy evaluations: systematic review and meta-analysis. *J Gerontol A Biol Sci Med Sci*. 2007;62(6):664-72. DOI: 10.1093/gerona/62.6.664
- Aranda-Gallardo M et al. Instruments for assessing the risk of falls in acute hospitalized patients: systematic review and meta-analysis. *BMC Health Serv Res*. 2013;13:122. DOI: 10.1186/1472-6963-13-122

**Scoring Items**:

| Item | Response | Points |
|---|---|---|
| **1. History of falling** (current admission or immediate pre-admission) | No | 0 |
| | Yes | 25 |
| **2. Secondary diagnosis** (≥2 medical diagnoses) | No | 0 |
| | Yes | 15 |
| **3. Ambulatory aid** | None / bed rest / nurse assist | 0 |
| | Crutches / cane / walker | 15 |
| | Furniture (holds onto furniture to ambulate) | 30 |
| **4. IV therapy / heparin lock** | No | 0 |
| | Yes | 20 |
| **5. Gait** | Normal / bed rest / wheelchair | 0 |
| | Weak | 10 |
| | Impaired | 20 |
| **6. Mental status** | Oriented to own ability | 0 |
| | Overestimates ability / forgets limitations | 15 |

**Total score range**: 0–125

**Standard Risk Cutoffs** (from Morse 1989):

| Score | Risk Level | Action |
|---|---|---|
| 0–24 | Low risk | Basic nursing care |
| 25–44 | Moderate risk | Standard fall prevention interventions |
| ≥45 | High risk | High-risk fall prevention protocol |

**Rush University Medical Center cutoffs**: ≥25 moderate risk, ≥45 high risk (aligns with standard Morse cutoffs)

**Published Validation Statistics** (vary by setting):
- Original validation (Morse 1989): sensitivity 78%, specificity 83%, 80.5% correct classification, interrater reliability r = 0.96
- Typical ranges across studies: sensitivity 65–90%, specificity 58–85%
- At threshold 45: sensitivity ~80–88%, specificity ~58–83% (setting-dependent)
- At threshold 25: higher sensitivity but lower specificity (trades false negatives for false positives)

**Key Characteristics for Interpretation**:
- Manually scored by nurses — subject to interrater variability and assessment frequency constraints
- Includes fall history as a scored item (unlike Epic PMFRS, which deliberately excludes it)
- Items are binary or ordinal (not continuous) — score distribution is discrete/multimodal
- Standard of care comparator: well-established clinical validity, regulatory acceptance
- Treatment paradox applies equally: patients scored high-risk receive interventions that may prevent the fall the score predicted

---

## Key Design Decisions

### Score Timing Hierarchy

| Priority | Predictor | Columns | Bias Risk |
|---|---|---|---|
| **Primary** | First score at admission | `epic_score_admission`, `morse_score_admission` | None |
| Sensitivity 1 | Score before fall / before discharge | `epic_score_before_fall`, `morse_score_before_fall` | Moderate |
| Sensitivity 2 | Max score during encounter | `epic_score_max`, `morse_score_max` | High |
| Sensitivity 3 | Mean score during encounter | `epic_score_mean`, `morse_score_mean` | Moderate |

### Critical Methodological Rules

**DO**:
- Use 2000+ stratified bootstrap resamples (preserving `fall_flag` ratio) for all CIs
- Use DeLong test for paired AUROC comparison (both models on same encounters)
- Use GEE (exchangeable correlation, robust SE) for clustering by `mrn`
- Report AUPRC alongside AUROC (at 1% prevalence, AUROC alone is insufficient)
- Report event NRI and non-event NRI **separately** (Pepe 2015)
- Present calibration as observed fall rate per score category for ordinal scores
- Use LOWESS with `frac=0.3` to `0.5` for calibration plots (rare events need wider bandwidth)
- Include spike/rug plot on calibration figures showing score distribution

**DO NOT**:
- Upsample, SMOTE, or oversample the data — this is validation, not model development
- Use matched case-control sampling — distorts prevalence-dependent metrics
- Rely on accuracy or F1-score — misleading with 99:1 class imbalance
- Use Hosmer-Lemeshow test — underpowered for rare events, sensitive to binning
- Import pandas for data manipulation — Polars only (convert at library boundaries)

---

## Figures and Tables Plan

Mapped to Wong et al. JAMA Network Open 2026:

### Main Manuscript

| Element | Title | Notebook |
|---|---|---|
| **Table 1** | Patient characteristics by fall status | `03_table1.py` |
| **Table 2** | Model performance: PMFRS vs MFS (AUROC, sens, spec, PPV, NPV, NNE) | `04_primary_analysis.py` |
| **Table 3** | Reclassification analysis (NRI, IDI) | `08_reclassification.py` |
| **Figure 1** | Multi-panel discrimination (sens, spec, PPV, NPV across thresholds) | `04_primary_analysis.py` |
| **Figure 2** | Dot plot: AUROC comparison across score timings | `11_dot_plot.py` |
| **Figure 3** | Decision curve analysis | `06_decision_curve.py` |
| **Notebook 03b** | Recalibration explainer (no numbered figure; generates `recalibration_mapping`) | `03b_recalibration.py` |

### Supplement

| Element | Title | Notebook |
|---|---|---|
| **eFigure 1** | Calibration plot: Epic PMFRS | `05_calibration.py` |
| **eFigure 2** | Calibration plot: Morse Fall Scale | `05_calibration.py` |
| **eFigure 3** | Threshold comparison overlay on ROC | `07_threshold_analysis.py` |
| **eFigure 4** | CONSORT-style cohort flow diagram | `02_cohort_flow.py` |
| **eFigure 5** | Score distributions with flag rate annotations | `07_threshold_analysis.py` |
| **eTable 1** | Fairness audit: stratified by age | `09_fairness_audit.py` |
| **eTable 2** | Fairness audit: stratified by race/ethnicity | `09_fairness_audit.py` |
| **eTable 3** | Fairness audit: stratified by unit type | `09_fairness_audit.py` |
| **eTable 4** | Sensitivity analyses summary (all score timings) | `10_sensitivity_analyses.py` |
| **eTable 5** | Calibration summary (CITL, slope, ICI) | `05_calibration.py` |
| **eTable 6** | Threshold analysis: optimal cutpoints by method | `07_threshold_analysis.py` |
| **eTable 7** | DCA net benefit at selected threshold probabilities | `06_decision_curve.py` |
| **eTable 8** | Fairness audit: stratified by gender | `09_fairness_audit.py` |
| **eTable 9** | Literature benchmarking: MFS validation studies | `13_master_report.py` |
| **eTable 10** | Classification metrics by score timing | `10_sensitivity_analyses.py` |
| **Notebook 12** | Aggregate report — collects key results from all notebooks | `12_report.py` |
| **Notebook 13** | Master report — full pipeline reproduction in one notebook | `13_master_report.py` |

---

## Threshold Selection Methods

Compare head-to-head for both models:

1. **Youden index**: `J = max(sensitivity + specificity - 1)`
2. **Closest-to-(0,1)**: `min(sqrt((1-sens)^2 + (1-spec)^2))`
3. **Fixed sensitivity at 60%**: per Wong et al.
4. **Fixed sensitivity at 80%**: safety-critical target
5. **Value-optimizing (NMB)**: Parsons et al. JAMIA 2023 — Monte Carlo with cost distributions:
   - Cost of fall: ~$14,000 (Normal(14000, 3000))
   - Cost of intervention: ~$200 (Normal(200, 50))
   - Intervention effectiveness: ~30% RR reduction (Beta(40, 60))
   - QALY loss per fall: 0.0036 (Normal(0.0036, 0.0005))
   - WTP: $100,000/QALY
6. **DCA-derived**: range where net benefit > treat-all AND > treat-none
7. **Standard cutoffs**: MFS ≥25 (moderate), ≥45 (high risk at Rush); Epic 3-tier: low (0–34), medium (35–69), high (70–100); Epic 2-tier: low (0–49), high (50–100)

**Note on Epic ≥70**: At admission, only 0.4% of encounters score ≥70. This cutoff is included in descriptive analyses but omitted from categorical NRI (insufficient events for meaningful reclassification).

---

## Technology Stack

### Environment

- **Package manager**: `uv` (Astral). Use `uv sync` to install, `uv run` to execute.
- **Config**: `pyproject.toml` (PEP 621). Dev deps in `[dependency-groups]` (PEP 735).
- **Lockfile**: `uv.lock` committed for reproducibility.
- **Python**: 3.12 (pinned in `.python-version`). Upper bound `<3.14` (dcurves constraint).
- **Task runner**: `Makefile`. Key targets: `make setup`, `make edit`, `make run-all`, `make quality`.
- **Pre-commit**: ruff lint+format, `marimo check`, check-added-large-files.

### Polars (NOT Pandas)

All data manipulation uses Polars. Never import pandas for data work.

```python
import polars as pl

df = pl.read_parquet("data/processed/analytic.parquet")

# Lazy evaluation for large operations
lf = pl.scan_parquet("data/processed/analytic.parquet")
result = lf.filter(pl.col("fall_flag") == 1).collect()

# Convert to numpy ONLY at the sklearn/scipy boundary
y_true = df["fall_flag"].to_numpy()
y_score = df["epic_score_admission"].to_numpy()

# Convert to pandas ONLY at the dcurves/statsmodels boundary
pdf = df.to_pandas()
```

Key Polars patterns:
- `pl.read_csv()` / `pl.read_parquet()` for I/O
- `.filter()`, `.select()`, `.with_columns()` for transforms
- `.group_by().agg()` for grouped statistics
- `.to_numpy()` only when passing to sklearn/scipy
- `.to_pandas()` only when passing to dcurves or statsmodels GEE
- Use `pl.Expr` chaining, not apply/map
- Use `pl.scan_parquet()` + `.collect()` for lazy evaluation

### Marimo Notebooks

All analysis code is Marimo notebooks (.py files). Marimo is reactive, Git-friendly, and has native Polars support.

```bash
uv run marimo edit notebooks/01_data_discovery.py   # Interactive editing
uv run marimo run notebooks/01_data_discovery.py     # Run as script (CI)
uv run marimo export html notebooks/01_data_discovery.py -o outputs/html/01_data_discovery.html
```

Marimo notebook structure:
```python
import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    return mo, pl, np, plt, Path

@app.cell
def _(mo):
    mo.md("# Section Title")
    return

@app.cell
def _(pl, Path):
    df = pl.read_parquet(Path("data/processed/analytic.parquet"))
    return (df,)
```

Marimo conventions:
- Each cell returns variables used by downstream cells via `return (var,)` tuple
- Use `mo.md()` for narrative sections
- Use `mo.ui.table(df)` for interactive dataframe display
- Use `mo.ui.slider()` / `mo.ui.dropdown()` for interactive parameter exploration
- Use `mo.stop()` for conditional cell execution
- SQL cells via `mo.sql()` use DuckDB under the hood, return Polars DataFrames
- Set `on_cell_change = "lazy"` in marimo config for expensive bootstrap/GEE cells

### Cross-Framework Data Flow

```
[xlsx] --> Polars (all manipulation)
             |---> .to_numpy()  --> sklearn (AUROC, metrics)
             |---> .to_pandas() --> dcurves (DCA) -- dcurves v1.1.7 requires pandas
             |---> .to_pandas() --> statsmodels (GEE, LOWESS calibration)
             |---> GT(polars_df) --> great-tables (publication tables) -- native Polars support
             +---> .to_numpy()  --> matplotlib (figures)
```

---

## JAMA Figure Specifications

All figures comply with JAMA Network Open technical requirements.

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

JAMA_STYLE = {
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.linewidth": 0.5,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "legend.frameon": False,
}
mpl.rcParams.update(JAMA_STYLE)

COLORS = {
    "epic": "#2166AC",
    "morse": "#B2182B",
    "treat_all": "#777777",
    "treat_none": "#333333",
    "ci_fill": "0.85",
}

FIG_SINGLE_COL = (3.5, 2.8)    # Single column
FIG_DOUBLE_COL = (7.0, 4.5)    # Full width
FIG_MULTI_PANEL = (7.0, 8.0)   # Multi-panel

def save_figure(fig, name, formats=("pdf", "png")):
    """Save figure in multiple formats for JAMA submission."""
    for fmt in formats:
        fig.savefig(f"outputs/figures/{name}.{fmt}", format=fmt,
                    dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
```

**Figure rules**:
- JAMA style: Arial font, minimum 8pt for ALL text (no exceptions)
- No text overlap: titles, legends, footnotes, axis labels, cell annotations must not collide
- Legends must NOT block data: place outside plot area (below, beside, or in whitespace)
- Axis labels: sentence case, JAMA format (e.g. "Respondents, %" not "% Endorsing")
- Titles: 10-12pt bold; axis labels: 9-10pt; tick labels: 8-9pt; annotations/legends/footnotes: 8pt minimum
- Color: blue (#2166AC) for Epic, red (#B2182B) for Morse (colorblind-friendly)
- Export as vector PDF (primary) + 300 DPI PNG (backup), each < 1 MB
- After generating a figure, visually inspect the output image to verify no overlap or blocked data

**Figure layout rules** (mandatory for all figures with below-axes elements):

Every figure that has a caption below the axes MUST follow this pattern:
```python
fig.subplots_adjust(bottom=0.22)                           # Reserve space
fig.legend(..., bbox_to_anchor=(0.5, -Y_LEGEND), ...)      # Y_LEGEND >= 0.06
fig.text(0.5, -(Y_LEGEND + 0.06), "Caption...", ...)       # At least 0.06 below legend
save_figure(fig, name, bbox_inches=None, pad_inches=0.15)  # NEVER use bbox_inches="tight"
```

- **NEVER use `bbox_inches="tight"` when `fig.text()` or `fig.legend()` is placed in negative y-space** — it clips elements outside the axes. Always pass `bbox_inches=None, pad_inches=0.15` to `save_figure()`.
- **NEVER use `tight_layout()` combined with manual element positioning** — `tight_layout()` overrides `subplots_adjust()` and invalidates manually placed legends/captions. Use `subplots_adjust()` exclusively.
- **Maintain >= 0.06 gap** in figure coordinates between the legend `bbox_to_anchor` y-position and the caption `fig.text()` y-position.
- **Use collision avoidance for annotation labels** when multiple threshold lines cluster together:
  ```python
  _placed_y = []
  for _t, _lbl in _thresholds:
      _y_pos = ax.get_ylim()[1] * 0.85
      for _py in _placed_y:
          if abs(_y_pos - _py) < ax.get_ylim()[1] * 0.15:
              _y_pos -= ax.get_ylim()[1] * 0.15
      ax.text(_t + 1, _y_pos, label_text, ...)
      _placed_y.append(_y_pos)
  ```
- **If a figure exists in both an individual notebook AND `13_master_report.py`, both copies MUST use identical spacing values.** When editing figure layout in one file, always sync the other.
- **After saving, visually inspect the PNG output** — check all four edges for clipped text, verify legends do not overlap captions, confirm annotation labels do not collide.

---

## Code Organization

```
rushfalla/
├── CLAUDE.md                           # This file
├── pyproject.toml                      # Dependencies (uv)
├── uv.lock                            # Pinned lockfile (committed)
├── Makefile                           # Task runner
├── .python-version                    # 3.12
├── .pre-commit-config.yaml            # ruff + marimo check
├── .gitignore                         # Excludes data/, outputs/, .xlsx
├── scripts/
│   └── generate_docx_tables.py        # JAMA-formatted DOCX tables from CSVs
├── notebooks/
│   ├── 01_data_discovery.py           # Column standardization, profiling, QC
│   ├── 02_cohort_flow.py             # CONSORT diagram (eFigure 4)
│   ├── 03_table1.py                  # Patient characteristics (Table 1)
│   ├── 03b_recalibration.py          # Recalibration explainer + mapping figure
│   ├── 04_primary_analysis.py        # AUROC, DeLong, metrics (Table 2, Figure 1)
│   ├── 05_calibration.py             # Calibration plots (eFigures 1-2)
│   ├── 06_decision_curve.py          # DCA (Figure 3)
│   ├── 07_threshold_analysis.py      # Threshold comparison (eFigure 3)
│   ├── 08_reclassification.py        # NRI/IDI (Table 3)
│   ├── 09_fairness_audit.py          # Stratified performance (eTables 1-3)
│   ├── 10_sensitivity_analyses.py    # All sensitivity runs (eTable 4)
│   ├── 11_dot_plot.py                # Model comparison dot plot (Figure 2)
│   ├── 12_report.py                  # Aggregate report — collects all results
│   └── 13_master_report.py           # Master report — full pipeline in one notebook
├── utils/
│   ├── __init__.py
│   ├── metrics.py                     # DeLong, bootstrap, NRI/IDI, value-optimizing threshold
│   ├── plotting.py                    # JAMA_STYLE, save_figure(), COLORS, figure sizes
│   └── constants.py                   # Column maps, thresholds, labels, department-to-unit map
├── data/
│   ├── raw/                           # output_table_v4.xlsx (NOT committed, in .gitignore)
│   └── processed/                     # analytic.parquet (NOT committed)
├── outputs/
│   ├── docx/                           # JAMA Word tables (manuscript + supplement)
│   ├── figures/                        # PDF + PNG per figure
│   ├── tables/                         # CSV per table
│   └── html/                           # Exported marimo notebooks
└── docs/
    └── plans/                          # Implementation plans
```

---

## Execution Order

```bash
# Phase 1: Data preparation
make edit NB=notebooks/01_data_discovery.py
make edit NB=notebooks/02_cohort_flow.py

# Phase 2: Descriptives
make edit NB=notebooks/03_table1.py
make edit NB=notebooks/03b_recalibration.py

# Phase 3: Primary analysis
make edit NB=notebooks/04_primary_analysis.py
make edit NB=notebooks/05_calibration.py
make edit NB=notebooks/06_decision_curve.py
make edit NB=notebooks/07_threshold_analysis.py

# Phase 4: Secondary analyses
make edit NB=notebooks/08_reclassification.py
make edit NB=notebooks/09_fairness_audit.py
make edit NB=notebooks/10_sensitivity_analyses.py
make edit NB=notebooks/11_dot_plot.py

# Phase 5: Aggregation
make edit NB=notebooks/12_report.py
make edit NB=notebooks/13_master_report.py

# Batch run all (CI/reproducibility)
make run-all

# Export HTML for co-authors
make export-html
```

---

## Statistical Specifications

| Parameter | Value | Justification |
|---|---|---|
| Alpha (primary) | 0.05, two-sided | Standard for validation |
| Bootstrap replicates | 2000 (5000 for final) | Adequate for 1% prevalence |
| Bootstrap method | Stratified by `fall_flag`, BCa | Preserve event ratio |
| Random seed | 42 | Reproducibility |
| GEE family | Binomial (logit link) | Binary outcome |
| GEE correlation | Exchangeable | Within-patient clustering (778 patients) |
| GEE SE | Robust (sandwich) | statsmodels default; valid regardless of misspecification |
| Min subgroup events | 20 | Below this, AUROC unreliable |
| DCA threshold range | 0-10% | Clinically relevant for 1% prevalence |
| LOWESS bandwidth | frac=0.3 | Increased for rare events |
| NRI reporting | Event and non-event separately | Combined NRI misleading (Pepe 2015) |

---

## Known Gaps Requiring Clarification

Before implementing certain notebooks, these questions need answers from the data team:

1. ~~**Does the dataset contain Epic probability output or only ordinal scores?**~~ **REVISED**: The model brief says "integer score 0–100" from ordinal logistic regression, but our dataset contains **continuous Float64 values** with 5 decimal places (61,908 unique values, range 0.117–96.706). This is likely the raw model probability output (×100) before bedside integer rounding. **Action**: Treat as a continuous risk score in [0, 100] for all analyses. This is actually better than integers for discrimination analysis (finer rank ordering). For calibration, use logistic recalibration (score → probability) since the raw score is not a calibrated probability regardless. **Needs confirmation from data team**: Was integer rounding bypassed in the extract, or is this the internal model representation?
2. **Are max/mean scores restricted to pre-fall assessments for fallers?** (Look-ahead bias if post-fall included)
3. **What is the exact definition of "admission score"?** PARTIALLY ANSWERED: Epic model brief states the model runs continuously and can generate scores for any patient with an active hospital admission. "Admission score" in our dataset is likely the first score generated after admission, but exact timing window still needs confirmation from data team.
4. **Study date range?** (Required for TRIPOD+AI Item 5b)
5. **What constitutes the non-faller comparator for pre-fall scores?** (Last score before discharge? Or not in dataset?)
6. **Note**: The Epic PMFRS does NOT include patient history of falls as an input variable (deliberate exclusion). The Morse Fall Scale includes it as a 25-point scored item. This asymmetry is relevant context for interpretation of any head-to-head comparison.

---

## Notes for Claude

### Data & Computation
- Use **Polars** for ALL data manipulation. Never import pandas for data work.
- Convert to numpy only at the sklearn/scipy boundary: `df["col"].to_numpy()`
- Convert to pandas only at the dcurves/statsmodels boundary: `df.to_pandas()`
- Use Polars lazy evaluation (`scan_parquet`, `collect()`) for large operations
- Write all analysis as **Marimo notebooks** stored as .py files
- Bootstrap with 2000 stratified resamples (preserving `fall_flag` ratio) for all CIs
- DeLong test (Sun & Xu 2014 fast algorithm) for paired AUROC comparison
- GEE (exchangeable, robust SE) for primary analysis to handle within-patient clustering
- Do NOT upsample, SMOTE, or use matched case-control for any analysis
- Treatment paradox is certain — AUROC estimates are conservative
- Epic PMFRS is an ordinal logistic regression. The model brief says "integer score 0–100" but our dataset contains **continuous float scores** (0.117–96.706, 61,908 unique values). Treat as a continuous risk score for discrimination (AUROC, DeLong) and use logistic recalibration (score → probability) for calibration, DCA, and NRI/IDI. It is NOT a calibrated probability.
- Epic's recommended thresholds (35/70 for 3-tier; 50 for 2-tier) were designed for continuous monitoring, not admission screening. At admission, 97% of encounters score < 35. Include these thresholds in analyses to demonstrate the gap, but interpret accordingly.
- Morse scores are discrete integers with only 24 unique values (items sum to 0–125). ROC curves and threshold plots will appear step-wise. Report score distributions as median [IQR], not just mean ± SD.
- The model's training target included interventions and high-risk assessments (target 0/1/2), but our validation target is falls only (binary). This is the correct approach for prospective validation per the updated Epic validation methodology.

### Figures
- Follow JAMA figure specs defined in `utils/plotting.py`
- Use matplotlib with `JAMA_STYLE` rcParams
- Arial font, 8pt minimum body text, no top/right spines, no grid
- Color: blue (#2166AC) for Epic, red (#B2182B) for Morse
- Always save both PDF and PNG via `save_figure()`
- Figure widths: 3.5in (single column) or 7.0in (full width)
- Each file < 1 MB
- Visually inspect every figure for text overlap before presenting
- **ALWAYS** pass `bbox_inches=None, pad_inches=0.15` to `save_figure()` when the figure has `fig.text()` or `fig.legend()` placed below the axes — `bbox_inches="tight"` clips these elements
- **NEVER** use `tight_layout()` on figures with manually positioned legends or captions — use `subplots_adjust()` instead
- Maintain >= 0.06 figure-coordinate gap between legend and caption y-positions
- Use collision avoidance (track placed y-positions, stagger if overlapping) when multiple annotation labels cluster together
- When editing figure layout in any individual notebook, always sync the same values to the corresponding section in `13_master_report.py`

### Code Style
- Prefer direct, concise code. No unnecessary abstraction.
- Write code that produces publication-ready output, not intermediate exploration.
- Every notebook must be runnable as `uv run marimo run notebooks/XX_*.py`
- Use `mo.md()` for narrative sections within notebooks.
- Use `mo.ui.table()` to display Polars DataFrames interactively.
- Ruff ignores `B018` in `notebooks/*.py` (last expression is marimo display output).
- If unsure about Marimo API, check docs.marimo.io or use MCP tools to verify.

### Key References
- **Blueprint**: Wong et al. JAMA Netw Open 2026;9(2):e260181
- **Reporting**: Collins et al. BMJ 2024;385:e078378 (TRIPOD+AI)
- **Sample size**: Riley et al. Stat Med 2021;40(19):4230-4251
- **DeLong**: Sun X, Xu W. Biometrics 2014;70(3):514-524
- **DCA**: Vickers AJ et al. Diagn Progn Res 2019;3:18
- **Calibration**: Van Calster B et al. BMC Medicine 2019;17:230
- **Thresholds**: Parsons R et al. JAMIA 2023;30(6):1103-1113
- **NRI caution**: Pepe MS et al. Stat Med 2015;34(1):110-128
- **Morse Fall Scale**: Morse JM, Morse RM, Tylko SJ. *Can J Aging*. 1989;8(4):366-377
- **MFS meta-analysis**: Aranda-Gallardo M et al. *BMC Health Serv Res*. 2013;13:122
- **Epic PMFRS Model Brief**: Epic Systems. Cognitive Computing Model Brief: Inpatient Risk of Falls. August 13, 2025
- **MFS design bias**: Haines TP et al. *J Gerontol A Biol Sci Med Sci*. 2007;62(6):664-672 (PMID: 17595425)
- **MFS prospective validation**: Ji Y et al. 2023 (PMID: 37305899); Shim J et al. 2022 (PMID: 36128798)
