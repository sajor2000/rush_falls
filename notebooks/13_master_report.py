import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from scipy import stats

    return (
        FancyArrowPatch,
        FancyBboxPatch,
        Path,
        json,
        matplotlib,
        mo,
        mpatches,
        np,
        pl,
        plt,
        stats,
    )


@app.cell
def _():
    from utils.constants import (
        AGE_BINS,
        AGE_LABELS,
        ALPHA,
        COLUMN_RENAME_MAP,
        DCA_THRESHOLD_MAX,
        DCA_THRESHOLD_MIN,
        DCA_THRESHOLD_STEP,
        EPIC_2TIER_HIGH,
        EPIC_3TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        ETHNICITY_CLEAN_MAP,
        GENDER_CLEAN_MAP,
        MFS_HIGH,
        MFS_MODERATE,
        MIN_SUBGROUP_EVENTS,
        MODEL_LABELS,
        N_BOOTSTRAP,
        RACE_CLEAN_MAP,
        RANDOM_SEED,
        SCORE_TIMING,
    )

    return (
        AGE_BINS,
        AGE_LABELS,
        ALPHA,
        COLUMN_RENAME_MAP,
        DCA_THRESHOLD_MAX,
        DCA_THRESHOLD_MIN,
        DCA_THRESHOLD_STEP,
        EPIC_2TIER_HIGH,
        EPIC_3TIER_HIGH,
        EPIC_3TIER_MEDIUM,
        ETHNICITY_CLEAN_MAP,
        GENDER_CLEAN_MAP,
        MFS_HIGH,
        MFS_MODERATE,
        MIN_SUBGROUP_EVENTS,
        MODEL_LABELS,
        N_BOOTSTRAP,
        RACE_CLEAN_MAP,
        RANDOM_SEED,
        SCORE_TIMING,
    )


@app.cell
def _():
    from utils.metrics import (
        calibration_metrics,
        classification_metrics_at_threshold,
        closest_topleft_threshold,
        compute_categorical_nri,
        compute_nri_idi,
        delong_ci,
        delong_roc_test,
        extract_dca_threshold_range,
        fixed_sensitivity_threshold,
        logistic_recalibration,
        stratified_bootstrap,
        value_optimizing_threshold,
        youden_threshold,
    )

    return (
        calibration_metrics,
        classification_metrics_at_threshold,
        closest_topleft_threshold,
        compute_categorical_nri,
        compute_nri_idi,
        delong_ci,
        delong_roc_test,
        extract_dca_threshold_range,
        fixed_sensitivity_threshold,
        logistic_recalibration,
        stratified_bootstrap,
        value_optimizing_threshold,
        youden_threshold,
    )


@app.cell
def _():
    from utils.cluster_auroc import (
        cluster_bootstrap_auroc_comparison,
        estimate_design_effect,
        gee_discrimination_check,
    )
    from utils.plotting import (
        COLORS,
        FIG_DOUBLE_COL,
        FIG_MULTI_PANEL,
        FIG_SINGLE_COL,
        JAMA_STYLE,
        save_figure,
    )

    return (
        COLORS,
        FIG_DOUBLE_COL,
        FIG_MULTI_PANEL,
        FIG_SINGLE_COL,
        JAMA_STYLE,
        cluster_bootstrap_auroc_comparison,
        estimate_design_effect,
        gee_discrimination_check,
        save_figure,
    )


@app.cell
def _(mo):
    mo.md("""
    # Validation of the Epic Predictive Model Fall Risk Score vs the Morse Fall Scale for Inpatient Fall Prediction

    **Rush University Medical Center**

    Single-center retrospective validation study guided by TRIPOD+AI (2024) reporting standards.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "📋 Study Design Overview": mo.md(
            """
            This study provides a rigorous head-to-head comparison of two fall risk assessment tools in a real-world academic hospital setting.

            ## Research Objective

            Compare a machine learning–driven predictive model (**Epic Predictive Model Fall Risk Score, PMFRS**) against a nurse-administered bedside instrument (**Morse Fall Scale**) for predicting inpatient falls at the time of admission.

            ## Study Design

            - **Type**: Retrospective external temporal validation (TRIPOD Type 3)
            - **Comparative framework**: Both tools evaluated on identical patient encounters
            - **Primary timing**: Admission-time screening (first assessment after arrival)
            - **Sensitivity analyses**: Alternative score timings encountered during hospitalization

            ## Study Population

            - **Setting**: Rush University Medical Center (academic tertiary care hospital)
            - **Eligibility**: Adult inpatients (age ≥18) with complete Epic PMFRS and Morse Fall Scale assessments at admission
            - **Analytic cohort**: 63,324 hospital encounters comprising 803 fall events (1.27% event rate)
            - **Statistical power**: Sufficient to detect AUROC differences as small as 0.02 with 80% power at α = 0.05

            ## Clinical Outcome

            **Inpatient fall** — any fall event documented in the electronic health record during hospitalization, detected via either:

            - Recorded fall date/timestamp
            - Documented fall location within the hospital

            ## Analysis Scope

            | Dimension | What We tested | Why it matters |
            |-----------|---------------|----------------|
            | **Discrimination** | AUROC comparison | Can the tools correctly rank fallers vs non-fallers? |
            | **Calibration** | Accuracy of predicted probabilities | Do risk estimates match observed fall rates? |
            | **Clinical utility** | Decision curve analysis | Which tool provides meaningful net benefit? |
            | **Reclassification** | NRI, IDI metrics | Does Epic improve risk categorization vs Morse? |
            | **Fairness** | Subgroup performance by age, race, ethnicity, gender | Are there systematic performance disparities? |
            | **Timing** | Admission vs pre-fall vs max scores | When scores are collected affects performance |

            ## Reporting Standards

            All analyses adhere to **TRIPOD+AI** (Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis with Artificial Intelligence) guidelines (Collins et al., *BMJ* 2024;385:e078378).
            """
        ),
        "🎯 Key Questions": mo.md(
            """
            This study addresses six core questions relevant to hospital administrators, clinicians, and researchers evaluating fall risk stratification tools.

            ### 1. Discrimination Performance

            **Does Epic PMFRS outperform Morse Fall Scale in identifying patients who will fall?**

            The AUROC (Area Under the Receiver Operating Characteristic curve) measures each tool's ability to rank fallers higher than non-fallers. This is the primary metric for evaluating discriminatory power.

            ### 2. Calibration Accuracy

            **Do predicted risk probabilities match observed fall outcomes?**

            A well-calibrated tool provides trustworthy risk estimates. If a tool predicts 5% fall risk, approximately 5% of patients with that score should actually fall. We assess this through calibration plots, slope, intercept, and the Integrated Calibration Index.

            ### 3. Clinical Utility

            **Which tool provides greater benefit for decision-making?**

            Discrimination alone is insufficient—clinical decisions weigh the benefits of true positives against the costs of false alarms. Decision curve analysis quantifies net benefit across a range of intervention thresholds, revealing which tool actually improves outcomes when deployed.

            ### 4. Risk Reclassification

            **Does Epic PMFRS meaningfully reclassify patients into more appropriate risk categories?**

            If a hospital switched from Morse to Epic, would patients end up in correct risk tiers more often? Net Reclassification Improvement (NRI) measures this directly. We report NRI separately for fallers (who should be classified as high-risk) and non-fallers (who should be low-risk), as combined NRI can be misleading.

            ### 5. Subgroup Fairness

            **Do performance differences exist across patient subgroups?**

            A prediction tool should work equally well for all patients. We evaluate both tools by age, race/ethnicity, gender, and clinical unit. Subgroups with fewer than 20 events are excluded due to unreliable estimates.

            ### 6. Timing Sensitivity

            **How does performance vary when using admission scores versus scores closer to fall events?**

            Our primary analysis uses admission-time screening (clinically actionable for early intervention). However, Epic's model was designed for continuous monitoring with changing scores over time. We test multiple timing strategies to understand how score dynamics affect performance.
            """
        ),
        "📊 Analytic Framework": mo.md(
            """
            ## Comparative Validation Design

            **Head-to-head comparison** on identical encounters — both Epic PMFRS and Morse Fall Scale assessments are available for each patient in our analytic cohort. This paired design enables fair statistical comparison via the DeLong test for AUROC differences.

            - **No model refitting**: Primary assessment uses out-of-the-box performance — no recalibration or parameter tuning applied, reflecting real-world deployment
            - **Temporal validation**: Models trained on earlier populations, validated on our later-time-period cohort
            - **Identical outcome**: Both tools evaluated against the same gold-standard fall events

            ## Score Timing Hierarchy

            | Priority | Score Type | Clinical Usefulness | Bias Risk |
            |----------|-----------|---------------------|-----------|
            | **Primary** | Admission score | ✅ Actionable for triage, early intervention | ✅ None |
            | Secondary | Pre-fall/pre-discharge score | ⚠️ May not be available at critical decision points | ⚠️ Moderate |
            | Exploratory | Maximum score during encounter | ❌ Lookahead bias — includes post-fall data | ❌ High |
            | Exploratory | Mean score during encounter | ⚠️ Lookahead bias — includes post-fall data | ❌ Moderate |

            **Critical distinction**: Epic's published thresholds (≥35 for 3-tier medium risk, ≥70 for high risk) were validated using **maximum scores over full encounters**. Our admission-time validation uses **first scores only**. This timing difference is a key finding — Epic's threshold recommendations are not appropriate for admission screening.

            ## Complete-Case Analysis

            Encounters missing either Epic PMFRS or Morse Fall Scale admission scores are excluded. This introduces minimal bias:

            - **Missing Epic scores**: 229 encounters (0.27%), **0 fallers** — no impact on event rates
            - **Missing Morse scores**: 416 encounters (0.49%), **1 faller** — negligible impact
            - **Overall**: >99% of eligible encounters retained, virtually complete case ascertainment

            ## Statistical Methodology

            All analyses use rigorous uncertainty quantification and cluster-robust inference:

            ### Confidence Intervals

            - **Method**: Stratified bootstrap resampling with 2,000 replicates
            - **Stratification**: Preserves exact fall/non-fall ratio (1:79) in each resample
            - **Signal**: Bias-corrected and accelerated (BCa) confidence intervals, more accurate than percentile intervals for skewed distributions
            - **Reproducibility**: Random seed = 42 fixed across all analyses

            ### Within-Patient Clustering

            - **Concern**: 778 unique patients contribute 63,324 encounters — repeated admissions from the same patients are not statistically independent
            - **Solution**: Generalized Estimating Equations (GEE) with **exchangeable correlation structure** and **robust sandwich standard errors**
            - **Result**: Valid inference even if correlation structure is misspecified

            ### Model Comparison

            - **AUROC comparison**: DeLong test for paired ROC curves (Sun & Xu, *Biometrics* 2014)
            - **Significance threshold**: α = 0.05, two-sided
            - **Multiple testing**: Bonferroni correction applied to sensitivity analyses with multiple comparisons

            ## Output Generation

            - **Figures**: JAMA Network Open technical specifications (Arial, 8pt minimum, 300 DPI, single- and double-column formats)
            - **Tables**: Publication-ready CSV files with 95% confidence intervals for all point estimates
            - **Reproducibility**: Complete audit trail from raw Excel export to final results — no pre-computed values
            """
        ),
    })
    return


@app.cell
def _(mo):
    mo.md("""
    ## Reader Guide

    This master notebook contains the complete analytical pipeline from raw hospital data to the final manuscript outputs. Every step of the analysis is documented, reproducible, and transparent.

    ### How to Use This Notebook

    **For clinicians and hospital administrators** — Focus on the section introductions and "🎯 Key Findings" summary boxes. These provide the practical takeaways without technical jargon.

    **For methodologists and reviewers** — Expand the collapsible "Statistical details" accordions to examine the analytical methods, statistical tests, and software implementations.

    **For co-authors** — Use the "🎯 Key Findings" boxes to quickly gather results for manuscript sections. All tables and figures are generated from the underlying data, not pre-computed.

    **For reproducibility verification** — Every cell is computationally tractable. The notebook can be run end-to-end to regenerate all tables and figures from the raw Excel export.

    ### Analytical Workflow

    The analysis proceeds through thirteen logical sections:

    1. **Study Overview** — Design, objectives, and methodology
    2. **Data Loading** — Raw data processing and quality checks
    3. **Cohort Definition** — CONSORT flow diagram and exclusion criteria
    4. **Descriptive Statistics** — Table 1: Patient characteristics
    5. **Primary Analysis** — Table 2: Discrimination performance
    6. **Calibration** — eFigures 1-2: Risk estimation accuracy
    7. **Decision Curve Analysis** — Figure 3: Clinical utility
    8. **Thresholds** — eFigures 3, 5: Cutpoint optimization
    9. **Reclassification** — Table 3: NRI and IDI metrics
    10. **Fairness** — eTables 1-3, 8: Subgroup performance
    11. **Sensitivity Analyses** — eTable 4: Timing strategies
    12. **Key Findings** — Synthesis for clinical interpretation
    13. **Output Inventory** — All generated figures and tables

    ### Navigation

    - **Scroll** through sections sequentially for a complete walkthrough
    - **Use the outline** above to jump directly to specific analyses
    - **Expand accordions** for technical methodology (collapsed by default for readability)
    - **Check output inventory** in Section 13 to verify all manuscript figures and tables were generated
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "📚 Data Dictionary": mo.md(
            """
            A complete reference for all analytic variables used throughout the notebook.

            ## Core Identifiers

            | Variable | Type | Definition | Key Notes |
            |----------|------|------------|-----------|
            | `mrn` | String | Medical record number | Unique per patient; repeated for multiple hospitalizations (clustering unit) |
            | `encounter_csn` | String | Contact serial number | Unique per hospital stay — **unit of analysis** |

            ## Demographic Characteristics

            | Variable | Type | Categories | Range |
            |----------|------|------------|-------|
            | `age` | Integer | — | 18+ years (all patients are adults) |
            | `race` | Categorical | White, Black, Asian, Other/Unknown | 4 standardized categories from 11 raw values |
            | `ethnicity` | Categorical | Hispanic/Latino, Not Hispanic/Latino, Unknown | 3 standardized categories from 8 raw values |
            | `gender` | Categorical | Female, Male, Other/Unknown | 3 standardized categories from 6 raw values |

            ### Data Standardization Notes

            Raw EHR codes were harmonized to facilitate subgroup analysis:
            - **Race**: 11 raw categories collapsed to 4 (White, Black, Asian, Other/Unknown)
            - **Ethnicity**: 8 raw categories collapsed to 3 (Hispanic/Latino, Not Hispanic/Latino, Unknown)
            - **Gender**: 6 raw categories collapsed to 3 (Female, Male, Other/Unknown)
            - **Small subgroups** (<20 fall events) are excluded from fairness analyses due to unreliable estimates

            ## Clinical Context Variables

            | Variable | Type | Definition | Clinical Relevance |
            |----------|------|------------|--------------------|
            | `admission_date` | Datetime | Timestamp when patient was admitted to hospital | Defines study epoch, calculates length of stay |
            | `discharge_date` | Datetime | Timestamp when patient was discharged | Needed for study period and LOS calculations |
            | `los_days` | Float | Length of stay in fractional days | Calculated: (discharge - admission) / 86,400 seconds |
            | `admitting_department` | String | Clinical unit where patient was admitted | Used for unit-level analyses (e.g., ICU vs Med-Surg) |
            | `discharge_department` | String | Clinical unit at discharge | Exclusion criterion if missing (indicates incomplete encounter) |

            ## Fall Outcomes

            | Variable | Type | Definition | Detection Logic |
            |----------|------|------------|----------------|
            | `fall_datetime` | Datetime | Timestamp when fall occurred | EHR-documented fall time |
            | `unit_fall_occurred` | String | Hospital unit where fall happened | Documentation of fall location |
            | `fall_flag` | Integer (binary) | Fall outcome indicator | **1** if either `fall_datetime` OR `unit_fall_occurred` is not null; **0** otherwise |

            ### Outcome Definition

            A fall is defined as **any event** captured in the EHR via:
            - Documented fall date/timestamp, OR
            - Documented fall location within the hospital

            This comprehensive definition minimizes false negatives. Null values for both fields indicate no fall occurred during the encounter.

            ## Epic PMFRS Risk Scores

            The Epic Predictive Model Fall Risk Score is a **machine learning model** that runs automatically in the background, generating continuous scores updated throughout the hospitalization.

            ### Score Characteristics

            - **Type**: Continuous floating-point values (not integers despite model brief description)
            - **Range**: Approximately 0–100 (observed: 0.117–96.706)
            - **Distribution**: Heavily right-skewed — median 5.5, mean 8.7 at admission
            - **Granularity**: 5 decimal places (61,908 unique values)

            ### Score Variables

            | Variable | Timing | Definition | Notes |
            |----------|--------|------------|-------|
            | `epic_score_admission` | At admission | First score generated after patient arrival | **Primary analysis — clinically actionable** |
            | `epic_score_before_fall` | Conditional | Most recent score before fall (fallers) or discharge (non-fallers) | Clinical utility for just-in-time intervention; includes post-fall if fall never occurred |
            | `epic_score_max` | Throughout | Maximum score during entire encounter | **Lookahead bias** if used prospectively; represents upper-bound performance |
            | `epic_score_mean` | Throughout | Mean score across all scored points during stay | **Lookahead bias**; includes post-fall data |

            ### Model Distinction

            Epic PMFRS differs fundamentally from Morse Fall Scale:

            | Aspect | Epic PMFRS | Morse Fall Scale |
            |--------|------------|------------------|
            **Source** | Machine learning model (MLPR) | Checklist assessment |
            **Automation** | Runs continuously, auto-updated | Completed manually by nurses |
            **Frequency** | Updates multiple times per stay | One-time or periodic reassessment |
            **Input data** | Medications, labs, vitals, procedures, demographics | 6 nurse-assessed items |
            **Output** | Continuous score | Discrete integer score |

            ## Morse Fall Scale Scores

            The Morse Fall Scale is a **nurse-administered bedside checklist** that patients are scored on at admission and periodically throughout their stay.

            ### Score Characteristics

            - **Type**: Discrete integer sum of 6 scored items
            - **Range**: 0–125 (sum of 6 items with specific point values)
            - **Distribution**: Multi-modal at clinically meaningful thresholds (35, 125, 60, 45)
            - **Granularity**: 24 unique values

            ### Score Variables

            | Variable | Timing | Definition | Notes |
            |----------|--------|------------|-------|
            | `morse_score_admission` | At admission | First nurse assessment after patient arrival | **Primary analysis — clinically actionable** |
            | `morse_score_before_fall` | Conditional | Most recent assessment before fall (fallers) or discharge (non-fallers) | Includes post-fall if fall never occurred |
            | `morse_max` | Throughout | Maximum score during entire encounter | **Lookahead bias**; includes post-fall data |
            | `morse_mean` | Throughout | Mean score across all assessments during stay | **Lookahead bias**; includes post-fall data |

            ### Standard Risk Categories

            | Morse Score | Risk Level | Interpretation |
            |-------------|------------|----------------|
            | 0–24 | Low risk | Basic nursing care |
            | 25–44 | Moderate risk | Standard fall prevention interventions |
            | ≥45 | High risk | High-risk fall prevention protocol |

            Rush University Medical Center follows these standard Morse cutoffs.

            ## Timing Considerations

            The choice of score timing substantially affects performance estimates:

            - **Admission score**: Clinically actionable for early intervention and triage; our **primary analysis**
            - **Pre-fall score**: May not be available at critical decision moments; potential for selection bias
            - **Maximum score**: **Lookahead bias** — includes assessments performed after a fall occurred; not clinically actionable for prediction
            - **Mean score**: **Lookahead bias** — includes post-fall data; represents theoretical upper bound

            **Critical finding**: Epic PMFRS thresholds (≥35, ≥70) were validated using **maximum scores over full encounters**. Our admission-time validation shows that only 3% of encounters score ≥35 at admission, making these cutoffs inappropriate for admission screening.

            ## Missing Data

            - **Epic admission scores**: 229 encounters (0.27%), **0 fallers** — negligible impact
            - **Morse admission scores**: 416 encounters (0.49%), **1 faller** — negligible impact
            - **Complete-case analysis**: Excludes encounters with any missing score (retains >99% of eligible encounters)

            Missingness among fallers is minimal, justifying the complete-case analytic approach.
            """
        ),
        "Technical specifications": mo.md(
            """
            - **Reporting standard**: TRIPOD+AI (Collins et al. BMJ 2024;385:e078378)
            - **Python**: 3.12 | **Data**: Polars | **Stats**: scipy, sklearn, statsmodels
            - **Reproducibility**: seed = 42 | 2000 stratified bootstrap replicates | alpha = 0.05
            - **Figures**: JAMA Network Open style (Arial, 8pt minimum, 300 DPI)
            - **Source**: Raw Excel export — full audit trail, no pre-computed results
            """
        ),
    })
    return


@app.cell
def _(mo):
    mo.outline(label="Table of Contents")
    return


@app.cell
def _(mo):
    mo.accordion({
        "📖 Interpretation Guide for Co-Authors": mo.md("""
            **Metrics you'll see throughout the notebook:**

            **AUROC** (Area Under ROC Curve):
            - Range: 0.5 (random guessing) to 1.0 (perfect discrimination)
            - Interpretation: Probability that a random faller scores higher than a random non-faller
            - Clinical context: AUROC 0.75-0.80 = good discrimination; >0.80 = excellent; <0.70 = poor
            - Caveat: AUROC alone can be misleading with rare events (1% prevalence) — check sensitivity/specificity too

            **Sensitivity & Specificity**:
            - Sensitivity: % of fallers correctly identified as high-risk
            - Specificity: % of non-fallers correctly identified as low-risk
            - Trade-off: Higher sensitivity = lower specificity (more false positives)
            - Clinical choice: Fall prevention may favor high sensitivity (miss fewer falls), accepting more false alarms

            **PPV & NPV** (Predictive Values):
            - PPV: Probability that a patient flagged high-risk actually falls
            - NPV: Probability that a patient flagged low-risk doesn't fall
            - Depend HEAVILY on prevalence (1% for our dataset)
            - At 1% prevalence, PPV will be low even with high sensitivity — always consider prevalence

            **Calibration**:
            - Do predicted probabilities match observed outcomes?
            - Well-calibrated: 5% predicted risk ≈ 5% observed event rate
            - Slope = 1.0 = perfect calibration; <1.0 = overestimates risk for high-risk patients
            - Intercept = 0 = perfect calibration-in-the-large (no overall bias)

            **NRI** (Net Reclassification Improvement):
            - Does the new model (Epic) reclassify patients into correct risk tiers better than the old model (Morse)?
            - Positive NRI = Epic does better; Negative = Morse does better
            - Report event NRI (fallers) and non-event NRI (non-fallers) separately — combined NRI is misleading

            **IDI** (Integrated Discrimination Improvement):
            - Average improvement in risk separation between models
            - Higher values = better distinction between fallers and non-fallers

            **Decision Curve Analysis (DCA)**:
            - Compares net benefit of using a prediction model vs simple strategies (treat-all, treat-none)
            - Net benefit: True positives gained minus false positives weighted by threshold
            - Model useful where its net benefit > treat-all AND > treat-none

            **Threshold Selection Methods**:
            - Youden: Maximizes (sensitivity + specificity - 1) — statistical optimum, may not be clinically optimal
            - Closest to (0,1): Minimizes distance to perfect discrimination
            - Fixed sensitivity: Set target (e.g., 80%) and find cutoff that achieves it
            - Value-optimizing: Maximizes net monetary benefit given cost of fall vs cost of intervention

            **Timing Strategies**:
            - Admission: First score after admission (clinical actionability, our primary analysis)
            - Before fall: Most recent score before event (LOOKAHEAD BIAS if not careful)
            - Max score: Highest score during entire encounter (LOOKAHEAD BIAS includes post-fall)
            - Mean score: Average score during encounter (LOOKAHEAD BIAS includes post-fall)

            **TRIPOD+AI Reporting** (Collins et al. BMJ 2024):
            - External temporal validation (Type 3): Apply developed model to new population, new time period
            - No refitting, no calibration in primary analysis (assess out-of-the-box performance)
            - Report discrimination, calibration, clinical utility across subgroups

            **Statistical Methods**:
            - 2000 stratified bootstrap: Resample preserving fall/non-fall ratio, compute CIs
            - DeLong test: Paired comparison of AUROC (both models on same encouters)
            - GEE: Generalized estimating equations with exchangeable correlation (handle within-patient clustering)
            - Bonferroni correction: Adjust for multiple comparisons in sensitivity analyses

            **When results differ between admission and max-timing**:
            - Max scores are UPPER BOUND estimates (not clinically actionable)
            - Include post-fall assessments — clinically unrealizable for prospective screening
            - Admission scores represent realistic clinical utility
            - Expected that discrimination improves with more data, but doesn't inform real-time decisions
        """),
    })
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            """
            ## 2. Data Loading & Standardization

            We start with the complete set of hospital encounters exported from Epic's electronic
            health record. Each row represents one hospital stay for one patient. The raw data includes
            demographics, admission/discharge dates, fall events, and risk scores from both the
            Epic predictive model and nurse-completed Morse assessments.
            """
        ),
        mo.accordion({
            "Statistical details: data standardization": mo.md(
                """
                **Column standardization**: 24 columns renamed via `COLUMN_RENAME_MAP` (e.g.,
                `EncounterEpicCsn` -> `encounter_csn`). Datetime parsing for `admission_date`,
                `discharge_date`, `fall_datetime`. String whitespace stripping, empty -> null.

                **Label harmonization**: Race (11 raw -> 4 standardized: White, Black, Asian,
                Other/Unknown), Ethnicity (8 -> 3: Hispanic/Latino, Not Hispanic/Latino, Unknown),
                Gender (6 -> 3: Female, Male, Other/Unknown). Maps defined in `utils/constants.py`.

                **Derived variables**: `fall_flag` = 1 if `fall_datetime` OR `unit_fall_occurred`
                not null. `los_days` = (discharge - admission) in fractional days. Literal "NULL"
                strings in `unit_fall_occurred` converted to actual nulls.
                """
            ),
        }),
    ])
    return


@app.cell
def _(COLUMN_RENAME_MAP, Path, mo, pl):
    mo.vstack([
        mo.md("""
        ### Load Raw Hospital Encounter Data

        **What this cell does**: Reads the complete hospital encounter dataset from the Epic EHR export. Each row represents one hospital stay with demographics, admission/discharge dates, fall events, and both Epic PMFRS and Morse Fall Scale risk scores.

        **Data used**: `data/raw/output_table_v4.xlsx` (raw export from Rush University Medical Center EHR)

        **Key operations**: 
        - Renames 24 columns to standardized lowercase format (e.g., `EncounterEpicCsn` → `encounter_csn`)
        - Parses datetime columns for temporal analysis
        - Creates the foundation dataset for all downstream analyses
        """),
        mo.accordion({
            "📊 Statistical Details": mo.md("""
            **Analysis**: Data standardization and type conversion

            **Parameters**:
            - Column rename map: `COLUMN_RENAME_MAP` from `utils/constants.py`
            - Datetime columns: `admission_date`, `discharge_date`, `fall_datetime`

            **Output**: `df_renamed` (Polars DataFrame with standardized column names and datetime types)

            **Assumptions**:
            - Raw Excel file contains all expected columns from the EHR export
            - Date/datetime columns can be parsed automatically or are already in Date/Datetime format
            - Column names follow a consistent pattern that maps to standardized names
            """)
        }),
    ])

    # Load raw Excel and rename columns
    _raw_path = Path("data/raw/output_table_v4.xlsx")
    _df_raw = pl.read_excel(_raw_path)

    _rename = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in _df_raw.columns}
    df_renamed = _df_raw.rename(_rename)

    # Lowercase any remaining columns not in the map
    _remaining = {
        c: c.lower()
        for c in df_renamed.columns
        if c not in _rename.values() and c != c.lower()
    }
    if _remaining:
        df_renamed = df_renamed.rename(_remaining)

    # Parse datetime columns
    for _col in ["admission_date", "discharge_date", "fall_datetime"]:
        if _col in df_renamed.columns:
            _dtype = df_renamed[_col].dtype
            if _dtype == pl.Date:
                df_renamed = df_renamed.with_columns(
                    pl.col(_col).cast(pl.Datetime).alias(_col)
                )
            elif _dtype != pl.Datetime and _dtype != pl.Datetime("us") and (
                _dtype == pl.String or _dtype == pl.Utf8
            ):
                df_renamed = df_renamed.with_columns(
                    pl.col(_col).str.to_datetime(strict=False).alias(_col)
                )
    return (df_renamed,)


@app.cell
def _(
    ETHNICITY_CLEAN_MAP,
    GENDER_CLEAN_MAP,
    RACE_CLEAN_MAP,
    df_renamed,
    mo,
    pl,
):
    mo.vstack([
        mo.md("""
        ### Standardize Demographic Labels

        **What this cell does**: Cleans up text values by removing extra spaces, converting empty values to missing data, and standardizing demographic categories (race, ethnicity, gender) to enable subgroup analysis.

        **Data used**: 
        - String columns from `df_renamed`
        - Label harmonization maps from `utils/constants.py`

        **Key findings**:
        - Race: 11 raw categories → 4 standardized (White, Black, Asian, Other/Unknown)
        - Ethnicity: 8 raw categories → 3 standardized (Hispanic/Latino, Not Hispanic/Latino, Unknown)
        - Gender: 6 raw categories → 3 standardized (Female, Male, Other/Unknown)
        - These standardized categories enable meaningful subgroup performance comparisons
        """),
        mo.accordion({
            "📊 Statistical Details": mo.md("""
            **Analysis**: Data cleaning and categorical harmonization

            **Parameters**:
            - Whitespace stripping: Applied to all string columns
            - Empty string handling: Converted to `None` (null)
            - Label maps: `RACE_CLEAN_MAP`, `ETHNICITY_CLEAN_MAP`, `GENDER_CLEAN_MAP`

            **Output**: `df_cleaned` (Polars DataFrame with standardized demographic labels)

            **Assumptions**:
            - Raw EHR data uses inconsistent category labels (e.g., "WHITE", "White", "white")
            - Small subgroups can be combined into "Other/Unknown" without losing clinical interpretability
            - Subgroups with <20 fall events will be excluded from fairness analyses

            **Rationale for harmonization**: The original EHR categories included granular distinctions (e.g., separate codes for different Asian ethnicities) that would create too many sparse subgroups for reliable AUROC estimation with only 803 fall events.
            """)
        }),
    ])

    # Strip whitespace from all string columns; empty strings -> null
    _str_cols = [
        c for c in df_renamed.columns if df_renamed[c].dtype in (pl.String, pl.Utf8)
    ]
    _df = df_renamed.with_columns(
        [pl.col(c).str.strip_chars().alias(c) for c in _str_cols]
    ).with_columns(
        [
            pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c)
            for c in _str_cols
        ]
    )

    # Apply label standardization
    if "race" in _df.columns:
        _df = _df.with_columns(
            pl.col("race")
            .replace_strict(RACE_CLEAN_MAP, default="Other/Unknown")
            .alias("race")
        )
    if "ethnicity" in _df.columns:
        _df = _df.with_columns(
            pl.col("ethnicity")
            .replace_strict(ETHNICITY_CLEAN_MAP, default="Unknown")
            .alias("ethnicity")
        )
    if "gender" in _df.columns:
        _df = _df.with_columns(
            pl.col("gender")
            .replace_strict(GENDER_CLEAN_MAP, default="Unknown")
            .alias("gender")
        )

    df_cleaned = _df
    return (df_cleaned,)


@app.cell
def _(df_cleaned, mo, pl):
    mo.vstack([
        mo.md("""
        ### Create Derived Variables

        **What this cell does**: Creates two essential variables for the analysis: the primary outcome variable (`fall_flag`) indicating whether a patient fell, and the length of stay in days (`los_days`) for comparing fallers vs non-fallers.

        **Data used**:
        - `fall_datetime`: Timestamp of documented fall event
        - `unit_fall_occurred`: Hospital unit where fall was recorded
        - `admission_date` and `discharge_date`: For calculating length of stay

        **Key findings**:
        - Fall outcome defined comprehensively: Any encounter with either a fall timestamp OR a recorded fall location is counted as a fall
        - This dual-capture approach minimizes false negatives by catching all documented falls
        - Length of stay calculated as fractional days to enable precise comparisons
        """),
        mo.accordion({
            "📊 Statistical Details": mo.md("""
            **Analysis**: Variable derivation and outcome definition

            **Parameters**:
            - Fall flag logic: Boolean OR (`fall_datetime` is not null OR `unit_fall_occurred` is not null)
            - LOS calculation: `(discharge_date - admission_date)` in seconds, divided by 86,400 (seconds per day)
            - String handling: Literal "NULL" strings converted to actual null values

            **Output**: `df_derived` with two new columns:
            - `fall_flag`: Binary indicator (1 = fall, 0 = no fall)
            - `los_days`: Continuous float, length of stay in fractional days

            **Assumptions**:
            - Falls documented via either timestamp or location represent true fall events
            - Encounters with both fields null did not experience a fall
            - Length of stay is clinically meaningful even for very short stays (fractional days)

            **Clinical context**: The inclusive fall definition (timestamp OR location) captures falls that might be documented in different parts of the EHR by different clinical staff. This is standard practice in fall research to maximize sensitivity.
            """)
        }),
    ])

    # Replace literal "NULL" strings with actual nulls in unit_fall_occurred
    _df = df_cleaned.with_columns(
        pl.when(pl.col("unit_fall_occurred").str.to_uppercase() == "NULL")
        .then(None)
        .otherwise(pl.col("unit_fall_occurred"))
        .alias("unit_fall_occurred")
    )
    df_derived = _df.with_columns(
        [
            # fall_flag: 1 if fall_datetime OR unit_fall_occurred is not null
            pl.when(
                pl.col("fall_datetime").is_not_null()
                | pl.col("unit_fall_occurred").is_not_null()
            )
            .then(1)
            .otherwise(0)
            .alias("fall_flag"),
            # los_days: fractional days
            (
                (pl.col("discharge_date") - pl.col("admission_date")).dt.total_seconds()
                / 86400
            ).alias("los_days"),
        ]
    )
    return (df_derived,)


@app.cell
def _(df_derived, mo, pl):
    _n = df_derived.height
    _n_falls = df_derived.filter(pl.col("fall_flag") == 1).height
    _n_patients = df_derived["mrn"].n_unique()
    _miss_epic = df_derived["epic_score_admission"].null_count()
    _miss_morse = df_derived["morse_score_admission"].null_count()
    mo.md(
        f"""
        ### Data Quality Summary

        - **Total encounters**: {_n:,}
        - **Falls**: {_n_falls:,} ({_n_falls / _n * 100:.1f}%)
        - **Unique patients**: {_n_patients:,}
        - **Missing Epic admission score**: {_miss_epic:,} ({_miss_epic / _n * 100:.2f}%)
        - **Missing Morse admission score**: {_miss_morse:,} ({_miss_morse / _n * 100:.2f}%)
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            """
            ## 3. Cohort Definition & CONSORT Flow Diagram

            Not every encounter in the raw data belongs in our study. We excluded children
            (age <18), outpatient/observation stays, encounters missing a discharge department,
            and encounters missing either risk score. After these filters, we have the analytic
            cohort used for all analyses.
            """
        ),
        mo.accordion({
            "Statistical details: cohort exclusions": mo.md(
                """
                **Exclusion criteria** (applied sequentially):
                1. Age < 18 years at admission
                2. Non-inpatient accommodation code (observation, outpatient, etc.)
                3. Missing discharge department (indicates incomplete encounter)
                4. Missing Epic PMFRS admission score OR missing Morse Fall Scale admission score

                **Complete-case rationale**: Only 1 faller was lost due to missing Morse score;
                229 encounters lacked Epic scores (0 fallers). Complete-case analysis is
                appropriate given negligible missingness among events.

                **Selection bias assessment**: Excluded encounters had similar demographics
                to included encounters (age, sex distributions comparable).
                """
            ),
        }),
    ])
    return


@app.cell
def _(Path, df_derived, mo, pl):
    mo.vstack([
        mo.md("""
        ### Apply Sequential Exclusion Criteria

        **What this cell does**: Applies exclusion criteria step-by-step to arrive at the final analytic cohort. Removes children (age <18), encounters missing discharge department, and encounters missing either risk score. Tracks exactly how many patients and falls are lost at each step. All encounters in the dataset are from inpatient-admitted patients regardless of billing accommodation code.

        **Data used**:
        - `df_derived`: Raw encounter dataset with all patients
        - `age`: Patient age at admission
        - `discharge_department`: Clinical unit at discharge
        - `epic_score_admission` and `morse_score_admission`: Risk scores at admission

        **Key findings**:
        - Only 1 faller lost due to missing Morse score; 0 fallers lost due to missing Epic score
        - Complete case ascertainment with negligible event loss
        """),
        mo.accordion({
            "Statistical Details": mo.md("""
            **Analysis**: Sequential cohort selection with complete-case analysis

            **Exclusion criteria** (applied in order):
            1. **Age <18 years**: Pediatric population excluded (study focuses on adult inpatient falls)
            2. **Missing discharge department**: Indicates incomplete encounter record
            3. **Missing risk scores**: Encounters without both Epic PMFRS and Morse Fall Scale admission scores excluded

            **Parameters**:
            - Complete-case threshold: Both scores must be present

            **Output**:
            - `df_analytic`: Final cohort for all analyses
            - `flow_counts`: Dictionary tracking exclusions at each step
            - `data/processed/analytic.parquet`: Saved for compatibility with other notebooks

            **Assumptions**:
            - All encounters are from inpatient-admitted patients (confirmed by data team)
            - Complete-case analysis is appropriate given minimal missingness among events (1 faller)
            - Excluded encounters had similar demographics to included encounters (selection bias assessment)
            - Patients with missing scores are missing completely at random (MCAR)

            **Missing data profile**:
            - Missing Epic admission score: 229 encounters (0.27%), **0 fallers**
            - Missing Morse admission score: 416 encounters (0.49%), **1 faller**
            - This negligible missingness among fallers justifies complete-case approach

            **TRIPOD+AI compliance**: This cell implements Item 5b (study dates and locations) and Item 12 (flow of participants).
            """)
        }),
    ])

    # Step-by-step exclusion tracking
    _df = df_derived
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

    # Step 4: characterise missing scores
    n_miss_epic = _df_eligible.filter(pl.col("epic_score_admission").is_null()).height
    n_miss_epic_falls = int(
        _df_eligible.filter(pl.col("epic_score_admission").is_null())["fall_flag"].sum()
    )
    n_miss_morse = _df_eligible.filter(pl.col("morse_score_admission").is_null()).height
    n_miss_morse_falls = int(
        _df_eligible.filter(pl.col("morse_score_admission").is_null())["fall_flag"].sum()
    )

    # Step 5: complete-case cohort
    df_analytic = _df_eligible.filter(
        pl.col("epic_score_admission").is_not_null()
        & pl.col("morse_score_admission").is_not_null()
    )
    n_analytic = df_analytic.height
    n_analytic_falls = int(df_analytic["fall_flag"].sum())
    n_analytic_nofall = n_analytic - n_analytic_falls

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

    # Write analytic parquet for compatibility with other notebooks
    _parquet_dir = Path("data/processed")
    _parquet_dir.mkdir(parents=True, exist_ok=True)
    df_analytic.write_parquet(_parquet_dir / "analytic.parquet")
    return df_analytic, flow_counts


@app.cell
def _(flow_counts, mo, pl):
    _rows = [
        {"Step": "All encounters (raw)", "N": flow_counts["n_raw"], "Falls": flow_counts["n_raw_falls"]},
        {"Step": "Excluded: age < 18", "N": flow_counts["n_excl_age"], "Falls": flow_counts["n_excl_age_falls"]},
        {"Step": "After age exclusion", "N": flow_counts["n_after_age"], "Falls": flow_counts["n_after_age_falls"]},
        {"Step": "Excluded: missing discharge dept", "N": flow_counts["n_excl_dept"], "Falls": flow_counts["n_excl_dept_falls"]},
        {"Step": "Eligible encounters", "N": flow_counts["n_eligible"], "Falls": flow_counts["n_eligible_falls"]},
        {"Step": "Missing Epic score", "N": flow_counts["n_miss_epic"], "Falls": flow_counts["n_miss_epic_falls"]},
        {"Step": "Missing Morse score", "N": flow_counts["n_miss_morse"], "Falls": flow_counts["n_miss_morse_falls"]},
        {"Step": "Analytic cohort", "N": flow_counts["n_analytic"], "Falls": flow_counts["n_analytic_falls"]},
    ]
    _summary_table = pl.DataFrame(_rows)
    mo.vstack([
        mo.md("### Exclusion Summary"),
        mo.ui.table(_summary_table),
    ])
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
        _fig, _ax = plt.subplots(figsize=(7.0, 10.0))
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1)
        _ax.axis("off")

        _BOX_FACE = "#FFFFFF"
        _BOX_EDGE = "#333333"
        _ARROW_CLR = "#333333"
        _EXCL_FACE = "#F5F5F5"
        _SPLIT_FALL = "#FFF0F0"
        _SPLIT_NO = "#EBF0F7"

        _MX = 0.37
        _MW = 0.38
        _EX = 0.81
        _EW = 0.30
        _BH = 0.068
        _SH = 0.055

        _YL = {
            "raw": 0.935, "age": 0.820, "dept": 0.700,
            "eligible": 0.570, "missing": 0.420, "analytic": 0.260, "split": 0.090,
        }

        def _box(xc, yc, w, h, text, face=_BOX_FACE, edge=_BOX_EDGE, fs=9, fw="normal", lw=0.75):
            _ax.add_patch(FancyBboxPatch(
                (xc - w / 2, yc - h / 2), w, h,
                boxstyle="round,pad=0.008", facecolor=face, edgecolor=edge,
                linewidth=lw, transform=_ax.transAxes, zorder=2,
            ))
            _ax.text(xc, yc, text, transform=_ax.transAxes, ha="center", va="center",
                     fontsize=fs, fontweight=fw, linespacing=1.35, zorder=3)

        def _arrow(xs, ys, xe, ye, cs="arc3,rad=0.0", lw=0.75):
            _ax.add_patch(FancyArrowPatch(
                (xs, ys), (xe, ye), arrowstyle="-|>", mutation_scale=9,
                color=_ARROW_CLR, linewidth=lw, connectionstyle=cs,
                transform=_ax.transAxes, zorder=1,
            ))

        def _excl_arrow(y):
            _arrow(_MX + _MW / 2, y, _EX - _EW / 2, y)

        def _vert(y_top_level, y_bot_level, h_top=_BH, h_bot=_BH):
            _arrow(_MX, _YL[y_top_level] - h_top / 2 - 0.003,
                   _MX, _YL[y_bot_level] + h_bot / 2 + 0.003)

        # Node 1: All encounters
        _box(_MX, _YL["raw"], _MW, _BH,
             f"All encounters in study period\n(N\u202f=\u202f{flow_counts['n_raw']:,}; {flow_counts['n_raw_falls']:,} falls)",
             fs=9, fw="bold")

        # Node 2: After age exclusion
        _vert("raw", "age")
        _box(_MX, _YL["age"], _MW, _BH,
             f"Age \u226518 years\n(n\u202f=\u202f{flow_counts['n_after_age']:,}; {flow_counts['n_after_age_falls']:,} falls)", fs=9)
        _box(_EX, _YL["age"], _EW, _SH,
             f"Excluded: age <18\n(n\u202f=\u202f{flow_counts['n_excl_age']:,})", face=_EXCL_FACE, fs=8)
        _excl_arrow(_YL["age"])

        # Node 3: After discharge-dept filter
        _vert("age", "dept")
        _box(_MX, _YL["dept"], _MW, _BH,
             f"Discharge department recorded\n(n\u202f=\u202f{flow_counts['n_eligible']:,}; {flow_counts['n_eligible_falls']:,} falls)", fs=9)
        _box(_EX, _YL["dept"], _EW, _SH,
             f"Excluded: missing discharge\ndepartment (n\u202f=\u202f{flow_counts['n_excl_dept']:,})", face=_EXCL_FACE, fs=8)
        _excl_arrow(_YL["dept"])

        # Node 5: Eligible cohort
        _vert("dept", "eligible")
        _box(_MX, _YL["eligible"], _MW, _BH + 0.008,
             f"Eligible encounters\n(N\u202f=\u202f{flow_counts['n_eligible']:,}; {flow_counts['n_eligible_falls']:,} falls [{flow_counts['n_eligible_falls'] / flow_counts['n_eligible'] * 100:.1f}%])",
             fs=9, fw="bold", edge="#2166AC", lw=1.2)

        # Node 6a/6b: Missing score branches
        _MISS_LX = 0.195
        _MISS_RX = 0.565
        _MISS_W = 0.295
        _miss_bh = _BH + 0.005

        _box(_MISS_LX, _YL["missing"], _MISS_W, _miss_bh,
             f"Missing Epic PMFRS\nadmission score\n(n\u202f=\u202f{flow_counts['n_miss_epic']:,}; {flow_counts['n_miss_epic_falls']:,} falls)",
             face=_EXCL_FACE, fs=8)
        _box(_MISS_RX, _YL["missing"], _MISS_W, _miss_bh,
             f"Missing Morse Fall Scale\nadmission score\n(n\u202f=\u202f{flow_counts['n_miss_morse']:,}; {flow_counts['n_miss_morse_falls']:,} fall{'s' if flow_counts['n_miss_morse_falls'] != 1 else ''})",
             face=_EXCL_FACE, fs=8)

        _elig_bot = _YL["eligible"] - (_BH + 0.008) / 2 - 0.004
        _miss_top = _YL["missing"] + _miss_bh / 2 + 0.004
        _arrow(_MX - 0.06, _elig_bot, _MISS_LX, _miss_top, cs="arc3,rad=0.12")
        _arrow(_MX + 0.06, _elig_bot, _MISS_RX, _miss_top, cs="arc3,rad=-0.12")

        _ax.text(_MX, (_YL["eligible"] + _YL["missing"]) / 2 + 0.025,
                 "Excluded for missing scores (complete-case analysis)",
                 transform=_ax.transAxes, ha="center", va="center", fontsize=8,
                 fontstyle="italic", color="#555555")

        # Node 7: Analytic cohort
        _analytic_bh = _BH + 0.012
        _box(_MX, _YL["analytic"], _MW + 0.02, _analytic_bh,
             f"Analytic cohort (complete case)\n(N\u202f=\u202f{flow_counts['n_analytic']:,}; {flow_counts['n_analytic_falls']:,} falls [{flow_counts['n_analytic_falls'] / flow_counts['n_analytic'] * 100:.1f}%])",
             fs=10, fw="bold", edge="#2166AC", lw=1.4)

        _miss_bot = _YL["missing"] - _miss_bh / 2 - 0.004
        _analy_top = _YL["analytic"] + _analytic_bh / 2 + 0.004
        _arrow(_MISS_LX, _miss_bot, _MX - 0.06, _analy_top, cs="arc3,rad=-0.12")
        _arrow(_MISS_RX, _miss_bot, _MX + 0.06, _analy_top, cs="arc3,rad=0.12")
        _ax.text(_MX, (_YL["missing"] + _YL["analytic"]) / 2 - 0.01, "Both scores present",
                 transform=_ax.transAxes, ha="center", va="center", fontsize=8,
                 fontstyle="italic", color="#555555")

        # Node 8a/8b: Fall vs No-fall split
        _FALL_X = 0.195
        _NOFALL_X = 0.565
        _SPLIT_W = 0.295
        _box(_FALL_X, _YL["split"], _SPLIT_W, _BH,
             f"Fall\n(n\u202f=\u202f{flow_counts['n_analytic_falls']:,})",
             face=_SPLIT_FALL, fs=9, fw="bold", edge="#B2182B", lw=1.0)
        _box(_NOFALL_X, _YL["split"], _SPLIT_W, _BH,
             f"No fall\n(n\u202f=\u202f{flow_counts['n_analytic_nofall']:,})",
             face=_SPLIT_NO, fs=9, fw="bold", edge="#2166AC", lw=1.0)

        _analy_bot = _YL["analytic"] - _analytic_bh / 2 - 0.004
        _split_top = _YL["split"] + _BH / 2 + 0.004
        _arrow(_MX - 0.06, _analy_bot, _FALL_X, _split_top, cs="arc3,rad=0.12")
        _arrow(_MX + 0.06, _analy_bot, _NOFALL_X, _split_top, cs="arc3,rad=-0.12")

        _fig.text(
            0.5, -0.04,
            "eFigure 4. CONSORT-style cohort flow diagram",
            ha="center", va="top", fontsize=10, fontweight="bold",
        )

        save_figure(_fig, "efigure4_cohort_flow", formats=("pdf", "png"), bbox_inches=None, pad_inches=0.15)
    return


@app.cell
def _(Path, flow_counts, mo, pl):
    # A1: Export cohort flow counts CSV (matches NB02 output)
    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _flow_df = pl.DataFrame([{"metric": k, "value": v} for k, v in flow_counts.items()])
    _flow_df.write_csv(_out_dir / "efigure4_cohort_flow_counts.csv")
    mo.md("**eFigure 4 saved** to `outputs/figures/efigure4_cohort_flow.pdf` and `.png`.\n\n**Flow counts saved** to `outputs/tables/efigure4_cohort_flow_counts.csv`.")
    return


@app.cell
def _(flow_counts, mo):
    # B3: Cohort stats dashboard
    mo.hstack([
        mo.stat(f"{flow_counts['n_analytic']:,}", label="Analytic cohort", bordered=True),
        mo.stat(f"{flow_counts['n_analytic_falls']:,}", label="Falls", bordered=True),
        mo.stat(f"{flow_counts['n_analytic_falls'] / flow_counts['n_analytic'] * 100:.1f}%", label="Event rate", bordered=True),
    ], justify="center", gap=1)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            """
            ## 3b. Fall Event Characterization — Who Falls, Where, and When

            This section describes the fall events themselves: study period, locations, temporal
            patterns, and timing relative to admission. These descriptive analyses support
            TRIPOD+AI Item 5b (study dates) and provide clinical context for interpreting
            model performance.
            """
        ),
        mo.accordion({
            "Statistical details": mo.md(
                """
                **Study period**: Extracted from admission/discharge/fall datetime fields.
                **Location**: Grouped by `unit_fall_occurred` (where fall happened, not admitting unit).
                **Fall rate**: Computed as falls / encounters per admitting department.
                **Temporal**: Hour-of-day and day-of-week from `fall_datetime`; fallers with missing
                datetime are excluded from temporal analyses but included in all other analyses.
                **Admission-to-fall**: Time in hours from `admission_date` to `fall_datetime`; only
                encounters with positive lead time included.
                """
            ),
        }),
    ])
    return


@app.cell
def _(df_analytic, mo, pl):
    # Study period stat cards
    _adm_min = df_analytic["admission_date"].min()
    _adm_max = df_analytic["admission_date"].max()
    _dc_max = df_analytic["discharge_date"].max()
    _fall_dates = df_analytic.filter(pl.col("fall_datetime").is_not_null())["fall_datetime"]
    _fall_min = _fall_dates.min()
    _fall_max = _fall_dates.max()
    _duration_days = (_dc_max - _adm_min).total_seconds() / 86400
    _duration_months = _duration_days / 30.44

    mo.hstack([
        mo.stat(f"{_adm_min.strftime('%b %Y')} – {_dc_max.strftime('%b %Y')}", label="Study period", bordered=True),
        mo.stat(f"{_duration_months:.0f}", label="Duration, months", bordered=True),
        mo.stat(_fall_min.strftime("%Y-%m-%d"), label="Earliest fall", bordered=True),
        mo.stat(_fall_max.strftime("%Y-%m-%d"), label="Latest fall", bordered=True),
    ], justify="center", gap=1)
    return


@app.cell
def _(FIG_DOUBLE_COL, Path, df_analytic, mo, pl, plt, save_figure):
    # Where falls occur — top 10 units
    _fall_units = (
        df_analytic.filter(pl.col("unit_fall_occurred").is_not_null())
        .group_by("unit_fall_occurred")
        .agg(pl.len().alias("n_falls"))
        .sort("n_falls", descending=True)
        .head(10)
    )
    _total_falls = df_analytic.filter(pl.col("fall_flag") == 1).height
    _fall_units = _fall_units.with_columns(
        (pl.col("n_falls") / _total_falls * 100).round(1).alias("pct_of_falls")
    )

    _units = _fall_units["unit_fall_occurred"].to_list()[::-1]
    _counts = _fall_units["n_falls"].to_list()[::-1]

    _fig, _ax = plt.subplots(figsize=FIG_DOUBLE_COL)
    _ax.barh(_units, _counts, color="#4A7C8A", edgecolor="white", linewidth=0.3)
    _ax.set_xlabel("Number of falls")
    _ax.set_ylabel("Unit")
    for _i, _v in enumerate(_counts):
        _ax.text(_v + max(_counts) * 0.01, _i, str(_v), va="center", fontsize=8)
    _fig.tight_layout()
    _fig.text(
        0.5, -0.04,
        "Top 10 units by fall count",
        ha="center", va="top", fontsize=10, fontweight="bold",
    )
    save_figure(_fig, "fall_locations", bbox_inches=None, pad_inches=0.15)

    # Export CSV
    _out = Path("outputs/tables")
    _out.mkdir(parents=True, exist_ok=True)
    _fall_units.write_csv(_out / "fall_location_summary.csv")

    mo.vstack([
        mo.md("### Where Falls Occur (Top 10 Units)"),
        mo.ui.table(_fall_units),
        mo.image(src="outputs/figures/fall_locations.png", width=700),
    ])
    return


@app.cell
def _(FIG_DOUBLE_COL, Path, df_analytic, mo, pl, plt, save_figure):
    # Fall rate by admitting department — top 10 by volume
    _dept_stats = (
        df_analytic.group_by("admitting_department")
        .agg([
            pl.len().alias("n_encounters"),
            pl.col("fall_flag").sum().alias("n_falls"),
        ])
        .with_columns(
            (pl.col("n_falls") / pl.col("n_encounters") * 100).round(2).alias("fall_rate_pct")
        )
        .sort("n_encounters", descending=True)
        .head(10)
    )

    _sorted = _dept_stats.sort("fall_rate_pct", descending=False)
    _depts = _sorted["admitting_department"].to_list()
    _rates = _sorted["fall_rate_pct"].to_list()

    _fig, _ax = plt.subplots(figsize=FIG_DOUBLE_COL)
    _ax.barh(_depts, _rates, color="#4A7C8A", edgecolor="white", linewidth=0.3)
    _ax.set_xlabel("Fall rate, %")
    _ax.set_ylabel("Department")
    for _i, _v in enumerate(_rates):
        _ax.text(_v + max(_rates) * 0.01, _i, f"{_v:.1f}%", va="center", fontsize=8)
    _fig.tight_layout()
    _fig.text(
        0.5, -0.04,
        "Fall rate by admitting department (top 10 by volume)",
        ha="center", va="top", fontsize=10, fontweight="bold",
    )
    save_figure(_fig, "department_fall_rates", bbox_inches=None, pad_inches=0.15)

    _dept_stats.write_csv(Path("outputs/tables/department_fall_rates.csv"))

    mo.vstack([
        mo.md("### Fall Rate by Admitting Department (Top 10 by Volume)"),
        mo.ui.table(_dept_stats),
        mo.image(src="outputs/figures/department_fall_rates.png", width=700),
    ])
    return


@app.cell
def _(FIG_DOUBLE_COL, df_analytic, mo, pl, plt, save_figure):
    # Temporal patterns — two-panel figure
    _with_dt = df_analytic.filter(pl.col("fall_datetime").is_not_null())
    _n_with_dt = _with_dt.height
    _n_without_dt = df_analytic.filter(
        (pl.col("fall_flag") == 1) & pl.col("fall_datetime").is_null()
    ).height

    _hours = _with_dt["fall_datetime"].dt.hour().to_numpy()
    _weekdays = _with_dt["fall_datetime"].dt.weekday().to_numpy()
    _day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE_COL)

    _ax1.hist(_hours, bins=range(25), color="#4A7C8A", edgecolor="white", linewidth=0.3)
    _ax1.set_xlabel("Hour of day")
    _ax1.set_ylabel("Number of falls")
    _ax1.text(-0.14, 1.06, "A", transform=_ax1.transAxes, fontsize=10, fontweight="bold", va="top")
    _ax1.set_xticks(range(0, 24, 4))

    _day_counts = [int((_weekdays == d).sum()) for d in range(7)]
    _ax2.bar(_day_labels, _day_counts, color="#4A7C8A", edgecolor="white", linewidth=0.3)
    _ax2.set_xlabel("Day of week")
    _ax2.set_ylabel("Number of falls")
    _ax2.text(-0.14, 1.06, "B", transform=_ax2.transAxes, fontsize=10, fontweight="bold", va="top")

    _fig.tight_layout()
    _fig.text(
        0.5, -0.04,
        "Temporal patterns of fall events",
        ha="center", va="top", fontsize=10, fontweight="bold",
    )
    save_figure(_fig, "fall_temporal", bbox_inches=None, pad_inches=0.15)

    # Export CSV
    _hour_df = pl.DataFrame({"hour": list(range(24)), "n_falls": [int((_hours == h).sum()) for h in range(24)]})
    _day_df = pl.DataFrame({"day_of_week": _day_labels, "n_falls": _day_counts})
    _hour_df.write_csv("outputs/tables/fall_temporal_hourly.csv")
    _day_df.write_csv("outputs/tables/fall_temporal_daily.csv")

    mo.vstack([
        mo.md(
            f"### Temporal Patterns\n\n"
            f"{_n_with_dt:,} falls with valid datetime; "
            f"{_n_without_dt:,} fallers excluded (missing `fall_datetime`)."
        ),
        mo.image(src="outputs/figures/fall_temporal.png", width=700),
    ])
    return


@app.cell
def _(FIG_SINGLE_COL, df_analytic, mo, np, pl, plt, save_figure):
    # Admission-to-fall timing
    _fallers_dt = df_analytic.filter(
        (pl.col("fall_flag") == 1) & pl.col("fall_datetime").is_not_null()
    ).with_columns(
        ((pl.col("fall_datetime") - pl.col("admission_date")).dt.total_seconds() / 3600)
        .alias("hours_to_fall")
    ).filter(pl.col("hours_to_fall") > 0)

    _h2f = _fallers_dt["hours_to_fall"].to_numpy()
    _med = np.median(_h2f)
    _q1, _q3 = np.percentile(_h2f, [25, 75])

    _fig, _ax = plt.subplots(figsize=FIG_SINGLE_COL)
    _ax.hist(_h2f, bins=50, color="#4A7C8A", edgecolor="white", linewidth=0.3)
    _ax.axvline(_med, color="#B2182B", linestyle="--", linewidth=1.0)
    _ax.set_xlabel("Hours from admission to fall")
    _ax.set_ylabel("Number of falls")
    _ax.text(
        0.97, 0.95,
        f"Median: {_med:.0f}h [{_q1:.0f}\u2013{_q3:.0f}]",
        transform=_ax.transAxes, ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8"},
    )
    _fig.tight_layout()
    _fig.text(
        0.5, -0.04,
        "Admission-to-fall timing",
        ha="center", va="top", fontsize=10, fontweight="bold",
    )
    save_figure(_fig, "admission_to_fall_hours", bbox_inches=None, pad_inches=0.15)

    mo.md(
        f"**Admission-to-fall timing**: median {_med:.0f}h [{_q1:.0f}\u2013{_q3:.0f}], "
        f"n = {len(_h2f):,} (positive lead time)."
    )
    return


@app.cell
def _(FIG_SINGLE_COL, df_analytic, mo, np, pl, plt, save_figure):
    # LOS comparison: boxplot by fall status (matches NB01)
    _los_fallers = df_analytic.filter(pl.col("fall_flag") == 1)["los_days"].to_numpy()
    _los_nonfallers = df_analytic.filter(pl.col("fall_flag") == 0)["los_days"].to_numpy()
    _cap = np.percentile(np.concatenate([_los_fallers, _los_nonfallers]), 95)

    _fig, _ax = plt.subplots(figsize=FIG_SINGLE_COL)
    _bp = _ax.boxplot(
        [_los_nonfallers[_los_nonfallers <= _cap], _los_fallers[_los_fallers <= _cap]],
        labels=["Non-fallers", "Fallers"],
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "#B2182B", "linewidth": 1.0},
    )
    _bp["boxes"][0].set_facecolor("#D1E5F0")
    _bp["boxes"][1].set_facecolor("#FDDBC7")
    _ax.set_ylabel("Length of stay, days")
    _ax.set_title("LOS by fall status (capped at 95th pct)", fontweight="bold")
    _fig.tight_layout()
    save_figure(_fig, "los_comparison")

    mo.md(
        f"**LOS comparison**: fallers median {np.median(_los_fallers):.1f}d vs "
        f"non-fallers {np.median(_los_nonfallers):.1f}d."
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "**Interpretation guide**: Review which units have the highest fall "
            "concentrations — these may warrant targeted intervention regardless of "
            "model scores. Hour-of-day and day-of-week patterns may reflect staffing "
            "levels, patient activity cycles, or documentation timing. The median "
            "admission-to-fall interval indicates whether admission-time screening "
            "scores are capturing risk during the highest-risk window — if most falls "
            "occur days after admission, the admission score may have decayed in relevance."
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            """
            ## 4. Descriptive Statistics — Table 1

            Table 1 describes who is in our study. We compare fallers to non-fallers across age,
            sex, race, length of stay, and risk scores. The 'SMD' column flags variables where
            the two groups differ meaningfully (SMD > 0.10).
            """
        ),
        mo.accordion({
            "Statistical details: Table 1 methods": mo.md(
                """
                **SMD**: Cohen's d for continuous variables; max pairwise proportion difference
                for categorical variables. SMD > 0.10 indicates potential imbalance.

                **Tests**: Welch t-test (age, scores), Wilcoxon rank-sum (LOS — skewed),
                chi-square (categorical). With 99:1 class imbalance, even small absolute
                differences can be statistically significant — SMD is more informative than
                p-values.
                """
            ),
        }),
    ])
    return


@app.cell
def _(np, stats):
    def smd_continuous(s1, s2):
        """Cohen's d-style SMD for two Polars Series."""
        _m1, _sd1 = s1.mean(), s1.std()
        _m2, _sd2 = s2.mean(), s2.std()
        if _m1 is None or _m2 is None or _sd1 is None or _sd2 is None:
            return float("nan")
        _pooled = np.sqrt((_sd1**2 + _sd2**2) / 2)
        return float(abs(_m1 - _m2) / _pooled) if _pooled > 0 else float("nan")

    def smd_categorical(df_grp, col):
        """Max pairwise proportion difference SMD for categorical variable."""
        import polars as _pl
        _fall = df_grp.filter(_pl.col("fall_flag") == 1)
        _nofall = df_grp.filter(_pl.col("fall_flag") == 0)
        _vals = df_grp[col].drop_nulls().unique().to_list()
        if len(_vals) < 2:
            return float("nan")
        _n1 = max(_fall.height, 1)
        _n0 = max(_nofall.height, 1)
        _smds = []
        for _v in _vals:
            _p1 = _fall.filter(_pl.col(col) == _v).height / _n1
            _p0 = _nofall.filter(_pl.col(col) == _v).height / _n0
            _denom = np.sqrt((_p1 * (1 - _p1) + _p0 * (1 - _p0)) / 2)
            if _denom > 0:
                _smds.append(abs(_p1 - _p0) / _denom)
        return float(np.max(_smds)) if _smds else float("nan")

    def ttest_p(s1, s2):
        _a = s1.drop_nulls().to_numpy()
        _b = s2.drop_nulls().to_numpy()
        if len(_a) < 2 or len(_b) < 2:
            return float("nan")
        return float(stats.ttest_ind(_a, _b, equal_var=False).pvalue)

    def wilcoxon_p(s1, s2):
        _a = s1.drop_nulls().to_numpy()
        _b = s2.drop_nulls().to_numpy()
        if len(_a) < 2 or len(_b) < 2:
            return float("nan")
        return float(stats.mannwhitneyu(_a, _b, alternative="two-sided").pvalue)

    def chisq_p(df_grp, col):
        import polars as _pl
        _ct = (
            df_grp.group_by(["fall_flag", col])
            .agg(_pl.len().alias("n"))
            .pivot(index=col, on="fall_flag", values="n")
            .fill_null(0)
        )
        _num_cols = [c for c in _ct.columns if c != col]
        if len(_num_cols) < 2:
            return float("nan")
        _matrix = _ct.select(_num_cols).to_numpy()
        _chi2, _p, _dof, _exp = stats.chi2_contingency(_matrix)
        return float(_p)

    def fmt_pval(p):
        if np.isnan(p):
            return "\u2014"
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    def fmt_smd(s):
        if np.isnan(s):
            return "\u2014"
        return f"{s:.3f}"

    return (
        chisq_p,
        fmt_pval,
        fmt_smd,
        smd_categorical,
        smd_continuous,
        ttest_p,
        wilcoxon_p,
    )


@app.cell
def _(
    Path,
    chisq_p,
    df_analytic,
    fmt_pval,
    fmt_smd,
    np,
    pl,
    smd_categorical,
    smd_continuous,
    ttest_p,
    wilcoxon_p,
):
    _df = df_analytic
    _df_fall = _df.filter(pl.col("fall_flag") == 1)
    _df_nofall = _df.filter(pl.col("fall_flag") == 0)
    _n_total = _df.height
    _n_fall = _df_fall.height
    _n_nofall = _df_nofall.height

    _rows: list[dict] = []

    # Header row
    _rows.append({"Variable": "Encounters, n", "Overall": str(_n_total),
                  "No fall": str(_n_nofall), "Fall": str(_n_fall), "SMD": "\u2014", "P value": "\u2014"})

    # Age
    _age_all = _df["age"].drop_nulls()
    _age_fall = _df_fall["age"].drop_nulls()
    _age_nofall = _df_nofall["age"].drop_nulls()
    _rows.append({"Variable": "Age, years \u2014 mean \u00b1 SD",
                  "Overall": f"{_age_all.mean():.1f} \u00b1 {_age_all.std():.1f}",
                  "No fall": f"{_age_nofall.mean():.1f} \u00b1 {_age_nofall.std():.1f}",
                  "Fall": f"{_age_fall.mean():.1f} \u00b1 {_age_fall.std():.1f}",
                  "SMD": fmt_smd(smd_continuous(_age_fall, _age_nofall)),
                  "P value": fmt_pval(ttest_p(_age_fall, _age_nofall))})

    # LOS
    _los_all = _df["los_days"].drop_nulls()
    _los_fall = _df_fall["los_days"].drop_nulls()
    _los_nofall = _df_nofall["los_days"].drop_nulls()
    _rows.append({"Variable": "Length of stay, days \u2014 median [IQR]",
                  "Overall": f"{_los_all.median():.1f} [{np.percentile(_los_all.to_numpy(), 25):.1f}\u2013{np.percentile(_los_all.to_numpy(), 75):.1f}]",
                  "No fall": f"{_los_nofall.median():.1f} [{np.percentile(_los_nofall.to_numpy(), 25):.1f}\u2013{np.percentile(_los_nofall.to_numpy(), 75):.1f}]",
                  "Fall": f"{_los_fall.median():.1f} [{np.percentile(_los_fall.to_numpy(), 25):.1f}\u2013{np.percentile(_los_fall.to_numpy(), 75):.1f}]",
                  "SMD": fmt_smd(smd_continuous(_los_fall, _los_nofall)),
                  "P value": fmt_pval(wilcoxon_p(_los_fall, _los_nofall))})

    # Categorical helper
    def _cat_rows(col, section_label):
        _out = []
        _vals = _df[col].drop_nulls().value_counts().sort("count", descending=True)[col].to_list()
        _out.append({"Variable": section_label, "Overall": "", "No fall": "", "Fall": "",
                     "SMD": fmt_smd(smd_categorical(_df, col)), "P value": fmt_pval(chisq_p(_df, col))})
        for _v in _vals:
            _n_all_v = _df.filter(pl.col(col) == _v).height
            _n_nofall_v = _df_nofall.filter(pl.col(col) == _v).height
            _n_fall_v = _df_fall.filter(pl.col(col) == _v).height
            _n_denom = max(_df[col].drop_nulls().len(), 1)
            _n_nofall_denom = max(_df_nofall[col].drop_nulls().len(), 1)
            _n_fall_denom = max(_df_fall[col].drop_nulls().len(), 1)
            _out.append({"Variable": f"  {_v}",
                         "Overall": f"{_n_all_v:,} ({_n_all_v / _n_denom * 100:.1f}%)",
                         "No fall": f"{_n_nofall_v:,} ({_n_nofall_v / _n_nofall_denom * 100:.1f}%)",
                         "Fall": f"{_n_fall_v:,} ({_n_fall_v / _n_fall_denom * 100:.1f}%)",
                         "SMD": "\u2014", "P value": "\u2014"})
        return _out

    _rows.extend(_cat_rows("gender", "Sex \u2014 n (%)"))
    _rows.extend(_cat_rows("race", "Race \u2014 n (%)"))
    _rows.extend(_cat_rows("ethnicity", "Ethnicity \u2014 n (%)"))

    # Score variables
    for _col, _label in [
        ("epic_score_admission", "Epic PMFRS at admission \u2014 mean \u00b1 SD"),
        ("morse_score_admission", "Morse Fall Scale at admission \u2014 mean \u00b1 SD"),
        ("epic_score_max", "Epic PMFRS, max during encounter \u2014 mean \u00b1 SD"),
        ("morse_score_max", "Morse Fall Scale, max during encounter \u2014 mean \u00b1 SD"),
        ("epic_score_mean", "Epic PMFRS, mean during encounter \u2014 mean \u00b1 SD"),
        ("morse_score_mean", "Morse Fall Scale, mean during encounter \u2014 mean \u00b1 SD"),
    ]:
        if _col not in _df.columns:
            continue
        _s_all = _df[_col].drop_nulls()
        _s_fall = _df_fall[_col].drop_nulls()
        _s_nofall = _df_nofall[_col].drop_nulls()
        _rows.append({"Variable": _label,
                      "Overall": f"{_s_all.mean():.2f} \u00b1 {_s_all.std():.2f}",
                      "No fall": f"{_s_nofall.mean():.2f} \u00b1 {_s_nofall.std():.2f}",
                      "Fall": f"{_s_fall.mean():.2f} \u00b1 {_s_fall.std():.2f}",
                      "SMD": fmt_smd(smd_continuous(_s_fall, _s_nofall)),
                      "P value": fmt_pval(ttest_p(_s_fall, _s_nofall))})

    # Before-fall scores (fallers only)
    for _col, _label in [
        ("epic_score_before_fall", "Epic PMFRS before fall (fallers only) \u2014 mean \u00b1 SD"),
        ("morse_score_before_fall", "Morse Fall Scale before fall (fallers only) \u2014 mean \u00b1 SD"),
    ]:
        if _col not in _df.columns:
            continue
        _s_fall = _df_fall[_col].drop_nulls()
        if _s_fall.len() > 0:
            _rows.append({"Variable": _label, "Overall": "\u2014", "No fall": "\u2014",
                          "Fall": f"{_s_fall.mean():.2f} \u00b1 {_s_fall.std():.2f}",
                          "SMD": "\u2014", "P value": "\u2014"})

    table1_df = pl.DataFrame(_rows)

    # Export CSV
    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    table1_df.write_csv(_out_dir / "table1.csv")
    return (table1_df,)


@app.cell
def _(mo, table1_df):
    mo.vstack([
        mo.md("### Table 1. Patient Characteristics by Fall Status"),
        mo.ui.table(table1_df),
        mo.md("**Saved**: `outputs/tables/table1.csv`"),
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            """
            ## 5. Primary Discrimination Analysis — Table 2 & Figure 1

            The key question: which tool is better at telling fallers apart from non-fallers?
            We measure this with the AUROC \u2014 a number between 0.5 (useless) and 1.0 (perfect).
            We also test seven different ways to pick the 'cutoff score' that divides patients
            into high-risk and low-risk groups.
            """
        ),
        mo.accordion({
            "Statistical details: discrimination methods": mo.md(
                """
                **AUROC**: Via DeLong variance (Sun & Xu 2014). **AUPRC**: Reported alongside
                AUROC because at 1% prevalence, AUROC alone can be misleading (high AUROC with
                very low PPV).

                **Bootstrap**: 2000 stratified resamples (preserving fall_flag ratio, seed=42)
                for all CIs.

                **Logistic recalibration**: Single-predictor LogisticRegression mapping ordinal
                score to P(fall). Required for DCA, NRI, and calibration. Does NOT change AUROC
                (monotonic transform preserves ranking).

                **Value-optimizing threshold**: NMB with cost_fall=$14K, cost_intervention=$200,
                effectiveness Beta(40,60), QALY loss 0.0036, WTP $100K/QALY (Parsons JAMIA 2023).

                **Treatment paradox**: High-risk patients receive interventions that prevent falls,
                attenuating observed discrimination \u2014 AUROCs are conservative.
                """
            ),
        }),
    ])
    return


@app.cell
def _(df_analytic, logistic_recalibration, mo):
    mo.vstack([
        mo.md("""
        ### Prepare Data for Statistical Analysis — Recalibration

        **What this cell does**: Converts raw risk scores into calibrated fall probabilities. This step is necessary because Epic scores (0–100) and Morse scores (0–125) are not calibrated probabilities — they represent risk rank ordering, not actual fall likelihood.

        **Data used**:
        - `fall_flag`: Binary outcome (1 = patient fell, 0 = no fall)
        - `epic_score_admission`: First Epic score after admission (0–100 scale)
        - `morse_score_admission`: First Morse score after admission (0–125 scale)

        **Key findings**:
        - Logistic recalibration maps each raw score to a fall probability
        - This does NOT change discrimination (AUROC unchanged) — it's a monotonic transformation
        - Recalibrated probabilities are needed for DCA, NRI, IDI, and calibration analysis
        - Clinicians see raw scores; recalibrated probabilities are for analytic purposes
        """),
        mo.accordion({
            "📊 Statistical Details": mo.md("""
            **Analysis**: Logistic recalibration via single-predictor logistic regression

            **Method**: 
            - Fit `LogisticRegression()` with raw score as single predictor
            - Apply `.predict_proba()` to convert scores to probabilities
            - No intercept adjustment (model learns optimal intercept from data)

            **Parameters**:
            - Model: `sklearn.linear_model.LogisticRegression`
            - Input: Raw scores (continuous for Epic, discrete for Morse)
            - Output: `epic_prob`, `morse_prob` (probabilities in [0, 1])

            **Output variables**:
            - `y_true`: Ground truth fall outcomes (numpy array)
            - `epic_scores`, `morse_scores`: Raw scores (numpy arrays)
            - `epic_prob`, `morse_prob`: Recalibrated probabilities
            - `epic_lr`, `morse_lr`: Fitted logistic regression models (saved for threshold probability conversion)

            **Assumptions**:
            - Linear relationship between log-odds of fall and raw score
            - The recalibration model is appropriate for external validation (standard practice)
            - Raw scores preserve ranking; recalibration only affects probability estimation

            **Why recalibration is necessary**:
            The Epic PMFRS training target included fall interventions (target 0/1/2), not just falls. The raw score reflects this combined outcome, not pure fall probability. Morse scores sum 6 bedside assessment items — the total is a risk index, not a probability.
            """)
        }),
    ])

    # Extract arrays at sklearn boundary + logistic recalibration
    y_true = df_analytic["fall_flag"].to_numpy()
    epic_scores = df_analytic["epic_score_admission"].to_numpy()
    morse_scores = df_analytic["morse_score_admission"].to_numpy()

    epic_prob, epic_lr = logistic_recalibration(epic_scores, y_true)
    morse_prob, morse_lr = logistic_recalibration(morse_scores, y_true)
    return (
        epic_lr,
        epic_prob,
        epic_scores,
        morse_lr,
        morse_prob,
        morse_scores,
        y_true,
    )


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
    # Recalibration mapping figure (matches NB03b)
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

        _fig.subplots_adjust(wspace=0.35, bottom=0.18)

        _fig.text(
            0.5, -0.06,
            "Score-to-probability mapping via logistic recalibration",
            ha="center", va="top", fontsize=10, fontweight="bold",
        )

    save_figure(_fig, "recalibration_mapping", bbox_inches=None, pad_inches=0.15)
    _fig
    return


@app.cell
def _(delong_ci, delong_roc_test, epic_scores, morse_scores, y_true):
    from sklearn.metrics import average_precision_score

    epic_auc, epic_ci_lo, epic_ci_hi = delong_ci(y_true, epic_scores)
    morse_auc, morse_ci_lo, morse_ci_hi = delong_ci(y_true, morse_scores)
    epic_auprc = float(average_precision_score(y_true, epic_scores))
    morse_auprc = float(average_precision_score(y_true, morse_scores))
    delong_p = delong_roc_test(y_true, epic_scores, morse_scores)
    return (
        delong_p,
        epic_auc,
        epic_auprc,
        epic_ci_hi,
        epic_ci_lo,
        morse_auc,
        morse_auprc,
        morse_ci_hi,
        morse_ci_lo,
    )


@app.cell
def _(delong_p, epic_auc, epic_auprc, mo, morse_auc, morse_auprc):
    # B3: Discrimination stats dashboard
    mo.hstack([
        mo.stat(f"{epic_auc:.3f}", label="Epic AUROC", bordered=True),
        mo.stat(f"{morse_auc:.3f}", label="Morse AUROC", bordered=True),
        mo.stat(f"{epic_auprc:.3f}", label="Epic AUPRC", bordered=True),
        mo.stat(f"{morse_auprc:.3f}", label="Morse AUPRC", bordered=True),
        mo.stat(f"{delong_p:.4f}", label="DeLong p-value", bordered=True),
    ], justify="center", gap=1)
    return


@app.cell
def _(
    EPIC_2TIER_HIGH,
    EPIC_3TIER_HIGH,
    EPIC_3TIER_MEDIUM,
    MFS_HIGH,
    MFS_MODERATE,
    classification_metrics_at_threshold,
    closest_topleft_threshold,
    epic_prob,
    epic_scores,
    fixed_sensitivity_threshold,
    morse_prob,
    morse_scores,
    value_optimizing_threshold,
    y_true,
    youden_threshold,
):
    # Epic thresholds (data-driven, on ordinal scores)
    epic_t_youden = youden_threshold(y_true, epic_scores)
    epic_t_topleft = closest_topleft_threshold(y_true, epic_scores)
    epic_t_sens60 = fixed_sensitivity_threshold(y_true, epic_scores, 0.60)
    epic_t_sens80 = fixed_sensitivity_threshold(y_true, epic_scores, 0.80)
    epic_t_value, _, _ = value_optimizing_threshold(y_true, epic_prob)

    # Morse thresholds
    morse_t_youden = youden_threshold(y_true, morse_scores)
    morse_t_topleft = closest_topleft_threshold(y_true, morse_scores)
    morse_t_sens60 = fixed_sensitivity_threshold(y_true, morse_scores, 0.60)
    morse_t_sens80 = fixed_sensitivity_threshold(y_true, morse_scores, 0.80)
    morse_t_value, _, _ = value_optimizing_threshold(y_true, morse_prob)

    # Standard clinical cutoffs
    _epic_threshold_labels = [
        ("Youden index", epic_t_youden, epic_scores),
        ("Closest to (0,1)", epic_t_topleft, epic_scores),
        ("Fixed sensitivity 60%", epic_t_sens60, epic_scores),
        ("Fixed sensitivity 80%", epic_t_sens80, epic_scores),
        ("Value-optimizing (NMB)", epic_t_value, epic_prob),
        ("Standard cutoff: Epic \u226535 (3-tier medium)", float(EPIC_3TIER_MEDIUM), epic_scores),
        ("Standard cutoff: Epic \u226570 (3-tier high)", float(EPIC_3TIER_HIGH), epic_scores),
        ("Standard cutoff: Epic \u226550 (2-tier high)", float(EPIC_2TIER_HIGH), epic_scores),
    ]

    _morse_threshold_labels = [
        ("Youden index", morse_t_youden, morse_scores),
        ("Closest to (0,1)", morse_t_topleft, morse_scores),
        ("Fixed sensitivity 60%", morse_t_sens60, morse_scores),
        ("Fixed sensitivity 80%", morse_t_sens80, morse_scores),
        ("Value-optimizing (NMB)", morse_t_value, morse_prob),
        ("Standard cutoff: MFS \u226525", float(MFS_MODERATE), morse_scores),
        ("Standard cutoff: MFS \u226545", float(MFS_HIGH), morse_scores),
    ]

    epic_metrics_by_threshold: list[dict] = []
    for _label, _t, _score_arr in _epic_threshold_labels:
        _m = classification_metrics_at_threshold(y_true, _score_arr, _t)
        _m["label"] = _label
        epic_metrics_by_threshold.append(_m)

    morse_metrics_by_threshold: list[dict] = []
    for _label, _t, _score_arr in _morse_threshold_labels:
        _m = classification_metrics_at_threshold(y_true, _score_arr, _t)
        _m["label"] = _label
        morse_metrics_by_threshold.append(_m)
    return epic_metrics_by_threshold, morse_metrics_by_threshold


@app.cell
def _(
    N_BOOTSTRAP,
    RANDOM_SEED,
    epic_scores,
    mo,
    morse_scores,
    stratified_bootstrap,
    y_true,
):
    mo.vstack([
        mo.md("""
        ### Compute Bootstrap Confidence Intervals

        **What this cell does**: Quantifies uncertainty in all our discrimination estimates by resampling the data 2,000 times with replacement. This gives us the 95% confidence intervals reported throughout the paper.

        **Data used**:
        - `y_true`: Ground truth fall outcomes
        - `epic_scores` and `morse_scores`: Raw risk scores from both tools

        **Key findings**:
        - **Stratified bootstrap**: Each resample preserves the exact fall/non-fall ratio (1:79), ensuring stable metric calculation
        - **BCa intervals**: Bias-corrected and accelerated intervals are more accurate than simple percentile intervals for skewed distributions
        - **Seed=42**: Fixed random seed ensures reproducibility — running this cell again produces identical results
        """),
        mo.accordion({
            "📊 Statistical Details": mo.md("""
            **Analysis**: Stratified bootstrap resampling with BCa confidence intervals

            **Method**:
            - **Stratification**: Each bootstrap sample preserves the fall/non-fall ratio exactly
            - **Resamples**: 2,000 independent bootstrap replicates
            - **BCa correction**: Adjusts for median bias and skewness in the bootstrap distribution
            - **Paired bootstrap**: Both models evaluated on the same resamples for paired comparison

            **Parameters**:
            - `N_BOOTSTRAP`: 2,000 resamples (defined in `utils/constants.py`)
            - `RANDOM_SEED`: 42 (ensures reproducibility)
            - Stratification variable: `fall_flag`

            **Output**: `boot_results` dictionary containing:
            - `auc_a`: AUROC estimate and CI for Epic PMFRS
            - `auc_b`: AUROC estimate and CI for Morse Fall Scale
            - `auprc_a`: AUPRC estimate and CI for Epic PMFRS
            - `auprc_b`: AUPRC estimate and CI for Morse Fall Scale
            - `delta_auc`: AUROC difference and CI

            **Why stratified bootstrap**: With rare events (1.27% prevalence), standard bootstrap may yield resamples with zero or very few events, breaking AUROC calculation. Stratification guarantees at least the original number of events in each resample.

            **Why BCa**: For asymmetric sampling distributions (common with AUROC), BCa intervals provide better coverage than percentile intervals. The acceleration term corrects for skewness.

            **Reference**: Efron B. Better bootstrap confidence intervals. *JASA*. 1987;82(397):171-185.
            """)
        }),
    ])

    boot_results = stratified_bootstrap(
        y_true, epic_scores, pred_b=morse_scores, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED
    )
    return (boot_results,)


@app.cell
def _(
    Path,
    boot_results,
    delong_p,
    epic_auc,
    epic_auprc,
    epic_ci_hi,
    epic_ci_lo,
    epic_metrics_by_threshold: list[dict],
    morse_auc,
    morse_auprc,
    morse_ci_hi,
    morse_ci_lo,
    morse_metrics_by_threshold: list[dict],
    pl,
):
    def _fmt_ci(est, lo, hi, decimals=3):
        d = f".{decimals}f"
        return f"{est:{d}} ({lo:{d}}\u2013{hi:{d}})"

    def _metrics_to_row(m, model):
        return {
            "Model": model, "Threshold label": m["label"],
            "Threshold value": round(m["threshold"], 2),
            "Sensitivity, %": round(m["sensitivity"] * 100, 1),
            "Specificity, %": round(m["specificity"] * 100, 1),
            "Flag rate, %": round(m["flag_rate"], 1),
            "PPV, %": round(m["ppv"] * 100, 1),
            "NPV, %": round(m["npv"] * 100, 1),
            "NNE": round(m["nne"], 1) if m["nne"] != float("inf") else None,
        }

    _epic_header = {
        "Model": "Epic PMFRS", "Threshold label": "Overall",
        "Threshold value": None, "Sensitivity, %": None, "Specificity, %": None,
        "Flag rate, %": None, "PPV, %": None, "NPV, %": None, "NNE": None,
        "AUROC (95% CI)": _fmt_ci(epic_auc, epic_ci_lo, epic_ci_hi),
        "AUPRC (95% CI)": _fmt_ci(epic_auprc, boot_results["auprc_a"]["ci_lower"], boot_results["auprc_a"]["ci_upper"]),
        "DeLong p": round(delong_p, 4),
    }
    _morse_header = {
        "Model": "Morse Fall Scale", "Threshold label": "Overall",
        "Threshold value": None, "Sensitivity, %": None, "Specificity, %": None,
        "Flag rate, %": None, "PPV, %": None, "NPV, %": None, "NNE": None,
        "AUROC (95% CI)": _fmt_ci(morse_auc, morse_ci_lo, morse_ci_hi),
        "AUPRC (95% CI)": _fmt_ci(morse_auprc, boot_results["auprc_b"]["ci_lower"], boot_results["auprc_b"]["ci_upper"]),
        "DeLong p": round(delong_p, 4),
    }

    _threshold_rows = []
    for _m in epic_metrics_by_threshold:
        _row = _metrics_to_row(_m, "Epic PMFRS")
        _row.update({"AUROC (95% CI)": None, "AUPRC (95% CI)": None, "DeLong p": None})
        _threshold_rows.append(_row)
    for _m in morse_metrics_by_threshold:
        _row = _metrics_to_row(_m, "Morse Fall Scale")
        _row.update({"AUROC (95% CI)": None, "AUPRC (95% CI)": None, "DeLong p": None})
        _threshold_rows.append(_row)

    table2_df = pl.DataFrame(
        [_epic_header, _morse_header] + _threshold_rows,
        infer_schema_length=len(_threshold_rows) + 2,
    )

    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    table2_df.write_csv(Path("outputs/tables/table2.csv"))
    return (table2_df,)


@app.cell
def _(mo, table2_df):
    mo.vstack([
        mo.md("### Table 2. Discrimination Performance: Epic PMFRS vs Morse Fall Scale"),
        mo.ui.table(table2_df),
        mo.md("**Saved**: `outputs/tables/table2.csv`"),
    ])
    return


@app.cell
def _(
    COLORS,
    FIG_MULTI_PANEL,
    JAMA_STYLE,
    epic_prob,
    matplotlib,
    morse_prob,
    np,
    plt,
    save_figure,
    y_true,
):
    from sklearn.metrics import roc_curve as _roc_curve

    def _threshold_curve(y_true_arr, probs):
        _fpr, _tpr, _thresholds = _roc_curve(y_true_arr, probs)
        _sens, _spec, _ppv, _npv = [], [], [], []
        for _t in _thresholds:
            _pred = probs >= _t
            _tp = int((_pred & (y_true_arr == 1)).sum())
            _fp = int((_pred & (y_true_arr == 0)).sum())
            _fn = int((~_pred & (y_true_arr == 1)).sum())
            _tn = int((~_pred & (y_true_arr == 0)).sum())
            _sens.append(_tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0)
            _spec.append(_tn / (_tn + _fp) if (_tn + _fp) > 0 else 0.0)
            _ppv.append(_tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0)
            _npv.append(_tn / (_tn + _fn) if (_tn + _fn) > 0 else 0.0)
        return _thresholds, _sens, _spec, _ppv, _npv

    _epic_t, _epic_sens, _epic_spec, _epic_ppv, _epic_npv = _threshold_curve(y_true, epic_prob)
    _morse_t, _morse_sens, _morse_spec, _morse_ppv, _morse_npv = _threshold_curve(y_true, morse_prob)
    _x_max = min(max(float(np.max(epic_prob)), float(np.max(morse_prob))) * 1.15, 1.0)

    with matplotlib.rc_context(JAMA_STYLE):
        _fig, _axes = plt.subplots(2, 2, figsize=FIG_MULTI_PANEL)
        _fig.subplots_adjust(hspace=0.42, wspace=0.38, bottom=0.08)

        _panels = [
            ("Sensitivity", _epic_sens, _morse_sens),
            ("Specificity", _epic_spec, _morse_spec),
            ("PPV", _epic_ppv, _morse_ppv),
            ("NPV", _epic_npv, _morse_npv),
        ]
        for _idx, ((_title, _ey, _my), _ax, _lbl) in enumerate(
            zip(_panels, _axes.flat, ["A", "B", "C", "D"], strict=False)
        ):
            _ax.plot(_epic_t, _ey, color=COLORS["epic"], linewidth=1.0, label="Epic PMFRS")
            _ax.plot(_morse_t, _my, color=COLORS["morse"], linewidth=1.0, label="Morse Fall Scale", linestyle="--")
            _ax.set_xlabel("Predicted probability", fontsize=9)
            _ax.set_ylabel(_title, fontsize=9)
            _ax.set_ylim(-0.02, 1.05)
            _ax.set_xlim(0, _x_max)
            _ax.tick_params(labelsize=8)
            _ax.text(-0.18, 1.06, _lbl, transform=_ax.transAxes, fontsize=10, fontweight="bold", va="top")

        _handles, _labels_leg = _axes.flat[0].get_legend_handles_labels()
        _fig.legend(_handles, _labels_leg, loc="lower center", ncol=2, fontsize=8,
                    frameon=False, bbox_to_anchor=(0.5, -0.04))
        _fig.text(
            0.5, -0.10,
            "Figure 1. Classification metrics across score thresholds:\n"
            "Epic PMFRS vs Morse Fall Scale",
            ha="center", va="top", fontsize=10, fontweight="bold",
        )
        save_figure(_fig, "figure1_discrimination", bbox_inches=None, pad_inches=0.15)
    return


@app.cell
def _(
    delong_p,
    epic_auc,
    epic_ci_hi,
    epic_ci_lo,
    mo,
    morse_auc,
    morse_ci_hi,
    morse_ci_lo,
):
    mo.md(f"""
    ### Discrimination Summary

    - **Epic PMFRS AUROC**: {epic_auc:.3f} ({epic_ci_lo:.3f}\u2013{epic_ci_hi:.3f})
    - **Morse Fall Scale AUROC**: {morse_auc:.3f} ({morse_ci_lo:.3f}\u2013{morse_ci_hi:.3f})
    - **DeLong p-value**: {delong_p:.4f}

    **Figure 1 saved** to `outputs/figures/figure1_discrimination.pdf` and `.png`.
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            """
            ## 6. Calibration Analysis \u2014 eFigures 1-2

            Calibration checks whether the predicted risk numbers are trustworthy. If the model
            says a patient has a 5% chance of falling, do about 5 out of 100 similar patients
            actually fall? Good calibration means the numbers match reality.
            """
        ),
        mo.accordion({
            "Statistical details: calibration metrics": mo.md(
                """
                **CITL** (calibration-in-the-large, logit scale, ideal=0): Overall recalibration offset.
                **Calibration slope** (Cox's method, ideal=1.0): Degree of overfitting/underfitting.
                **ICI** (integrated calibration index, LOWESS frac=0.3, ideal=0): Average absolute
                difference between predicted and LOWESS-smoothed observed probabilities.

                Both models recalibrated via logistic recalibration in Section 5 \u2014 standard for
                externally developed models. Reference: Van Calster BMC Medicine 2019;17:230.
                """
            ),
        }),
    ])
    return


@app.cell
def _(matplotlib, np, plt):
    import statsmodels.api as _sm

    def make_calibration_plot(y_true_arr, y_prob_arr, cal_metrics, model_label, line_color, jama_style, fig_size):
        _lowess_result = _sm.nonparametric.lowess(y_true_arr, y_prob_arr, frac=0.3, it=3, return_sorted=True)
        _p_max = float(np.max(np.concatenate([y_prob_arr, _lowess_result[:, 1]])))
        _axis_max = np.ceil(_p_max * 100) / 100 + 0.005
        _axis_max = max(_axis_max, _p_max * 1.15)
        _axis_max = min(_axis_max, 1.0)

        _rug_y_event = -0.06 * _axis_max
        _rug_y_nonevent = -0.13 * _axis_max
        _rug_height = max(0.025 * _axis_max, 0.003)

        with matplotlib.rc_context(jama_style):
            _fig, _ax = plt.subplots(figsize=fig_size)
            _ref_x = np.linspace(0, _axis_max, 100)
            _ax.plot(_ref_x, _ref_x, color="#BBBBBB", linewidth=0.6, linestyle="--", zorder=1, label="Perfect calibration")
            _ax.plot(_lowess_result[:, 0], _lowess_result[:, 1], color=line_color, linewidth=1.8, zorder=3, label=model_label)

            # Grouped calibration points — decile bins
            _n_bins = min(10, len(np.unique(y_prob_arr)))
            _bin_edges = np.percentile(y_prob_arr, np.linspace(0, 100, _n_bins + 1))
            _bin_edges = np.unique(_bin_edges)
            _bin_x, _bin_y = [], []
            for _i in range(len(_bin_edges) - 1):
                _mask = (y_prob_arr >= _bin_edges[_i]) & (y_prob_arr < _bin_edges[_i + 1]) if _i < len(_bin_edges) - 2 else (y_prob_arr >= _bin_edges[_i])
                if _mask.sum() > 0:
                    _bin_x.append(float(np.mean(y_prob_arr[_mask])))
                    _bin_y.append(float(np.mean(y_true_arr[_mask])))
            _ax.scatter(_bin_x, _bin_y, color=line_color, s=18, zorder=4, edgecolors="white", linewidth=0.5)

            _events_mask = y_true_arr == 1
            _rng = np.random.RandomState(42)
            _non_idx = np.where(~_events_mask)[0]
            if len(_non_idx) > 5000:
                _non_idx = _rng.choice(_non_idx, size=5000, replace=False)
            _ax.vlines(y_prob_arr[_non_idx], ymin=_rug_y_nonevent - _rug_height / 2,
                       ymax=_rug_y_nonevent + _rug_height / 2, color="#777777", linewidth=0.3, alpha=0.4, zorder=2)
            _ax.vlines(y_prob_arr[_events_mask], ymin=_rug_y_event - _rug_height / 2,
                       ymax=_rug_y_event + _rug_height / 2, color=line_color, linewidth=0.5, alpha=0.7, zorder=2)

            _anno = f"CITL = {cal_metrics['citl']:.2f}\nSlope = {cal_metrics['calibration_slope']:.2f}\nICI = {cal_metrics['ici']:.4f}"
            _ax.text(0.95, 0.95, _anno, transform=_ax.transAxes, fontsize=8, va="top", ha="right",
                     linespacing=1.4, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", linewidth=0.5))
            _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False, handlelength=1.5)
            _ax.set_xlabel("Predicted probability", fontsize=9)
            _ax.set_ylabel("Observed event rate", fontsize=9)
            _ax.set_xlim(-0.001, _axis_max)
            _ax.set_ylim(_rug_y_nonevent - _rug_height, _axis_max)
            _ax.tick_params(labelsize=8)
            plt.subplots_adjust(bottom=0.25, right=0.85)
            _ax.text(1.01, _rug_y_event, "Falls", transform=_ax.get_yaxis_transform(), fontsize=8, va="center", ha="left", color=line_color)
            _ax.text(1.01, _rug_y_nonevent, "Non-falls", transform=_ax.get_yaxis_transform(), fontsize=8, va="center", ha="left", color="#777777")

        return _fig

    return (make_calibration_plot,)


@app.cell
def _(
    COLORS,
    FIG_SINGLE_COL,
    JAMA_STYLE,
    Path,
    calibration_metrics,
    epic_prob,
    make_calibration_plot,
    morse_prob,
    pl,
    save_figure,
    y_true,
):
    epic_cal = calibration_metrics(y_true, epic_prob, lowess_frac=0.3)
    morse_cal = calibration_metrics(y_true, morse_prob, lowess_frac=0.3)

    _efig1 = make_calibration_plot(y_true, epic_prob, epic_cal, "Epic PMFRS", COLORS["epic"], JAMA_STYLE, FIG_SINGLE_COL)
    _efig1.text(
        0.5, -0.22,
        "eFigure 1. Calibration plot: Epic PMFRS",
        ha="center", va="top", fontsize=10, fontweight="bold",
    )
    save_figure(_efig1, "efigure1_calibration_epic", bbox_inches=None, pad_inches=0.15)

    _efig2 = make_calibration_plot(y_true, morse_prob, morse_cal, "Morse Fall Scale", COLORS["morse"], JAMA_STYLE, FIG_SINGLE_COL)
    _efig2.text(
        0.5, -0.22,
        "eFigure 2. Calibration plot: Morse Fall Scale",
        ha="center", va="top", fontsize=10, fontweight="bold",
    )
    save_figure(_efig2, "efigure2_calibration_morse", bbox_inches=None, pad_inches=0.15)

    # Calibration summary
    _cal_rows = [
        {"Model": "Epic PMFRS", "CITL": round(epic_cal["citl"], 3),
         "Calibration slope": round(epic_cal["calibration_slope"], 3), "ICI": round(epic_cal["ici"], 4)},
        {"Model": "Morse Fall Scale", "CITL": round(morse_cal["citl"], 3),
         "Calibration slope": round(morse_cal["calibration_slope"], 3), "ICI": round(morse_cal["ici"], 4)},
    ]
    cal_summary_df = pl.DataFrame(_cal_rows)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    cal_summary_df.write_csv(Path("outputs/tables/etable5_calibration_summary.csv"))
    return (cal_summary_df,)


@app.cell
def _(cal_summary_df, mo):
    mo.vstack([
        mo.md("### Calibration Summary"),
        mo.ui.table(cal_summary_df),
        mo.md("**eFigures 1-2 saved** to `outputs/figures/`. **Table saved** to `outputs/tables/etable5_calibration_summary.csv`."),
    ])
    return


@app.cell
def _(Path, mo):
    # B6: Side-by-side calibration figure display
    _efig1 = Path("outputs/figures/efigure1_calibration_epic.png")
    _efig2 = Path("outputs/figures/efigure2_calibration_morse.png")
    mo.hstack([
        mo.vstack([mo.md("**eFigure 1.** Epic PMFRS"), mo.image(src=_efig1)]) if _efig1.exists() else mo.md("*eFigure 1 not yet generated*"),
        mo.vstack([mo.md("**eFigure 2.** Morse Fall Scale"), mo.image(src=_efig2)]) if _efig2.exists() else mo.md("*eFigure 2 not yet generated*"),
    ], gap=1)
    return


@app.cell
def _(mo):
    mo.accordion({
        "🎯 Key Findings: Calibration Analysis": mo.md("""
            **After logistic recalibration** (mapping raw scores to actual fall probabilities):
            - Epic PMFRS calibration-in-the-large (intercept): See calibration table — [interpretation: bias magnitude]
            - Epic PMFRS slope: See calibration table — [interpretation: slope < 1 indicates overfitting to training population]
            - Morse Fall Scale calibration-in-the-large: See calibration table
            - Morse Fall Scale slope: See calibration table

            **Integrated Calibration Index (ICI)** — mean absolute difference between predicted and observed probabilities:
            - Epic PMFRS: See calibration table
            - Morse Fall Scale: See calibration table

            **Interpretation**:
            - Both tools benefit from logistic recalibration when applied to our population (common in external validation)
            - Raw scores (Epic 0–100, Morse 0–125) are not calibrated probabilities — they represent risk rank ordering
            - Recalibration preserves discrimination (AUROC unchanged) while improving probability estimations
            - Clinicians use the raw scores directly; recalibrated probabilities are primarily for analytic and comparative purposes

            **Clinical context**: Epic's model expects continuous monitoring scores over full encounters; our validation uses admission-time screening only. This timing difference is expected to affect calibration but not discrimination.
        """),
    })
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **What the calibration results mean**

            After logistic recalibration (a standard statistical adjustment for externally developed
            models), both tools' predicted probabilities were reasonably close to observed fall rates.
            Note that clinicians see the raw scores (Epic 0–100, Morse 0–125), not recalibrated
            probabilities — recalibration is an analytic step that maps scores to actual risk
            estimates. Without this adjustment, the raw scores would overestimate or underestimate
            actual fall probability. The recalibration step is necessary for DCA, NRI, and any
            analysis that interprets scores as probabilities.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            """
            ## 7. Decision Curve Analysis \u2014 Figure 3

            Even if a tool can separate fallers from non-fallers, is it actually *useful* for
            making clinical decisions? Decision curve analysis answers this by comparing the
            'net benefit' of using a prediction tool versus two simple strategies: treat every
            patient as high-risk, or treat no one.
            """
        ),
        mo.accordion({
            "Statistical details: DCA methodology": mo.md(
                """
                **Net benefit** = (TP/N) - (FP/N) x (pt/(1-pt)) where pt = threshold probability.
                Threshold range 0-10% (clinically relevant for 1% prevalence).

                **dcurves library** v1.1.7 requires pandas DataFrame \u2014 convert at boundary via
                `.to_pandas()`. DCA-derived threshold range: interval where model NB > treat-all
                AND > treat-none. Reference: Vickers AJ *Diagn Progn Res* 2019;3:18.
                """
            ),
        }),
    ])
    return


@app.cell
def _(
    DCA_THRESHOLD_MAX,
    DCA_THRESHOLD_MIN,
    DCA_THRESHOLD_STEP,
    df_analytic,
    epic_prob,
    morse_prob,
    np,
    pl,
):
    from dcurves import dca as _dca

    _pdf = df_analytic.select(["fall_flag"]).with_columns([
        pl.Series("epic_prob", epic_prob),
        pl.Series("morse_prob", morse_prob),
    ]).to_pandas()

    df_dca = _dca(
        data=_pdf,
        outcome="fall_flag",
        modelnames=["epic_prob", "morse_prob"],
        thresholds=np.arange(DCA_THRESHOLD_MIN, DCA_THRESHOLD_MAX, DCA_THRESHOLD_STEP),
    )
    return (df_dca,)


@app.cell
def _(
    COLORS,
    EPIC_2TIER_HIGH,
    EPIC_3TIER_HIGH,
    EPIC_3TIER_MEDIUM,
    JAMA_STYLE,
    MFS_HIGH,
    MFS_MODERATE,
    df_dca,
    epic_lr,
    morse_lr,
    plt,
    save_figure,
):
    def _get_nb(model_name):
        _sub = df_dca[df_dca["model"] == model_name].sort_values("threshold")
        return _sub["threshold"].to_numpy(), _sub["net_benefit"].to_numpy()

    _thresh_epic, _nb_epic = _get_nb("epic_prob")
    _thresh_morse, _nb_morse = _get_nb("morse_prob")
    _thresh_all, _nb_all = _get_nb("all")
    _thresh_none, _nb_none = _get_nb("none")

    with plt.rc_context(JAMA_STYLE):
        _fig3, _ax3 = plt.subplots(figsize=(7.0, 4.5))

        _ax3.plot(_thresh_all * 100, _nb_all, color=COLORS["treat_all"], linewidth=1.0, linestyle="--", label="Treat all", zorder=1)
        _ax3.plot(_thresh_none * 100, _nb_none, color=COLORS["treat_none"], linewidth=0.75, linestyle=":", label="Treat none", zorder=1)
        _ax3.plot(_thresh_epic * 100, _nb_epic, color=COLORS["epic"], linewidth=1.5, label="Epic PMFRS", zorder=3)
        _ax3.plot(_thresh_morse * 100, _nb_morse, color=COLORS["morse"], linewidth=1.5, linestyle="--", label="Morse Fall Scale", zorder=2)

        _ax3.set_xlabel("Threshold probability (%)", fontsize=9)
        _ax3.set_ylabel("Net benefit", fontsize=9)
        _ax3.set_xlim(0, 10)
        _y_min = max(min(_nb_epic.min(), _nb_morse.min()) - 0.002, -0.01)
        _y_max = max(_nb_epic.max(), _nb_morse.max(), _nb_all.max()) + 0.002
        _ax3.set_ylim(_y_min, _y_max)

        # Standard cutoff annotations (matching NB06)
        _morse_prob_25 = float(morse_lr.predict_proba([[float(MFS_MODERATE)]])[0, 1])
        _morse_prob_45 = float(morse_lr.predict_proba([[float(MFS_HIGH)]])[0, 1])
        _epic_prob_35 = float(epic_lr.predict_proba([[float(EPIC_3TIER_MEDIUM)]])[0, 1])
        _epic_prob_50 = float(epic_lr.predict_proba([[float(EPIC_2TIER_HIGH)]])[0, 1])
        _epic_prob_70 = float(epic_lr.predict_proba([[float(EPIC_3TIER_HIGH)]])[0, 1])

        _cutoff_annotations = [
            (_morse_prob_25, "MFS \u226525", COLORS["morse"]),
            (_morse_prob_45, "MFS \u226545", COLORS["morse"]),
            (_epic_prob_35, "Epic \u226535", COLORS["epic"]),
            (_epic_prob_50, "Epic \u226550", COLORS["epic"]),
            (_epic_prob_70, "Epic \u226570", COLORS["epic"]),
        ]
        _placed_x = []
        _y_range = _y_max - _y_min
        for _cutoff_prob, _cutoff_label, _color in _cutoff_annotations:
            _x_pct = _cutoff_prob * 100
            if _x_pct > 10:
                continue
            _ax3.axvline(_x_pct, color=_color, linewidth=0.6, linestyle=":", alpha=0.6, zorder=1)
            _y_pos = _y_max - 0.001
            for _px in _placed_x:
                if abs(_x_pct - _px) < 0.6:
                    _y_pos -= _y_range * 0.12
            _ax3.text(_x_pct, _y_pos, _cutoff_label, fontsize=8, color=_color,
                      ha="center", va="top", rotation=90, alpha=0.8)
            _placed_x.append(_x_pct)

        _ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8, frameon=False)
        _ax3.axhline(0, color="black", linewidth=0.4, zorder=0)
        _ax3.set_xticks([0, 2, 4, 6, 8, 10])
        _ax3.set_xticklabels(["0%", "2%", "4%", "6%", "8%", "10%"], fontsize=8)
        _fig3.subplots_adjust(bottom=0.22)

        _fig3.text(
            0.5, -0.06,
            "Figure 3. Decision curve analysis: Epic PMFRS vs Morse Fall Scale",
            ha="center", va="top", fontsize=10, fontweight="bold",
        )

    save_figure(_fig3, "figure3_dca", bbox_inches=None, pad_inches=0.15)
    return


@app.cell
def _(Path, df_dca, extract_dca_threshold_range, json, mo):
    _epic_dca_range = extract_dca_threshold_range(df_dca, "epic_prob")
    _morse_dca_range = extract_dca_threshold_range(df_dca, "morse_prob")

    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables/dca_threshold_ranges.json").write_text(
        json.dumps({"epic": _epic_dca_range, "morse": _morse_dca_range}, indent=2)
    )

    def _fmt(val):
        return f"{val:.4f}" if val is not None else "\u2014"

    def _fmt_pct(val):
        return f"{val * 100:.2f}%" if val is not None else "\u2014"

    mo.md(
        f"""
        ### DCA-Derived Threshold Ranges

        | Model | Lower | Upper |
        |---|---|---|
        | Epic PMFRS | {_fmt_pct(_epic_dca_range['lower'])} | {_fmt_pct(_epic_dca_range['upper'])} |
        | Morse Fall Scale | {_fmt_pct(_morse_dca_range['lower'])} | {_fmt_pct(_morse_dca_range['upper'])} |

        **Figure 3 saved** to `outputs/figures/figure3_dca.pdf` and `.png`.
        """
    )
    return


@app.cell
def _(Path, df_dca, mo, pl):
    # A1: Export DCA net benefit table at key thresholds (matches NB06 output)
    _key_thresh = [0.01, 0.02, 0.03, 0.05]
    _rows = []
    for _t in _key_thresh:
        _epic_row = df_dca[(df_dca["model"] == "epic_prob") & (abs(df_dca["threshold"] - _t) < 0.0005)]
        _morse_row = df_dca[(df_dca["model"] == "morse_prob") & (abs(df_dca["threshold"] - _t) < 0.0005)]
        _all_row = df_dca[(df_dca["model"] == "all") & (abs(df_dca["threshold"] - _t) < 0.0005)]
        _nb_epic = float(_epic_row["net_benefit"].iloc[0]) if len(_epic_row) > 0 else float("nan")
        _nb_morse = float(_morse_row["net_benefit"].iloc[0]) if len(_morse_row) > 0 else float("nan")
        _nb_all = float(_all_row["net_benefit"].iloc[0]) if len(_all_row) > 0 else float("nan")
        _rows.append({
            "Threshold (%)": f"{_t * 100:.0f}%",
            "Epic PMFRS": round(_nb_epic, 5),
            "Morse Fall Scale": round(_nb_morse, 5),
            "Treat all": round(_nb_all, 5),
            "Treat none": 0.0,
        })
    _dca_tbl = pl.DataFrame(_rows)
    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _dca_tbl.write_csv(_out_dir / "etable7_dca_net_benefit.csv")
    mo.vstack([
        mo.md("### eTable 7. DCA Net Benefit at Selected Threshold Probabilities"),
        mo.ui.table(_dca_tbl),
        mo.md("**Saved**: `outputs/tables/etable7_dca_net_benefit.csv`"),
    ])
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **What the decision curve means**

            Both tools provide clinical benefit across a range of decision thresholds. This means
            using either tool to guide fall prevention is better than the two extreme strategies of
            treating every patient as high risk (maximum alerts, maximum intervention costs) or
            treating no one (zero prevention effort). The curves show that across the clinically
            relevant threshold range (0–10%), both models add value — neither is clearly dominant.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            """
            ## 8. Threshold Analysis \u2014 eFigures 3 & 5

            There are many ways to pick the 'cutoff score' that separates high-risk from
            low-risk patients. Each method involves a different trade-off: catch more fallers
            (higher sensitivity) but flag more non-fallers (higher alert burden), or miss some
            fallers but reduce unnecessary alerts.
            """
        ),
        mo.accordion({
            "Statistical details: threshold methods": mo.md(
                """
                **7 methods**: Youden (max J=sens+spec-1), closest-to-(0,1), fixed sensitivity
                60%/80%, value-optimizing NMB (Parsons JAMIA 2023), DCA-derived, standard clinical
                cutoffs. Epic thresholds (35/70/50) designed for continuous monitoring \u2014 at
                admission, 97% score <35. Flag rate = alert fatigue proxy.
                """
            ),
        }),
    ])
    return


@app.cell
def _(
    COLORS,
    EPIC_2TIER_HIGH,
    EPIC_3TIER_HIGH,
    EPIC_3TIER_MEDIUM,
    JAMA_STYLE,
    MFS_HIGH,
    MFS_MODERATE,
    classification_metrics_at_threshold,
    closest_topleft_threshold,
    epic_prob,
    epic_scores,
    fixed_sensitivity_threshold,
    morse_prob,
    morse_scores,
    mpatches,
    np,
    plt,
    save_figure,
    value_optimizing_threshold,
    y_true,
    youden_threshold,
):
    from sklearn.metrics import roc_curve as _roc_curve3

    _fpr_epic, _tpr_epic, _ = _roc_curve3(y_true, epic_scores)
    _fpr_morse, _tpr_morse, _ = _roc_curve3(y_true, morse_scores)

    # Compute all threshold operating points for overlay
    def _compute_thresholds(score, prob):
        _results = []
        _t = youden_threshold(y_true, score)
        _results.append(("Youden", _t, classification_metrics_at_threshold(y_true, score, _t)))
        _t = closest_topleft_threshold(y_true, score)
        _results.append(("Closest (0,1)", _t, classification_metrics_at_threshold(y_true, score, _t)))
        _t = fixed_sensitivity_threshold(y_true, score, 0.60)
        _results.append(("Sens \u2265 60%", _t, classification_metrics_at_threshold(y_true, score, _t)))
        _t = fixed_sensitivity_threshold(y_true, score, 0.80)
        _results.append(("Sens \u2265 80%", _t, classification_metrics_at_threshold(y_true, score, _t)))
        _t_nmb, _, _ = value_optimizing_threshold(y_true, prob)
        _results.append(("Value NMB", _t_nmb, classification_metrics_at_threshold(y_true, prob, _t_nmb)))
        return _results

    _epic_thresh = _compute_thresholds(epic_scores, epic_prob)
    _morse_thresh = _compute_thresholds(morse_scores, morse_prob)

    # Add standard cutoffs
    for _t, _lbl in [(float(EPIC_3TIER_MEDIUM), "Epic \u2265 35"), (float(EPIC_3TIER_HIGH), "Epic \u2265 70"), (float(EPIC_2TIER_HIGH), "Epic \u2265 50")]:
        _epic_thresh.append((_lbl, _t, classification_metrics_at_threshold(y_true, epic_scores, _t)))
    for _t, _lbl in [(float(MFS_MODERATE), "MFS \u2265 25"), (float(MFS_HIGH), "MFS \u2265 45")]:
        _morse_thresh.append((_lbl, _t, classification_metrics_at_threshold(y_true, morse_scores, _t)))

    _METHOD_STYLES = {
        "Youden": {"marker": "o", "color": "#1B7837"},
        "Closest (0,1)": {"marker": "s", "color": "#762A83"},
        "Sens \u2265 60%": {"marker": "^", "color": "#E08214"},
        "Sens \u2265 80%": {"marker": "v", "color": "#FDB863"},
        "Value NMB": {"marker": "D", "color": "#4393C3"},
        "MFS \u2265 25": {"marker": "P", "color": "#D6604D"},
        "MFS \u2265 45": {"marker": "*", "color": "#8C510A"},
        "Epic \u2265 35": {"marker": "h", "color": "#2166AC"},
        "Epic \u2265 70": {"marker": "H", "color": "#053061"},
        "Epic \u2265 50": {"marker": "X", "color": "#4393C3"},
    }

    with plt.rc_context(JAMA_STYLE):
        _fig_ef3, (_ax_e, _ax_m) = plt.subplots(1, 2, figsize=(7.0, 4.0))
        for _ax, _color, _fpr, _tpr, _thresholds in [
            (_ax_e, COLORS["epic"], _fpr_epic, _tpr_epic, _epic_thresh),
            (_ax_m, COLORS["morse"], _fpr_morse, _tpr_morse, _morse_thresh),
        ]:
            _ax.plot(_fpr, _tpr, color=_color, linewidth=1.25, zorder=2)
            _ax.plot([0, 1], [0, 1], color="#AAAAAA", linewidth=0.5, linestyle="--", zorder=1)
            for _method, _thresh, _mets in _thresholds:
                _sens = _mets["sensitivity"]
                _fpr_pt = 1 - _mets["specificity"]
                _style = _METHOD_STYLES.get(_method, {"marker": "o", "color": "#888888"})
                _ax.plot(_fpr_pt, _sens, marker=_style["marker"], color=_style["color"],
                         markersize=_style.get("ms", 7), markeredgecolor="white", markeredgewidth=0.5,
                         linestyle="none", zorder=5)
            _ax.set_xlim(-0.02, 1.02)
            _ax.set_ylim(-0.02, 1.02)
            _ax.set_xlabel("1 - Specificity", fontsize=9)
            _ax.set_ylabel("Sensitivity", fontsize=9)
            _ax.tick_params(labelsize=8)

        _ax_e.set_title("Epic PMFRS", fontsize=10, fontweight="bold")
        _ax_m.set_title("Morse Fall Scale", fontsize=10, fontweight="bold")

        _handles = [mpatches.Patch(color=s["color"], label=m) for m, s in _METHOD_STYLES.items()]
        _fig_ef3.legend(handles=_handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
                        ncol=4, fontsize=8, frameon=False, columnspacing=0.8)
        _fig_ef3.subplots_adjust(wspace=0.35, bottom=0.28)

        _fig_ef3.text(
            0.5, -0.14,
            "eFigure 3. Threshold method operating points on ROC curves:\n"
            "Epic PMFRS vs Morse Fall Scale",
            ha="center", va="top", fontsize=10, fontweight="bold",
        )

    save_figure(_fig_ef3, "efigure3_threshold_overlay", bbox_inches=None, pad_inches=0.15)

    # eFigure 5: Score distributions
    def _flag_rate(scores, threshold):
        return float(np.sum(scores >= threshold) / len(scores) * 100)

    with plt.rc_context(JAMA_STYLE):
        _fig_dist, (_ax_ep, _ax_mo) = plt.subplots(1, 2, figsize=(7.0, 4.0))

        _ax_ep.hist(epic_scores[y_true == 0], bins=100, range=(0, 100), color=COLORS["ci_fill"], edgecolor="none", alpha=0.8, label="Non-fallers")
        _ax_ep.hist(epic_scores[y_true == 1], bins=100, range=(0, 100), color=COLORS["epic"], edgecolor="none", alpha=0.7, label="Fallers")
        for _t, _lbl in [(float(EPIC_3TIER_MEDIUM), "\u226535\n(3-tier med)"), (float(EPIC_2TIER_HIGH), "\u226550\n(2-tier)"), (float(EPIC_3TIER_HIGH), "\u226570\n(3-tier high)")]:
            _fr = _flag_rate(epic_scores, _t)
            _ax_ep.axvline(_t, color="#333333", linewidth=0.8, linestyle="--", zorder=3)
            _ax_ep.text(_t + 1, _ax_ep.get_ylim()[1] * 0.85, f"{_lbl}\n{_fr:.1f}% flagged", fontsize=8, va="top", ha="left", color="#333333")
        _ax_ep.set_xlabel("Epic PMFRS score at admission", fontsize=9)
        _ax_ep.set_ylabel("Encounters", fontsize=9)
        _ax_ep.set_title("Epic PMFRS", fontsize=10, fontweight="bold")
        _ax_ep.set_xlim(0, 100)
        _ax_ep.tick_params(labelsize=8)
        _ax_ep.text(-0.14, 1.06, "A", transform=_ax_ep.transAxes, fontsize=10, fontweight="bold", va="top")

        _morse_bins = np.arange(-0.5, 130, 5)
        _ax_mo.hist(morse_scores[y_true == 0], bins=_morse_bins, color=COLORS["ci_fill"], edgecolor="none", alpha=0.8, label="Non-fallers")
        _ax_mo.hist(morse_scores[y_true == 1], bins=_morse_bins, color=COLORS["morse"], edgecolor="none", alpha=0.7, label="Fallers")
        for _t, _lbl in [(float(MFS_MODERATE), "\u226525\n(moderate)"), (float(MFS_HIGH), "\u226545\n(high risk)")]:
            _fr = _flag_rate(morse_scores, _t)
            _ax_mo.axvline(_t, color="#333333", linewidth=0.8, linestyle="--", zorder=3)
            _ax_mo.text(_t + 2, _ax_mo.get_ylim()[1] * 0.85, f"{_lbl}\n{_fr:.1f}% flagged", fontsize=8, va="top", ha="left")
        _ax_mo.set_xlabel("Morse Fall Scale score at admission", fontsize=9)
        _ax_mo.set_ylabel("Encounters", fontsize=9)
        _ax_mo.set_title("Morse Fall Scale", fontsize=10, fontweight="bold")
        _ax_mo.set_xlim(-2, 130)
        _ax_mo.tick_params(labelsize=8)
        _ax_mo.text(-0.14, 1.06, "B", transform=_ax_mo.transAxes, fontsize=10, fontweight="bold", va="top")

        _h, _l = _ax_ep.get_legend_handles_labels()
        _fig_dist.legend(_h, _l, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=False)
        _fig_dist.subplots_adjust(wspace=0.32, bottom=0.26)

        _fig_dist.text(
            0.5, -0.15,
            "eFigure 5. Score distributions at admission with standard threshold annotations",
            ha="center", va="top", fontsize=10, fontweight="bold",
        )

    save_figure(_fig_dist, "efigure5_score_distributions", bbox_inches=None, pad_inches=0.15)
    return


@app.cell
def _(
    EPIC_2TIER_HIGH,
    EPIC_3TIER_HIGH,
    EPIC_3TIER_MEDIUM,
    MFS_HIGH,
    MFS_MODERATE,
    Path,
    epic_metrics_by_threshold: list[dict],
    epic_scores,
    mo,
    morse_metrics_by_threshold: list[dict],
    morse_scores,
    np,
    pl,
):
    # A1: Export etable6_threshold_methods.csv (matches NB07 output)
    _thresh_rows = []
    for _m in epic_metrics_by_threshold:
        _thresh_rows.append({
            "Model": "Epic PMFRS", "Method": _m["label"],
            "Threshold": round(_m["threshold"], 2),
            "Sensitivity, %": round(_m["sensitivity"] * 100, 1),
            "Specificity, %": round(_m["specificity"] * 100, 1),
            "PPV, %": round(_m["ppv"] * 100, 1),
            "NPV, %": round(_m["npv"] * 100, 1),
            "Flag rate, %": round(_m["flag_rate"], 1),
            "NNE": round(_m["nne"], 1) if _m["nne"] != float("inf") else None,
        })
    for _m in morse_metrics_by_threshold:
        _thresh_rows.append({
            "Model": "Morse Fall Scale", "Method": _m["label"],
            "Threshold": round(_m["threshold"], 2),
            "Sensitivity, %": round(_m["sensitivity"] * 100, 1),
            "Specificity, %": round(_m["specificity"] * 100, 1),
            "PPV, %": round(_m["ppv"] * 100, 1),
            "NPV, %": round(_m["npv"] * 100, 1),
            "Flag rate, %": round(_m["flag_rate"], 1),
            "NNE": round(_m["nne"], 1) if _m["nne"] != float("inf") else None,
        })
    _thresh_df = pl.DataFrame(_thresh_rows)

    # A1: Export etable6_flag_rate_summary.csv (matches NB07 output)
    def _flag_rate_pct(scores, threshold):
        return round(float(np.sum(scores >= threshold) / len(scores) * 100), 1)

    _flag_rows = [
        {"Model": "Epic PMFRS", "Cutoff": "\u226535 (3-tier medium)", "Flag rate, %": _flag_rate_pct(epic_scores, float(EPIC_3TIER_MEDIUM))},
        {"Model": "Epic PMFRS", "Cutoff": "\u226550 (2-tier high)", "Flag rate, %": _flag_rate_pct(epic_scores, float(EPIC_2TIER_HIGH))},
        {"Model": "Epic PMFRS", "Cutoff": "\u226570 (3-tier high)", "Flag rate, %": _flag_rate_pct(epic_scores, float(EPIC_3TIER_HIGH))},
        {"Model": "Morse Fall Scale", "Cutoff": "\u226525 (moderate)", "Flag rate, %": _flag_rate_pct(morse_scores, float(MFS_MODERATE))},
        {"Model": "Morse Fall Scale", "Cutoff": "\u226545 (high risk)", "Flag rate, %": _flag_rate_pct(morse_scores, float(MFS_HIGH))},
    ]
    _flag_df = pl.DataFrame(_flag_rows)

    _out_dir = Path("outputs/tables")
    _out_dir.mkdir(parents=True, exist_ok=True)
    _thresh_df.write_csv(_out_dir / "etable6_threshold_methods.csv")
    _flag_df.write_csv(_out_dir / "etable6_flag_rate_summary.csv")

    mo.vstack([
        mo.md("### eTable 6. Optimal Cutpoints by Threshold Selection Method"),
        mo.ui.table(_thresh_df),
        mo.md("**Saved**: `outputs/tables/etable6_threshold_methods.csv`"),
        mo.md("### Flag Rate Summary"),
        mo.ui.table(_flag_df),
        mo.md("**Saved**: `outputs/tables/etable6_flag_rate_summary.csv`"),
        mo.md("**eFigures 3 and 5 saved** to `outputs/figures/`."),
    ])
    return


@app.cell
def _(Path, mo):
    # B6: Side-by-side threshold and score distribution figure display
    _efig3 = Path("outputs/figures/efigure3_threshold_overlay.png")
    _efig5 = Path("outputs/figures/efigure5_score_distributions.png")
    mo.hstack([
        mo.vstack([mo.md("**eFigure 3.** Threshold overlay"), mo.image(src=_efig3)]) if _efig3.exists() else mo.md("*eFigure 3 not yet generated*"),
        mo.vstack([mo.md("**eFigure 5.** Score distributions"), mo.image(src=_efig5)]) if _efig5.exists() else mo.md("*eFigure 5 not yet generated*"),
    ], gap=1)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            """
            ## 9. Reclassification Analysis \u2014 Table 3

            If our hospital switched from the Morse checklist to the Epic model, would patients
            end up in the *right* risk group more often? The Net Reclassification Improvement
            (NRI) measures exactly this.
            """
        ),
        mo.accordion({
            "Statistical details: NRI/IDI methodology": mo.md(
                """
                **Continuous NRI** = P(up|event) - P(down|event) + P(down|nonevent) - P(up|nonevent).
                Report event NRI and non-event NRI SEPARATELY per Pepe *Stat Med* 2015;34:110-128.

                **Categorical NRI**: At multiple thresholds (Morse Youden, Epic Youden, MFS \u226525,
                MFS \u226545, Epic \u226535, Epic \u226550). Epic \u226570 omitted (approximately 0.4% score \u226570 at
                admission \u2014 uninformative).

                **IDI** = mean probability difference for events minus mean probability difference
                for non-events. Bootstrap: 2000 stratified resamples, percentile CIs.
                """
            ),
        }),
    ])
    return


@app.cell
def _(
    EPIC_2TIER_HIGH,
    EPIC_3TIER_MEDIUM,
    MFS_HIGH,
    MFS_MODERATE,
    compute_categorical_nri,
    compute_nri_idi,
    epic_lr,
    epic_prob,
    mo,
    morse_lr,
    morse_prob,
    y_true,
    youden_threshold,
):
    mo.vstack([
        mo.md("""
        ### Compute Reclassification Metrics — NRI and IDI Point Estimates

        **What this cell does**: Calculates whether switching from Morse to Epic would move patients into more appropriate risk categories. Positive NRI means Epic categorizes more patients correctly; negative NRI means Morse does better.

        **Data used**:
        - `y_true`: Ground truth fall outcomes
        - `morse_prob`: Recalibrated Morse probabilities (reference model)
        - `epic_prob`: Recalibrated Epic probabilities (new model)

        **Key findings**:
        - **Event NRI**: Among fallers, proportion correctly moved up in risk minus proportion incorrectly moved down
        - **Non-event NRI**: Among non-fallers, proportion correctly moved down in risk minus proportion incorrectly moved up
        - **IDI**: Average improvement in risk separation between models
        - **Categorical NRI**: Reclassification at standard clinical thresholds (Morse ≥45, Epic ≥35, etc.)
        """),
        mo.accordion({
            "📊 Statistical Details": mo.md("""
            **Analysis**: Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI)

            **Continuous NRI formula**:
            ```
            NRI = (P_up|event - P_down|event) + (P_down|nonevent - P_up|nonevent)
            ```

            **Categorical NRI method**: Uses threshold to create binary categories (low vs high risk) for each model, then counts reclassifications

            **Parameters**:
            - Reference model: Morse Fall Scale
            - New model: Epic PMFRS
            - Thresholds tested:
              - Morse Youden-optimized
              - Epic Youden-optimized
              - Morse ≥25 (moderate risk)
              - Morse ≥45 (high risk at Rush)
              - Epic ≥35 (3-tier low/high cutoff)
              - Epic ≥50 (2-tier low/high cutoff)

            **Output variables**:
            - `nri_point_estimates`: Continuous NRI, event NRI, non-event NRI, categorical NRI, IDI
            - `nri_cat_estimates`: Categorical NRI at each standard threshold
            - `nri_thresholds`: Probability thresholds corresponding to each cutoff

            **Critical note**: Epic ≥70 threshold is **excluded** from categorical NRI because only 0.4% of encounters score ≥70 at admission — insufficient events for meaningful reclassification.

            **Assumptions**:
            - Higher risk category is correct for fallers; lower risk category is correct for non-fallers
            - Combined NRI can be misleading — event and non-event NRI should be reported separately (Pepe 2015)
            - Logistic recalibration produces probabilities suitable for probability-based NRI/IDI

            **References**:
            - Pencina MJ et al. Evaluating the added predictive ability of a new marker. *Stat Med*. 2008;27(2):157-172.
            - Pepe MS et al. Limitations of the net reclassification improvement. *Stat Med*. 2015;34(1):110-128.
            """)
        }),
    ])

    _morse_youden = youden_threshold(y_true, morse_prob)
    _epic_youden = youden_threshold(y_true, epic_prob)

    _morse_prob_at_25 = float(morse_lr.predict_proba([[float(MFS_MODERATE)]])[0, 1])
    _morse_prob_at_45 = float(morse_lr.predict_proba([[float(MFS_HIGH)]])[0, 1])
    _epic_prob_at_35 = float(epic_lr.predict_proba([[float(EPIC_3TIER_MEDIUM)]])[0, 1])
    _epic_prob_at_50 = float(epic_lr.predict_proba([[float(EPIC_2TIER_HIGH)]])[0, 1])

    nri_point_estimates = compute_nri_idi(y_true=y_true, prob_ref=morse_prob, prob_new=epic_prob, threshold=_morse_youden)

    _pe_cat_epic_youden = compute_categorical_nri(y_true, morse_prob, epic_prob, _epic_youden)
    _pe_cat_mfs25 = compute_categorical_nri(y_true, morse_prob, epic_prob, _morse_prob_at_25)
    _pe_cat_mfs45 = compute_categorical_nri(y_true, morse_prob, epic_prob, _morse_prob_at_45)
    _pe_cat_epic35 = compute_categorical_nri(y_true, morse_prob, epic_prob, _epic_prob_at_35)
    _pe_cat_epic50 = compute_categorical_nri(y_true, morse_prob, epic_prob, _epic_prob_at_50)

    nri_cat_estimates = {
        "morse_youden": nri_point_estimates["nri_categorical"],
        "epic_youden": _pe_cat_epic_youden,
        "mfs25": _pe_cat_mfs25,
        "mfs45": _pe_cat_mfs45,
        "epic35": _pe_cat_epic35,
        "epic50": _pe_cat_epic50,
    }
    nri_thresholds = {
        "morse_youden": _morse_youden,
        "epic_youden": _epic_youden,
        "morse_prob_at_25": _morse_prob_at_25,
        "morse_prob_at_45": _morse_prob_at_45,
        "epic_prob_at_35": _epic_prob_at_35,
        "epic_prob_at_50": _epic_prob_at_50,
    }
    return nri_cat_estimates, nri_point_estimates, nri_thresholds


@app.cell
def _(
    ALPHA,
    N_BOOTSTRAP,
    RANDOM_SEED,
    compute_categorical_nri,
    compute_nri_idi,
    epic_prob,
    mo,
    morse_prob,
    np,
    nri_thresholds,
    y_true,
):
    _rng = np.random.RandomState(RANDOM_SEED)
    _events_idx = np.where(y_true == 1)[0]
    _nonevents_idx = np.where(y_true == 0)[0]

    _boot_keys = ["nri_continuous", "nri_events", "nri_nonevents", "nri_categorical", "idi", "idi_events", "idi_nonevents"]
    _boot_results: dict[str, list[float]] = {k: [] for k in _boot_keys}
    _boot_cat: dict[str, list[float]] = {k: [] for k in ["epic_youden", "mfs25", "mfs45", "epic35", "epic50"]}

    for _ in mo.status.progress_bar(range(N_BOOTSTRAP), title="Bootstrap", subtitle="NRI/IDI confidence intervals"):
        _be = _rng.choice(_events_idx, size=len(_events_idx), replace=True)
        _bn = _rng.choice(_nonevents_idx, size=len(_nonevents_idx), replace=True)
        _idx = np.concatenate([_be, _bn])
        _y_b = y_true[_idx]
        _ref_b = morse_prob[_idx]
        _new_b = epic_prob[_idx]

        _est = compute_nri_idi(_y_b, _ref_b, _new_b, threshold=nri_thresholds["morse_youden"])
        for _k in _boot_keys:
            _v = _est[_k]
            if _v is not None:
                _boot_results[_k].append(float(_v))

        _boot_cat["epic_youden"].append(compute_categorical_nri(_y_b, _ref_b, _new_b, nri_thresholds["epic_youden"]))
        _boot_cat["mfs25"].append(compute_categorical_nri(_y_b, _ref_b, _new_b, nri_thresholds["morse_prob_at_25"]))
        _boot_cat["mfs45"].append(compute_categorical_nri(_y_b, _ref_b, _new_b, nri_thresholds["morse_prob_at_45"]))
        _boot_cat["epic35"].append(compute_categorical_nri(_y_b, _ref_b, _new_b, nri_thresholds["epic_prob_at_35"]))
        _boot_cat["epic50"].append(compute_categorical_nri(_y_b, _ref_b, _new_b, nri_thresholds["epic_prob_at_50"]))

    _lo_pct = 100 * ALPHA / 2
    _hi_pct = 100 * (1 - ALPHA / 2)
    nri_boot_ci: dict[str, dict[str, float]] = {}
    for _k, _vals in _boot_results.items():
        _arr = np.array(_vals)
        nri_boot_ci[_k] = {"ci_lower": float(np.percentile(_arr, _lo_pct)), "ci_upper": float(np.percentile(_arr, _hi_pct))}
    for _k, _vals in _boot_cat.items():
        _arr = np.array(_vals)
        nri_boot_ci[f"nri_categorical_{_k}"] = {"ci_lower": float(np.percentile(_arr, _lo_pct)), "ci_upper": float(np.percentile(_arr, _hi_pct))}
    return (nri_boot_ci,)


@app.cell
def _(
    Path,
    mo,
    nri_boot_ci: dict[str, dict[str, float]],
    nri_cat_estimates,
    nri_point_estimates,
    pl,
):
    _sections = [
        ("Continuous NRI", nri_point_estimates["nri_events"], "nri_events", "  Events (upward \u2212 downward)"),
        ("Continuous NRI", nri_point_estimates["nri_nonevents"], "nri_nonevents", "  Non-events (downward \u2212 upward)"),
        ("Continuous NRI", nri_point_estimates["nri_continuous"], "nri_continuous", "  Total continuous NRI"),
        ("Categorical NRI", nri_cat_estimates["morse_youden"], "nri_categorical", "  Morse Youden threshold"),
        ("Categorical NRI", nri_cat_estimates["epic_youden"], "nri_categorical_epic_youden", "  Epic Youden threshold"),
        ("Categorical NRI", nri_cat_estimates["mfs25"], "nri_categorical_mfs25", "  MFS \u226525 threshold"),
        ("Categorical NRI", nri_cat_estimates["mfs45"], "nri_categorical_mfs45", "  MFS \u226545 threshold"),
        ("Categorical NRI", nri_cat_estimates["epic35"], "nri_categorical_epic35", "  Epic \u226535 threshold"),
        ("Categorical NRI", nri_cat_estimates["epic50"], "nri_categorical_epic50", "  Epic \u226550 threshold"),
        ("IDI", nri_point_estimates["idi_events"], "idi_events", "  Events"),
        ("IDI", nri_point_estimates["idi_nonevents"], "idi_nonevents", "  Non-events"),
        ("IDI", nri_point_estimates["idi"], "idi", "  Total IDI"),
    ]

    _rows = []
    for _section, _pe_val, _boot_key, _label in _sections:
        _ci = nri_boot_ci[_boot_key]
        _est_str = f"{_pe_val:.3f}" if _pe_val is not None else "\u2014"
        _rows.append({"Section": _section, "Metric": _label, "Estimate": _est_str,
                      "95% Bootstrap CI": f"{_ci['ci_lower']:.3f} to {_ci['ci_upper']:.3f}"})

    table3 = pl.DataFrame(_rows)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    table3.write_csv(Path("outputs/tables/table3.csv"))

    mo.vstack([
        mo.md("### Table 3. Reclassification Analysis: Epic PMFRS vs Morse Fall Scale"),
        mo.ui.table(table3),
        mo.md("**Saved**: `outputs/tables/table3.csv`"),
        mo.accordion({
            "🎯 Key Findings: Reclassification Analysis": mo.md("""
                **Continuous NRI** (probability-based reclassification):
                - Event NRI (fallers): [Value] — [interpretation from table: positive = Epic better for fallers]
                - Non-event NRI (non-fallers): [Value] — [interpretation: positive = Epic better for non-fallers]
                - Total continuous NRI: [Value]

                **IDI** (Integrated Discrimination Improvement) — mean difference in predicted probabilities:
                - Total IDI: [Value] — [interpretation: magnitude of risk separation improvement]

                **Categorical NRI** (threshold-based reclassification):
                - Using Youden-optimized thresholds: [Value]
                - Using standard Morse ≥25 threshold: [Value]
                - Using Morse ≥45 threshold: [Value]
                - Using Epic ≥35 (3-tier low/high cutoff): [Value]
                - Using Epic ≥50 (2-tier low/high cutoff): [Value]

                **Interpretation**:
                - NRI measures *reclassification*: does Epic move patients into more appropriate risk tiers compared to Morse?
                - Report event and non-event NRI separately (combined NRI can be misleading per Pepe 2015)
                - A positive event NRI means Epic correctly identifies more fallers as high-risk than Morse
                - A positive non-event NRI means Epic correctly identifies more non-fallers as low-risk than Morse
                - Epic might excel for fallers but underperform for non-fallers (or vice versa) — both matter clinically
                - IDI measures average risk separation improvement; higher values indicate better stratification

                **Clinical context**: Replacing Morse with Epic would reclassify approximately X% of patients into different risk categories. Whether these reclassifications improve outcomes depends on intervention effectiveness and cost.

                **Note**: Epic ≥70 threshold excluded from categorical NRI (insufficient events at admission — only 0.4% of encounters reach this score).
            """),
        }),
    ])
    return


@app.cell
def _(mo, nri_point_estimates):
    _event_nri = nri_point_estimates["nri_events"]
    _nonevent_nri = nri_point_estimates["nri_nonevents"]

    def _nri_direction(val):
        if val is None:
            return "could not be determined"
        if val > 0:
            return f"is positive ({val:.3f}), meaning Epic correctly reclassifies more patients than Morse"
        if val < 0:
            return f"is negative ({val:.3f}), meaning Morse correctly classifies more patients than Epic"
        return "is zero, meaning neither tool reclassifies better"

    mo.callout(
        mo.md(
            f"""
            **What the reclassification results mean**

            The reclassification analysis measures whether switching from the Morse checklist to the
            Epic model would move patients into better or worse risk categories.

            - **Event NRI** (fallers) {_nri_direction(_event_nri)}
            - **Non-event NRI** (non-fallers) {_nri_direction(_nonevent_nri)}

            These two components should be interpreted separately (Pepe *Stat Med* 2015): a tool can
            improve classification for fallers while worsening it for non-fallers, or vice versa. In
            practical terms, if your unit switched from Morse to Epic, these numbers tell you how many
            patients would be reclassified — and whether those reclassifications would be correct or
            incorrect.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            """
            ## 10. Fairness Audit \u2014 eTables 1–3, 8

            A prediction tool should work equally well for all patient groups. We tested whether
            the Epic model and the Morse scale perform differently for younger vs older patients,
            different racial groups, men vs women, and across hospital units. Groups with fewer
            than 20 falls are too small for reliable comparison.
            """
        ),
        mo.accordion({
            "Statistical details: fairness methods": mo.md(
                """
                **TRIPOD+AI Item 14e** mandates subgroup performance reporting. Stratified bootstrap
                (2000 reps) within each subgroup \u2014 same seed for paired comparison.

                Age bins: 18-64, 65-79, \u226580. Race: White, Black, Asian, Other/Unknown.
                Min 20 events/subgroup (below \u2192 AUROC unreliable, report as "\u2014").

                Causal inference not possible \u2014 apparent disparities may reflect case mix,
                documentation practices, or true score properties.
                """
            ),
        }),
    ])
    return


@app.cell
def _(
    AGE_BINS,
    AGE_LABELS,
    MIN_SUBGROUP_EVENTS,
    N_BOOTSTRAP,
    Path,
    RANDOM_SEED,
    df_analytic,
    pl,
    stratified_bootstrap,
):
    from sklearn.metrics import roc_auc_score as _roc_auc_score

    def _compute_subgroup_row(sub_df, group_label, group_value):
        n = sub_df.height
        n_falls = int(sub_df["fall_flag"].sum())
        event_rate = n_falls / n if n > 0 else 0.0
        row = {"Subgroup": group_label, "Category": group_value, "N encounters": n,
               "N falls": n_falls, "Event rate, %": round(event_rate * 100, 1)}
        if n_falls < MIN_SUBGROUP_EVENTS or n - n_falls < MIN_SUBGROUP_EVENTS:
            row.update({"Epic AUROC": "\u2014", "Epic 95% CI": "\u2014", "Morse AUROC": "\u2014", "Morse 95% CI": "\u2014", "Note": f"<{MIN_SUBGROUP_EVENTS} events"})
            return row
        _y = sub_df["fall_flag"].to_numpy()
        _es = sub_df["epic_score_admission"].to_numpy()
        _ms = sub_df["morse_score_admission"].to_numpy()
        _epic_auc = float(_roc_auc_score(_y, _es))
        _morse_auc = float(_roc_auc_score(_y, _ms))
        _boot = stratified_bootstrap(_y, _es, pred_b=_ms, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
        row.update({
            "Epic AUROC": f"{_epic_auc:.3f}",
            "Epic 95% CI": f"({_boot['auc_a']['ci_lower']:.3f}\u2013{_boot['auc_a']['ci_upper']:.3f})",
            "Morse AUROC": f"{_morse_auc:.3f}",
            "Morse 95% CI": f"({_boot['auc_b']['ci_lower']:.3f}\u2013{_boot['auc_b']['ci_upper']:.3f})",
            "Note": "",
        })
        return row

    # Add age_group
    _df = df_analytic.with_columns(
        pl.when(pl.col("age") < AGE_BINS[1]).then(pl.lit(AGE_LABELS[0]))
        .when(pl.col("age") < AGE_BINS[2]).then(pl.lit(AGE_LABELS[1]))
        .otherwise(pl.lit(AGE_LABELS[2])).alias("age_group")
    )

    # eTable 1: by age
    _rows_age = [_compute_subgroup_row(_df.filter(pl.col("age_group") == _lbl), "Age group", _lbl) for _lbl in AGE_LABELS]
    etable1_df = pl.DataFrame(_rows_age)

    # eTable 2: by race
    _race_cats = df_analytic.group_by("race").agg(pl.len().alias("n")).sort("n", descending=True)["race"].to_list()
    _rows_race = [_compute_subgroup_row(df_analytic.filter(pl.col("race") == _c), "Race", str(_c)) for _c in _race_cats if _c is not None]
    etable2_df = pl.DataFrame(_rows_race)

    # eTable 3: by unit (top 10)
    _top10 = df_analytic.group_by("admitting_department").agg(pl.len().alias("n")).sort("n", descending=True).head(10)["admitting_department"].to_list()
    _rows_unit = [_compute_subgroup_row(df_analytic.filter(pl.col("admitting_department") == _d), "Department", str(_d)) for _d in _top10 if _d is not None]
    etable3_df = pl.DataFrame(_rows_unit)

    # eTable 8 (gender)
    _gender_cats = df_analytic.group_by("gender").agg(pl.len().alias("n")).sort("n", descending=True)["gender"].to_list()
    _rows_gender = [_compute_subgroup_row(df_analytic.filter(pl.col("gender") == _g), "Gender", str(_g)) for _g in _gender_cats if _g is not None]
    etable8_gender_df = pl.DataFrame(_rows_gender)

    # Export
    _out = Path("outputs/tables")
    _out.mkdir(parents=True, exist_ok=True)
    etable1_df.write_csv(_out / "etable1_age.csv")
    etable2_df.write_csv(_out / "etable2_race.csv")
    etable3_df.write_csv(_out / "etable3_unit.csv")
    etable8_gender_df.write_csv(_out / "etable8_gender.csv")
    return etable1_df, etable2_df, etable3_df, etable8_gender_df


@app.cell
def _(etable1_df, etable2_df, etable3_df, etable8_gender_df, mo):
    mo.ui.tabs({
        "Age": mo.vstack([mo.md("**eTable 1.** Model Performance by Age Group"), mo.ui.table(etable1_df)]),
        "Race": mo.vstack([mo.md("**eTable 2.** Model Performance by Race/Ethnicity"), mo.ui.table(etable2_df)]),
        "Unit": mo.vstack([mo.md("**eTable 3.** Model Performance by Department (Top 10)"), mo.ui.table(etable3_df)]),
        "Gender": mo.vstack([mo.md("**eTable 8.** Model Performance by Gender"), mo.ui.table(etable8_gender_df)]),
    })
    return


@app.cell
def _(mo):
    mo.md("""
    **eTables 1-3, 8 saved** to `outputs/tables/etable1_age.csv`, `etable2_race.csv`, `etable3_unit.csv`, `etable8_gender.csv`.
    """)
    return


@app.cell
def _(Path, df_analytic, mo, morse_auc, morse_scores, np, pl, y_true):
    mo.vstack([
        mo.md("""
        ### Generate Literature Benchmarking Table — eTable 9

        **What this cell does**: Compares our Morse Fall Scale validation results against published validation studies. This provides context for interpreting whether our findings are consistent with external evidence or represent unusual performance.

        **Data used**:
        - `df_analytic`: Our cohort for computing "Present study" row
        - `y_true` and `morse_scores`: For computing our sensitivity/specificity at MFS ≥45
        - Published literature: 17 prior MFS validation studies extracted from PubMed

        **Key findings**:
        - **Our MFS AUROC (0.635)** is consistent with methodologically comparable prospective studies (Ji 2023: 0.63, Lohse 2021: 0.64)
        - **Higher published AUROCs (0.76–0.86)** typically reflect case-control designs, which artificially inflate discrimination
        - **Our sensitivity at MFS ≥45 (83.8%)** falls within published range (31–88%) and matches Epic's validation sites (80.5–87.6%)
        - **Low specificity (34.6%)** reflects our cohort's score distribution — 65.7% score ≥45 at admission
        """),
        mo.accordion({
            "📊 Statistical Details": mo.md("""
            **Analysis**: Systematic literature comparison with design-adjusted interpretation

            **Method**:
            - Extracted 17 prior MFS validation studies from PubMed
            - Computed "Present study" row dynamically from our data
            - Stratified studies by design: prospective cohort vs case-control

            **Parameters**:
            - Standard cutoff for comparison: MFS ≥45 (high risk at Rush)
            - Sensitivity calculation: TP / (TP + FN)
            - Specificity calculation: TN / (TN + FP)

            **Output**: `etable9_df` — table with 18 rows (17 literature + present study)

            **Key comparisons**:
            - **Design bias**: Case-control studies inflate Youden Index by +0.22 on average (Haines 2007)
            - **Score timing**: Studies using max-before-fall scores report higher AUROC than admission-time screening
            - **Prevalence dependence**: Very low prevalence (0.10% in Jiang 2025) may contribute to higher AUROC
            - **Peer-reviewed vs manufacturer data**: Epic PMFRS has no peer-reviewed publications; validation data from unpublished model brief

            **Interpretation framework**:
            - **Concordant studies**: Prospective cohort designs with admission-time screening
            - **Expected higher performance**: Case-control, max-score timing
            - **Expected lower PPV**: All studies at 1–2% prevalence

            **References for design bias**:
            - Haines TP et al. Design-related bias in hospital fall risk screening tool predictive accuracy evaluations. *J Gerontol A Biol Sci Med Sci*. 2007;62(6):664-72. PMID: 17595425

            **TRIPOD+AI compliance**: This cell supports Item 15 (model performance comparison with related studies).
            """)
        }),
    ])

    # Compute "Present study" values dynamically
    _n_enc = df_analytic.height
    _n_falls = int(df_analytic["fall_flag"].sum())
    _prev_pct = f"{_n_falls / _n_enc * 100:.2f}"
    _n_fmt = f"{_n_enc:,}"
    # Sensitivity and specificity at MFS >= 45
    _pred_pos = morse_scores >= 45
    _tp = int(np.sum(_pred_pos & (y_true == 1)))
    _fn = int(np.sum(~_pred_pos & (y_true == 1)))
    _fp = int(np.sum(_pred_pos & (y_true == 0)))
    _tn = int(np.sum(~_pred_pos & (y_true == 0)))
    _sens = f"{_tp / (_tp + _fn) * 100:.1f}" if (_tp + _fn) > 0 else "NR"
    _spec = f"{_tn / (_tn + _fp) * 100:.1f}" if (_tn + _fp) > 0 else "NR"

    _benchmark_rows = [
        {
            "Study": "Ji 2023",
            "Setting": "Korean tertiary (Asan Medical Center)",
            "N": "2,028",
            "Fall prevalence, %": "1.23",
            "Study design": "Prospective; CFS vs MFS comparison",
            "Score timing": "Admission",
            "MFS AUROC (95% CI)": "0.63 (NR)",
            "Sensitivity, %": "60",
            "Specificity, %": "68",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u226545",
            "PMID": "37305899",
        },
        {
            "Study": "Shim 2022",
            "Setting": "Korean tertiary (Samsung Medical Center)",
            "N": "191,778",
            "Fall prevalence, %": "0.14",
            "Study design": "Retrospective cohort",
            "Score timing": "Multiple (EMR-integrated, continuous)",
            "MFS AUROC (95% CI)": "0.652 (NR)",
            "Sensitivity, %": "NR",
            "Specificity, %": "NR",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u2014",
            "PMID": "36128798",
        },
        {
            "Study": "Lohse 2021",
            "Setting": "US inpatient rehabilitation",
            "N": "2,007",
            "Fall prevalence, %": "6.5",
            "Study design": "Prospective cohort",
            "Score timing": "Admission",
            "MFS AUROC (95% CI)": "0.64 (NR)",
            "Sensitivity, %": "NR",
            "Specificity, %": "NR",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u2014",
            "PMID": "34407447",
        },
        {
            "Study": "Kim 2007",
            "Setting": "Singapore acute care",
            "N": "5,489",
            "Fall prevalence, %": "NR",
            "Study design": "Prospective cohort",
            "Score timing": "Admission",
            "MFS AUROC (95% CI)": "NR",
            "Sensitivity, %": "88",
            "Specificity, %": "48.3",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u226525",
            "PMID": "17919164",
        },
        {
            "Study": "Yazdani 2017",
            "Setting": "2 US hospitals",
            "N": "33,058",
            "Fall prevalence, %": "NR",
            "Study design": "Retrospective cohort",
            "Score timing": "Admission",
            "MFS AUROC (95% CI)": "0.782 (NR)",
            "Sensitivity, %": "NR",
            "Specificity, %": "NR",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u2014",
            "PMID": "28007719",
        },
        {
            "Study": "Baek 2014",
            "Setting": "Korean university hospital",
            "N": "845",
            "Fall prevalence, %": "NR",
            "Study design": "Case-control, max score",
            "Score timing": "Max encounter",
            "MFS AUROC (95% CI)": "0.77 (NR)",
            "Sensitivity, %": "NR",
            "Specificity, %": "NR",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u2014",
            "PMID": "24112535",
        },
        {
            "Study": "Lindberg 2020",
            "Setting": "UF Health",
            "N": "NR",
            "Fall prevalence, %": "50 (matched)",
            "Study design": "Case-control, 50% prevalence",
            "Score timing": "NR (case-control, EHR features)",
            "MFS AUROC (95% CI)": "0.86 (NR)",
            "Sensitivity, %": "NR",
            "Specificity, %": "NR",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u2014",
            "PMID": "32980667",
        },
        {
            "Study": "Kim 2011",
            "Setting": "5 Korean hospitals",
            "N": "356",
            "Fall prevalence, %": "NR",
            "Study design": "Prospective, small sample",
            "Score timing": "NR (multiple timepoints)",
            "MFS AUROC (95% CI)": "0.761 (NR)",
            "Sensitivity, %": "NR",
            "Specificity, %": "NR",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u2014",
            "PMID": "25029947",
        },
        {
            "Study": "Choi 2023",
            "Setting": "Korean stroke rehabilitation",
            "N": "1,090",
            "Fall prevalence, %": "NR",
            "Study design": "Case-control 1:6",
            "Score timing": "NR (stroke rehab, case-control 1:6)",
            "MFS AUROC (95% CI)": "0.76 (NR)",
            "Sensitivity, %": "NR",
            "Specificity, %": "NR",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u2014",
            "PMID": "37915000",
        },
        {
            "Study": "Cai 2025",
            "Setting": "Chinese geriatric (\u226580 y)",
            "N": "82",
            "Fall prevalence, %": "NR",
            "Study design": "Small, specialized",
            "Score timing": "NR (prospective, \u226580y)",
            "MFS AUROC (95% CI)": "0.813 (NR)",
            "Sensitivity, %": "NR",
            "Specificity, %": "NR",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u2014",
            "PMID": "39950405",
        },
        {
            "Study": "Kim 2022",
            "Setting": "Korean acute care",
            "N": "NR",
            "Fall prevalence, %": "NR",
            "Study design": "Case-control",
            "Score timing": "Re-measured (initial + re-assessment)",
            "MFS AUROC (95% CI)": "NR",
            "Sensitivity, %": "85.7",
            "Specificity, %": "58.8",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u226550",
            "PMID": "34964175",
        },
        {
            "Study": "Chow 2007",
            "Setting": "Hong Kong rehabilitation",
            "N": "NR",
            "Fall prevalence, %": "NR",
            "Study design": "Prospective",
            "Score timing": "Admission",
            "MFS AUROC (95% CI)": "NR",
            "Sensitivity, %": "31",
            "Specificity, %": "83",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u226545",
            "PMID": "16464453",
        },
        {
            "Study": "Mao 2024",
            "Setting": "OB/GYN wards",
            "N": "NR",
            "Fall prevalence, %": "NR",
            "Study design": "Prospective",
            "Score timing": "NR (OB/GYN, electronic records)",
            "MFS AUROC (95% CI)": "NR",
            "Sensitivity, %": "71.3",
            "Specificity, %": "80.8",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u226545",
            "PMID": "39236031",
        },
        {
            "Study": "Jiang 2025",
            "Setting": "Chinese tertiary (Mianyang Central Hospital)",
            "N": "206,846",
            "Fall prevalence, %": "0.10",
            "Study design": "Retrospective, real-world",
            "Score timing": "Within 24h of admission",
            "MFS AUROC (95% CI)": "0.825 (NR)",
            "Sensitivity, %": "NR",
            "Specificity, %": "NR",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u2014",
            "PMID": "41084899",
        },
        {
            "Study": "Epic model brief (Site 1)",
            "Setting": "US Midwest",
            "N": "206,332",
            "Fall prevalence, %": "0.38",
            "Study design": "Retrospective, production scores",
            "Score timing": "Max before fall",
            "MFS AUROC (95% CI)": "NR",
            "Sensitivity, %": "87.6",
            "Specificity, %": "NR",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u226545",
            "PMID": "\u2014",
        },
        {
            "Study": "Epic model brief (Site 2)",
            "Setting": "US Northeast",
            "N": "322,131",
            "Fall prevalence, %": "0.40",
            "Study design": "Retrospective, production scores",
            "Score timing": "Max before fall",
            "MFS AUROC (95% CI)": "NR",
            "Sensitivity, %": "80.5",
            "Specificity, %": "NR",
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u226545",
            "PMID": "\u2014",
        },
        {
            "Study": "Present study (Rush)",
            "Setting": "US academic medical center",
            "N": _n_fmt,
            "Fall prevalence, %": _prev_pct,
            "Study design": "Retrospective cohort",
            "Score timing": "Admission",
            "MFS AUROC (95% CI)": f"{morse_auc:.3f}",
            "Sensitivity, %": _sens,
            "Specificity, %": _spec,
            "PPV, %": "NR",
            "NPV, %": "NR",
            "Cutoff": "\u226545",
            "PMID": "\u2014",
        },
    ]

    etable9_df = pl.DataFrame(_benchmark_rows)

    # Export CSV
    _out = Path("outputs/tables")
    _out.mkdir(parents=True, exist_ok=True)
    etable9_df.write_csv(_out / "etable9_literature_benchmarking.csv")
    return (etable9_df,)


@app.cell
def _(etable9_df, mo):
    mo.vstack([
        mo.md("**eTable 9.** Published Morse Fall Scale Validation Studies: Benchmarking Comparison"),
        mo.ui.table(etable9_df),
        mo.md(
            "**Note:** Score timing indicates when the risk score was captured relative to "
            "the fall event. Studies using max-before-fall or encounter-maximum scores report "
            "higher discrimination than admission-time screening. No peer-reviewed publications "
            "of the Epic PMFRS (Inpatient Risk of Falls) model exist; Epic validation data are "
            "from the unpublished model brief (August 2025). Ji 2023 primarily compares a "
            "Clinical Frailty Scale against MFS\u2014the MFS data are valid for falls benchmarking "
            "but the frailty comparison context should be noted."
        ),
        mo.md("**eTable 9 saved** to `outputs/tables/etable9_literature_benchmarking.csv`."),
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            """
            ## 11. Sensitivity Analyses \u2014 eTable 4 & Figure 2

            Our primary analysis used the first risk score recorded at admission. But what if
            we used the highest score during the entire stay, or the score just before the fall?
            We tested four different score timings to see if the results change. We also checked
            whether having some patients with multiple hospital visits affects our conclusions.
            """
        ),
        mo.accordion({
            "Statistical details: sensitivity analysis methods": mo.md(
                """
                **Timing hierarchy**: admission (no bias), before-fall/discharge (moderate bias),
                max (high bias, includes post-fall scores), mean (moderate-high bias).

                **First-encounter-only analysis**: eliminates within-patient clustering (patients
                with multiple encounters).

                **Cluster bootstrap**: patient-level resampling, BCa CIs, 2000 reps per
                Obuchowski *Biometrics* 1997. Design effect: DEFF = 1 + (avg cluster size - 1) x ICC.

                **Lead-time analysis**: admission-to-fall interval quantifies the prediction-to-event
                window.
                """
            ),
        }),
    ])
    return


@app.cell
def _(
    SCORE_TIMING,
    cluster_bootstrap_auroc_comparison,
    delong_ci,
    delong_roc_test,
    df_analytic,
    estimate_design_effect,
    np,
    pl,
):
    def _run_timing(sub_df, epic_col, morse_col, timing_label, note=""):
        _complete = sub_df.filter(sub_df[epic_col].is_not_null() & sub_df[morse_col].is_not_null())
        _n = _complete.height
        _n_falls = int(_complete["fall_flag"].sum())
        _row = {"Timing strategy": timing_label, "N encounters": _n, "N falls": _n_falls}
        if _n_falls < 20 or (_n - _n_falls) < 20:
            _row.update({"Epic AUROC": "\u2014", "Epic 95% CI": "\u2014", "Morse AUROC": "\u2014",
                         "Morse 95% CI": "\u2014", "DeLong p": "\u2014", "Note": note or "Insufficient events"})
            return _row
        _y = _complete["fall_flag"].to_numpy()
        _es = _complete[epic_col].to_numpy()
        _ms = _complete[morse_col].to_numpy()
        _ea, _elo, _ehi = delong_ci(_y, _es)
        _ma, _mlo, _mhi = delong_ci(_y, _ms)
        _p = delong_roc_test(_y, _es, _ms)
        _p_str = "<0.001" if _p < 0.001 else (f"{_p:.3f}" if _p < 0.01 else f"{_p:.2f}")
        _row.update({"Epic AUROC": f"{_ea:.3f}", "Epic 95% CI": f"({_elo:.3f}\u2013{_ehi:.3f})",
                     "Morse AUROC": f"{_ma:.3f}", "Morse 95% CI": f"({_mlo:.3f}\u2013{_mhi:.3f})",
                     "DeLong p": _p_str, "Note": note})
        return _row

    # All timing strategies
    _rows = []
    for _key, _label, _note in [
        ("admission", "Admission (primary)", ""),
        ("before_fall", "Before fall / discharge", "Moderate look-ahead bias"),
        ("max", "Maximum (encounter)", "High look-ahead bias"),
        ("mean", "Mean (encounter)", "Moderate look-ahead bias"),
    ]:
        _t = SCORE_TIMING[_key]
        _rows.append(_run_timing(df_analytic, _t["epic"], _t["morse"], _label, _note))

    # First encounter per patient
    _df_first = df_analytic.sort("admission_date").group_by("mrn").first()
    _t_adm = SCORE_TIMING["admission"]
    _rows.append(_run_timing(_df_first, _t_adm["epic"], _t_adm["morse"],
                             "Admission \u2014 first encounter per patient",
                             "Independent observations per patient"))

    # Cluster bootstrap
    _y_all = df_analytic["fall_flag"].to_numpy()
    _epic_all = df_analytic[_t_adm["epic"]].to_numpy()
    _morse_all = df_analytic[_t_adm["morse"]].to_numpy()
    _mrn_all = df_analytic["mrn"].to_numpy()

    _deff = estimate_design_effect(_y_all, _epic_all, _mrn_all)
    _cb = cluster_bootstrap_auroc_comparison(_y_all, _epic_all, _morse_all, _mrn_all,
                                             n_bootstrap=2000, seed=42, method="bca")
    _cb_p_str = "<0.001" if _cb.p_value < 0.001 else f"{_cb.p_value:.3f}"
    _row_cluster = {
        "Timing strategy": "Admission \u2014 cluster bootstrap",
        "N encounters": _cb.n_obs, "N falls": int(_y_all.sum()),
        "Epic AUROC": f"{_cb.auc_a:.3f}",
        "Epic 95% CI": f"({_cb.ci_lower:.3f}\u2013{_cb.ci_upper:.3f})",
        "Morse AUROC": f"{_cb.auc_b:.3f}",
        "Morse 95% CI": "(cluster-adjusted)",
        "DeLong p": _cb_p_str,
        "Note": f"DEFF={_deff['deff_overall']}, ICC={_deff['icc_estimate']}",
    }
    _rows.append(_row_cluster)

    sensitivity_etable4_df = pl.DataFrame(_rows)

    # Lead-time analysis
    _fallers = df_analytic.filter(
        pl.col("fall_flag") == 1, pl.col("fall_datetime").is_not_null(), pl.col("admission_date").is_not_null()
    ).with_columns(
        ((pl.col("fall_datetime") - pl.col("admission_date")).dt.total_seconds() / 3600).alias("lead_time_hours")
    ).filter(pl.col("lead_time_hours") > 0)

    if _fallers.height > 0:
        _lt = _fallers["lead_time_hours"].to_numpy()
        lead_time_stats = {
            "n": len(_lt), "median_hours": round(float(np.median(_lt)), 1),
            "q25_hours": round(float(np.percentile(_lt, 25)), 1),
            "q75_hours": round(float(np.percentile(_lt, 75)), 1),
        }
    else:
        lead_time_stats = {"n": 0}

    # AUROC results for dot plot
    auroc_results = []
    _TIMING_LABELS = {"admission": "Admission", "before_fall": "Before fall", "max": "Maximum", "mean": "Mean"}
    for _key, _cols in SCORE_TIMING.items():
        _sub = df_analytic.filter(pl.col(_cols["epic"]).is_not_null() & pl.col(_cols["morse"]).is_not_null())
        _n = _sub.height
        _nf = int(_sub["fall_flag"].sum())
        if _nf < 20 or (_n - _nf) < 20:
            auroc_results.append({"timing_key": _key, "label": _TIMING_LABELS.get(_key, _key), "n": _n, "n_falls": _nf, "feasible": False})
            continue
        _y = _sub["fall_flag"].to_numpy()
        _ea, _elo, _ehi = delong_ci(_y, _sub[_cols["epic"]].to_numpy())
        _ma, _mlo, _mhi = delong_ci(_y, _sub[_cols["morse"]].to_numpy())
        auroc_results.append({"timing_key": _key, "label": _TIMING_LABELS.get(_key, _key), "n": _n, "n_falls": _nf,
                              "epic_auc": _ea, "epic_lo": _elo, "epic_hi": _ehi,
                              "morse_auc": _ma, "morse_lo": _mlo, "morse_hi": _mhi, "feasible": True})
    return auroc_results, lead_time_stats, sensitivity_etable4_df


@app.cell
def _(Path, lead_time_stats, mo, sensitivity_etable4_df):
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    sensitivity_etable4_df.write_csv(Path("outputs/tables/etable4_sensitivity.csv"))

    _lt_text = ""
    if lead_time_stats.get("n", 0) > 0:
        _lt_text = (
            f"\n\n**Admission-to-fall lead time** (n={lead_time_stats['n']}): "
            f"median {lead_time_stats['median_hours']:.1f} hours "
            f"(IQR {lead_time_stats['q25_hours']:.1f}\u2013{lead_time_stats['q75_hours']:.1f})"
        )

    mo.vstack([
        mo.md("### eTable 4. Sensitivity Analyses: AUROC Across Score Timing Strategies"),
        mo.ui.table(sensitivity_etable4_df),
        mo.md(f"**Saved**: `outputs/tables/etable4_sensitivity.csv`{_lt_text}"),
    ])
    return


@app.cell
def _(
    COLORS,
    FIG_DOUBLE_COL,
    JAMA_STYLE,
    MODEL_LABELS,
    auroc_results,
    matplotlib,
    plt,
    save_figure,
):
    _DISPLAY_ORDER = ["admission", "before_fall", "max", "mean"]
    _feasible = [r for r in auroc_results if r["feasible"]]
    _ordered = sorted(_feasible, key=lambda r: _DISPLAY_ORDER.index(r["timing_key"]) if r["timing_key"] in _DISPLAY_ORDER else 99)
    _n_rows = len(_ordered)
    _OFFSET = 0.15

    with matplotlib.rc_context(JAMA_STYLE):
        _fig2, _ax = plt.subplots(figsize=FIG_DOUBLE_COL)

        for _i, _r in enumerate(_ordered):
            _ax.errorbar(x=_r["epic_auc"], y=_i + _OFFSET,
                         xerr=[[_r["epic_auc"] - _r["epic_lo"]], [_r["epic_hi"] - _r["epic_auc"]]],
                         fmt="o", color=COLORS["epic"], ecolor=COLORS["epic"], elinewidth=1.0,
                         capsize=3, capthick=1.0, markersize=6, zorder=3,
                         label=MODEL_LABELS["epic"] if _i == 0 else "_nolegend_")
            _ax.errorbar(x=_r["morse_auc"], y=_i - _OFFSET,
                         xerr=[[_r["morse_auc"] - _r["morse_lo"]], [_r["morse_hi"] - _r["morse_auc"]]],
                         fmt="D", color=COLORS["morse"], ecolor=COLORS["morse"], elinewidth=1.0,
                         capsize=3, capthick=1.0, markersize=5, zorder=3,
                         label=MODEL_LABELS["morse"] if _i == 0 else "_nolegend_")

        _ax.axvline(0.5, color="#333333", linestyle="--", linewidth=0.75, zorder=1, label="No discrimination")
        _ax.set_yticks(range(_n_rows))
        _ax.set_yticklabels([r["label"] for r in _ordered], fontsize=9)
        _ax.set_xlabel("AUROC (95% CI)", fontsize=9)
        _ax.set_xlim(0.45, 1.02)
        _ax.set_ylim(-0.6, _n_rows - 0.4)

        for _i, _r in enumerate(_ordered):
            _ax.text(1.01, _i, f"n falls={_r['n_falls']}", transform=_ax.get_yaxis_transform(),
                     va="center", ha="left", fontsize=8, color="#555555", clip_on=False)

        _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8,
                   frameon=False, handlelength=1.5)

        _fig2.text(
            0.5, -0.22,
            "Figure 2. Model discrimination across score timing strategies",
            ha="center", va="top", fontsize=10, fontweight="bold",
        )

        _fig2.subplots_adjust(left=0.12, right=0.78, bottom=0.18, top=0.95)

    save_figure(_fig2, "figure2_dot_plot", bbox_inches=None, pad_inches=0.15)
    return


@app.cell
def _(Path, json, lead_time_stats, mo, sensitivity_etable4_df):
    _adm_row = sensitivity_etable4_df.filter(
        sensitivity_etable4_df["Timing strategy"] == "Admission (primary)"
    ).row(0, named=True)

    _results_dict = {
        "primary_epic_auroc": _adm_row["Epic AUROC"],
        "primary_epic_ci": _adm_row["Epic 95% CI"],
        "primary_morse_auroc": _adm_row["Morse AUROC"],
        "primary_morse_ci": _adm_row["Morse 95% CI"],
        "primary_delong_p": _adm_row["DeLong p"],
        "n_encounters": _adm_row["N encounters"],
        "n_falls": _adm_row["N falls"],
        "lead_time": lead_time_stats,
    }

    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables/sensitivity_key_results.json").write_text(json.dumps(_results_dict, indent=2))

    mo.md("**Figure 2 saved** to `outputs/figures/figure2_dot_plot.pdf` and `.png`. Key results exported to `sensitivity_key_results.json`.")
    return


@app.cell
def _(SCORE_TIMING, df_analytic, pl):
    from utils.metrics import timing_classification_metrics

    _etable10_rows = []

    _TIMINGS = [
        ("before_fall", "Before fall / discharge", "Moderate look-ahead bias"),
        ("max", "Maximum (encounter)", "HIGH look-ahead bias — post-fall scores included"),
        ("mean", "Mean (encounter)", "Moderate-high look-ahead bias — post-fall scores included"),
    ]

    for _key, _label, _note in _TIMINGS:
        _t = SCORE_TIMING[_key]
        _epic_col, _morse_col = _t["epic"], _t["morse"]

        _sub = df_analytic.filter(
            pl.col(_epic_col).is_not_null() & pl.col(_morse_col).is_not_null()
        )
        _n = _sub.height
        _n_falls = int(_sub["fall_flag"].sum())

        if _key == "before_fall":
            _n_nonfallers = _n - _n_falls
            if _n_nonfallers < 100:
                continue

        if _n_falls < 20 or (_n - _n_falls) < 20:
            continue

        _y = _sub["fall_flag"].to_numpy()
        _epic = _sub[_epic_col].to_numpy()
        _morse = _sub[_morse_col].to_numpy()

        _etable10_rows.extend(
            timing_classification_metrics(_y, _epic, _morse, _label, _note)
        )

    sensitivity_etable10_df = pl.DataFrame(_etable10_rows)
    return (sensitivity_etable10_df,)


@app.cell
def _(Path, mo, sensitivity_etable10_df):
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    sensitivity_etable10_df.write_csv(Path("outputs/tables/etable10_timing_classification.csv"))

    mo.vstack([
        mo.md("### eTable 10. Classification Metrics Across Score Timing Strategies"),
        mo.ui.table(sensitivity_etable10_df),
        mo.md("**Saved**: `outputs/tables/etable10_timing_classification.csv`"),
    ])
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **Interpretation — eTable 10**

            - At maximum encounter score, Epic's standard ≥35 threshold flags substantially more
              encounters than at admission (where 97% score <35), demonstrating the threshold
              calibration gap between continuous monitoring and admission screening.

            - Epic's validation reported 82–86% sensitivity at thresholds 7–21 using
              max-score-before-fall. Compare our max-timing MFS ≥45 sensitivity with their
              reported range (80.5–87.6%).

            - Max and mean scores include post-fall assessments — these are upper-bound estimates,
              not clinically actionable. Thresholds applied to raw scores (no recalibration).
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(Path, df_analytic, gee_discrimination_check, mo, pl):
    # GEE supplementary clustering analysis
    _gee_cols = ["fall_flag", "epic_score_admission", "morse_score_admission", "mrn"]
    _gee_pdf = df_analytic.select(_gee_cols).to_pandas()

    _gee_results = gee_discrimination_check(
        _gee_pdf,
        score_col_a="epic_score_admission",
        score_col_b="morse_score_admission",
        outcome_col="fall_flag",
        cluster_col="mrn",
    )

    _gee_rows = []
    for _label, _key in [("Epic PMFRS", "model_a"), ("Morse Fall Scale", "model_b")]:
        _r = _gee_results.get(_key, {})
        if "error" in _r:
            _gee_rows.append({"Model": _label, "Status": f"Error: {_r['error']}"})
        else:
            _gee_rows.append({
                "Model": _label,
                "Score column": _r.get("score_col", ""),
                "Coefficient": f"{_r.get('coef', 0):.4f}",
                "Robust SE": f"{_r.get('se_robust', 0):.4f}",
                "P-value": f"{_r.get('p_value', 1):.4e}",
                "AUROC (GEE predictions)": f"{_r.get('auroc_gee_predictions', 0):.4f}",
                "ICC (exchangeable)": str(_r.get("icc_exchangeable", "")),
                "N clusters": str(_r.get("n_clusters", "")),
            })

    _gee_df = pl.DataFrame(_gee_rows)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    _gee_df.write_csv(Path("outputs/tables/gee_clustering_results.csv"))

    mo.vstack([
        mo.md(
            """
            ### GEE Supplementary Analysis: Clustering Adjustment

            Generalized Estimating Equations (GEE) with exchangeable correlation
            and robust (sandwich) standard errors account for within-patient
            clustering by `mrn`. This is a supplementary check — the primary
            analysis uses the DeLong test with cluster bootstrap sensitivity
            analysis above.
            """
        ),
        mo.ui.table(_gee_df),
        mo.md("**Saved**: `outputs/tables/gee_clustering_results.csv`"),
    ])
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **Planned future analysis: time-dependent AUC**

            Notebook 14 (`14_time_dependent_auc.py`) computes landmark AUROC by
            hospital day to assess how discrimination changes over the encounter.
            This analysis requires a timestamped score extract (individual score
            observations with timestamps) that is not yet available from the data
            team. Results will be added when the extract is provided.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(
    delong_p,
    epic_auc,
    epic_ci_hi,
    epic_ci_lo,
    epic_scores,
    mo,
    morse_auc,
    morse_ci_hi,
    morse_ci_lo,
    np,
):
    _pct_below_35 = f"{np.mean(epic_scores < 35) * 100:.0f}"
    mo.vstack([
        mo.md("## 12. Key Findings"),
        mo.callout(
            mo.md(
                """
                **What this means for clinical practice**

                We compared two tools for identifying patients at risk of falling during their
                hospital stay: the **Epic computer model** (which runs automatically) and the
                **Morse Fall Scale** (which nurses complete at the bedside). Here is what we found:

                - Both tools can distinguish fallers from non-fallers, but neither is dramatically
                  better than the other at the time of admission
                - The Epic model's built-in "high risk" thresholds (designed for continuous monitoring)
                  flag almost no one when applied at admission — they are not appropriate for
                  admission screening
                - The Morse scale's standard \u226545 "high risk" cutoff flags about two-thirds of
                  patients, catching most fallers but generating many alerts
                - Switching from Morse to Epic would not substantially improve which patients
                  get flagged as high-risk
                """
            ),
            kind="info",
        ),
        mo.md(
            f"""
            ### Statistical details

            - **Discrimination**: Epic PMFRS AUROC {epic_auc:.3f} ({epic_ci_lo:.3f}\u2013{epic_ci_hi:.3f}) vs Morse Fall Scale {morse_auc:.3f} ({morse_ci_lo:.3f}\u2013{morse_ci_hi:.3f}); DeLong p = {delong_p:.4f}
            - **Calibration**: Both models require logistic recalibration when applied at admission; the raw scores are not calibrated probabilities
            - **Reclassification**: NRI and IDI quantify the practical impact of switching tools; direction and magnitude depend on the threshold chosen
            - **Fairness**: Both models' discrimination varies across demographic subgroups; minimum 20 events required for reliable subgroup AUROC
            - **Clinical bottom line**: Epic's recommended thresholds (35/70/50) were designed for continuous monitoring, not admission screening \u2014 at admission, {_pct_below_35}% of encounters score <35
            """
        ),
    ])
    return


@app.cell
def _(mo):
    mo.accordion({
        "Technical discussion and methods summary": mo.md(
            """
            **Comparison to Epic validation data**: Epic's own validation (3 sites, 0.38-0.54%
            prevalence) used max-score-before-fall at the encounter level, achieving 82-86%
            sensitivity at thresholds 7-21. Our admission-time analysis operates on a different
            time scale \u2014 a single score at a single time point \u2014 which explains why Epic's
            recommended thresholds are poorly calibrated for this use case.

            **Limitations**: Single-center retrospective study; treatment paradox (high-risk
            patients receive fall prevention interventions); no injury severity data; Epic PMFRS
            score may not represent the integer-rounded bedside display.

            **Methods parameters**: seed=42, 2000 bootstrap resamples, alpha=0.05, DeLong test
            for primary paired comparison, cluster bootstrap (patient-level, BCa) for clustering
            sensitivity analysis, LOWESS frac=0.3, DCA threshold range 0-10%, minimum 20
            subgroup events.

            **TRIPOD+AI compliance**: This notebook addresses Items 1-22 of the TRIPOD+AI checklist
            (Collins et al. BMJ 2024) for external validation studies (Type 3).
            """
        ),
    })
    return


@app.cell
def _(epic_auc, mo, morse_auc):
    mo.vstack([
        mo.md("### Literature Comparison: MFS Discrimination Across Published Validation Studies"),
        mo.callout(
            mo.md(
                f"""
                **Benchmarking interpretation** (see **eTable 9** for the full comparison)

                - Our MFS AUROC ({morse_auc:.3f}) is consistent with methodologically comparable
                  prospective studies: Ji 2023 (0.63), Shim 2022 (0.65), Lohse 2021 (0.64).
                  Jiang 2025 (N\u2009=\u2009206,846; AUROC 0.825) used within-24h-of-admission
                  scoring at very low prevalence (0.10%), which may partly explain its higher
                  discrimination
                - Higher published AUROCs (0.76\u20130.86) reflect case-control inflation \u2014
                  retrospective designs inflate the Youden Index by +0.22 on average
                  (Haines TP et al, *J Gerontol A Biol Sci Med Sci*, 2007; PMID: 17595425)
                - Our Epic PMFRS AUROC ({epic_auc:.3f}) has no directly comparable published
                  benchmarks at admission \u2014 no peer-reviewed publications of the Epic PMFRS
                  exist; Epic's own unpublished validation used max-score-before-fall, which
                  inflates discrimination relative to admission-time screening
                - Sensitivity at MFS \u226545 (83.8%) falls within the published range (31\u201388%)
                  and is consistent with Epic's validation sites (80.5\u201387.6%)
                - Low specificity (34.6%) reflects our cohort's score distribution (65.7% score
                  \u226545 at admission) and is an inherent trade-off at this threshold
                - PPV is universally low at ~1% prevalence \u2014 this is inherent to the base rate,
                  not a failure of either model
                """
            ),
            kind="info",
        ),
    ])
    return


@app.cell
def _(Path, mo):
    _figures = [
        "efigure4_cohort_flow", "figure1_discrimination",
        "recalibration_mapping",
        "efigure1_calibration_epic", "efigure2_calibration_morse",
        "figure3_dca", "efigure3_threshold_overlay",
        "efigure5_score_distributions", "figure2_dot_plot",
        "fall_locations", "department_fall_rates",
        "fall_temporal", "admission_to_fall_hours",
        "los_comparison",
    ]
    _tables = [
        "table1.csv", "table2.csv", "table3.csv",
        "etable5_calibration_summary.csv", "dca_threshold_ranges.json",
        "efigure4_cohort_flow_counts.csv",
        "etable7_dca_net_benefit.csv",
        "etable6_threshold_methods.csv", "etable6_flag_rate_summary.csv",
        "etable1_age.csv", "etable2_race.csv", "etable3_unit.csv",
        "etable8_gender.csv", "etable4_sensitivity.csv",
        "etable9_literature_benchmarking.csv",
        "etable10_timing_classification.csv",
        "sensitivity_key_results.json",
        "gee_clustering_results.csv",
        "fall_location_summary.csv", "department_fall_rates.csv",
        "fall_temporal_hourly.csv", "fall_temporal_daily.csv",
    ]

    _fig_lines = []
    for _name in _figures:
        _pdf = Path(f"outputs/figures/{_name}.pdf")
        _png = Path(f"outputs/figures/{_name}.png")
        _pdf_kb = round(_pdf.stat().st_size / 1024, 1) if _pdf.exists() else "\u2014"
        _png_kb = round(_png.stat().st_size / 1024, 1) if _png.exists() else "\u2014"
        _fig_lines.append(f"| {_name} | {'Yes' if _pdf.exists() else 'No'} | {_pdf_kb} KB | {_png_kb} KB |")

    _tbl_lines = []
    for _name in _tables:
        _p = Path(f"outputs/tables/{_name}")
        _kb = round(_p.stat().st_size / 1024, 1) if _p.exists() else "\u2014"
        _tbl_lines.append(f"| {_name} | {'Yes' if _p.exists() else 'No'} | {_kb} KB |")

    _parquet = Path("data/processed/analytic.parquet")
    _pq_kb = round(_parquet.stat().st_size / 1024, 1) if _parquet.exists() else "\u2014"

    mo.md(
        "## 13. Output Inventory\n\n"
        "### Figures\n\n"
        "| Name | Exists | PDF | PNG |\n|---|---|---|---|\n"
        + "\n".join(_fig_lines)
        + "\n\n### Tables\n\n"
        "| Name | Exists | Size |\n|---|---|---|\n"
        + "\n".join(_tbl_lines)
        + f"\n\n### Analytic Dataset\n\n"
        f"| analytic.parquet | {'Yes' if _parquet.exists() else 'No'} | {_pq_kb} KB |"
    )
    return


@app.cell
def _(Path, mo):
    import importlib.metadata as _meta
    import sys as _sys
    from datetime import datetime as _dt

    def _pkg_version(name):
        try:
            return _meta.version(name)
        except _meta.PackageNotFoundError:
            return "\u2014"

    _parquet = Path("data/processed/analytic.parquet")

    mo.md(
        f"""
        ### Session Info

        | Component | Version |
        |---|---|
        | Python | {_sys.version.split()[0]} |
        | polars | {_pkg_version('polars')} |
        | scikit-learn | {_pkg_version('scikit-learn')} |
        | marimo | {_pkg_version('marimo')} |
        | dcurves | {_pkg_version('dcurves')} |
        | statsmodels | {_pkg_version('statsmodels')} |
        | matplotlib | {_pkg_version('matplotlib')} |
        | great-tables | {_pkg_version('great-tables')} |

        **Timestamp**: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}
        **Random seed**: 42
        **Bootstrap replicates**: 2000
        **Analytic cohort**: {_parquet.exists() and 'written' or 'not found'}
        """
    )
    return


if __name__ == "__main__":
    app.run()
