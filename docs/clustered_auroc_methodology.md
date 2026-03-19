# Comparing AUROCs with Within-Patient Clustering: Statistical Methodology

## Study Context

- 85,394 encounters from many unique patients
- 861 fall events from 778 unique patients (some patients fell on multiple admissions)
- Two models (Epic PMFRS, Morse Fall Scale) evaluated on the same encounters
- Unit of analysis: encounter. Clustering variable: patient (MRN).

---

## 1. Is the Standard DeLong Test Valid with Clustered Data?

**Short answer: No, it is not strictly valid. But the practical impact in this dataset is likely small.**

### The Theory

DeLong et al. (1988) derive the variance of the AUC estimator (and the covariance between two correlated AUCs) under the assumption that observations are **independent**. The DeLong test statistic is:

```
Z = (AUC_1 - AUC_2) / sqrt(Var(AUC_1) + Var(AUC_2) - 2*Cov(AUC_1, AUC_2))
```

When observations within a patient are correlated (within-patient clustering), the DeLong variance estimator **underestimates** the true variance because it treats correlated encounters as if they contribute fully independent information. This leads to:

- **Narrower CIs** than they should be
- **Inflated Type I error** (rejecting the null more often than the nominal alpha)
- The p-value from the naive DeLong test is **anti-conservative**

### The Key Reference

**Obuchowski NA (1997). "Nonparametric analysis of clustered ROC curve data." Biometrics 53(2):567-578.**

Obuchowski showed via Monte Carlo simulation that "the size of statistical tests that assume independence is inflated in the presence of intracluster correlation." She extended DeLong's structural components method using the **design effect** and **effective sample size** concepts from survey sampling (Rao & Scott 1992).

### The Mechanism

The design effect for the AUC variance is analogous to the classic formula:

```
DEFF = 1 + (m_bar - 1) * rho
```

where:
- `m_bar` = average cluster size (encounters per patient)
- `rho` = intracluster correlation coefficient for the diagnostic scores/ranks

The **effective sample size** is:

```
N_eff = N_total / DEFF
```

When DEFF > 1, the naive DeLong variance is too small by approximately a factor of DEFF, and the CIs are too narrow by approximately sqrt(DEFF).

---

## 2. How Critical Is This Adjustment for YOUR Data?

### Quantifying the Clustering

For your data, the clustering is **asymmetric** and **mild**:

**Fall encounters (the side that matters most for AUC):**
- 861 fall encounters from 778 unique patients
- Average cluster size among fallers: 861/778 = 1.107
- Only ~80 patients had >1 fall encounter (861 - 778 = 83 "extra" encounters)
- Most fall patients (approximately 700 of 778) contributed exactly 1 fall encounter

**Non-fall encounters:**
- ~84,500 non-fall encounters from many unique patients
- Some patients may have multiple non-fall admissions, but the large number of unique patients means the average cluster size is modest

### Design Effect Estimate

For the fall-encounter side, even with a generous ICC of rho = 0.5 for within-patient fall risk scores:

```
DEFF_falls = 1 + (1.107 - 1) * 0.5 = 1.054
```

This means the effective sample size for fallers is:

```
N_eff_falls = 861 / 1.054 = 817
```

A 5.4% reduction. The CI would widen by a factor of sqrt(1.054) = 1.027, or about **2.7%**.

Even with rho = 0.9 (extreme):

```
DEFF_falls = 1 + 0.107 * 0.9 = 1.096
CI widening = sqrt(1.096) = 1.047, about 4.7%
```

### Practical Bottom Line

**The naive DeLong result and the cluster-adjusted result will likely be very similar for this dataset.** The clustering is too mild (average cluster size barely above 1.0) to produce meaningfully different CIs or p-values. However, you should still perform the cluster-bootstrap analysis as a **sensitivity check** because:

1. A reviewer will ask about it (you already know this)
2. It takes minimal additional effort
3. If the results agree, you can state: "Results were robust to accounting for within-patient clustering (cluster-bootstrap 95% CI: [...] vs. naive DeLong 95% CI: [...])"
4. If they disagree, you would need to use the cluster-adjusted result

### Recommended Manuscript Language

> "Because some patients contributed multiple encounters, the primary DeLong analysis treats encounters as independent. As a sensitivity analysis, we performed cluster-bootstrap resampling at the patient level (2000 replicates, BCa intervals) to account for within-patient correlation. Results were concordant [or: CIs widened by <X percentage points], supporting the robustness of the primary findings."

---

## 3. Obuchowski (1997) Method

### How It Works

Obuchowski extends DeLong's structural components approach:

1. **DeLong's structural components (V10, V01)**: For each observation, compute its "placement value" -- the proportion of opposite-class observations it is correctly ranked against. The variance of AUC is computed from the variance of these placement values.

2. **Obuchowski's extension**: When observations are clustered, the placement values within a cluster are correlated. She adjusts the variance using a design effect:

```
Var_clustered(AUC) = Var_naive(AUC) * DEFF
```

where DEFF is estimated from the data by computing the ratio of the cluster-aware variance to the independence-assumed variance of the structural components.

3. **For comparing two AUCs**: The same design-effect adjustment applies to the variance of the AUC difference, including the covariance term.

### How It Differs from DeLong

| Feature | DeLong (1988) | Obuchowski (1997) |
|---------|--------------|-------------------|
| Assumes independence | Yes | No |
| Accounts for clustering | No | Yes, via design effect |
| Nonparametric | Yes | Yes |
| Handles unequal cluster sizes | N/A | Yes |
| Handles discordant outcomes within cluster | N/A | Yes (some subunits diseased, some not) |
| Software | Widely available (R pROC, Python) | R package (Cleveland Clinic), limited Python |

### Relationship to the Extended Mann-Whitney Approach

Goksuluk et al. (2022, PMC8586066) and the Leisenring-Pepe-Longton (2000, PMC3622772) paper further extend this by separating:
- **theta_c**: the AUC for within-cluster comparisons (comparing a diseased subunit to a non-diseased subunit from the SAME cluster)
- **theta**: the AUC for between-cluster comparisons

In your study, this distinction is less relevant because the outcome is at the encounter level and most patients have only 1 fall encounter, so within-cluster discordant pairs are rare.

---

## 4. Cluster-Bootstrap DeLong: The Recommended Approach

For your study, the **cluster bootstrap** is the most practical and defensible approach. It is:
- Easy to implement in Python
- Does not require specialized software
- Makes no parametric assumptions about the correlation structure
- Produces valid CIs even with complex clustering patterns
- Directly comparable to the naive DeLong for a sensitivity check

### Algorithm

```
For b = 1, ..., B (e.g., B = 2000):
    1. Get unique patient IDs from the dataset
    2. Sample N_patients patient IDs WITH REPLACEMENT
    3. For each sampled patient, include ALL of their encounters
       (if a patient is sampled k times, their encounters appear k times)
    4. Compute AUC_epic(b) and AUC_morse(b) on the bootstrap sample
    5. Compute delta(b) = AUC_epic(b) - AUC_morse(b)

CI_percentile = [delta_(alpha/2), delta_(1-alpha/2)]
CI_BCa = bias-corrected and accelerated interval
```

### Why Resample Patients, Not Encounters

If you resample encounters, you break the within-patient correlation structure. A bootstrap sample might include encounter 1 from patient A but not encounter 2, artificially reducing the correlation. By resampling at the patient (cluster) level, you preserve the entire correlation structure within each patient.

---

## 5. GEE Approach

### Can GEE Compare Model Discrimination with Clustering?

GEE can account for within-patient clustering when fitting a model, but extracting a **cluster-adjusted C-statistic** from a GEE model is not straightforward. Here is what you can and cannot do:

**What GEE does well:**
- Provides cluster-robust standard errors for regression coefficients
- Tests whether a predictor (score) is significantly associated with the outcome after accounting for clustering
- Compares nested models via QIC (quasi-information criterion)

**What GEE does NOT directly provide:**
- A cluster-adjusted AUROC/C-statistic
- A direct analog of the DeLong test for two non-nested prediction scores

### The GEE-Based Comparison Strategy

You can use GEE indirectly to compare discrimination:

1. **Fit two separate GEE models:**
   - Model 1: `fall ~ epic_score`, groups=mrn, Binomial(logit), Exchangeable
   - Model 2: `fall ~ morse_score`, groups=mrn, Binomial(logit), Exchangeable

2. **Extract predicted probabilities** from each model (these are marginal predictions accounting for clustering)

3. **Compare the AUROCs** of the predicted probabilities using the cluster bootstrap

However, this is **not the same question** as comparing the raw scores' AUROCs. The GEE predictions are recalibrated versions of the scores. For a head-to-head comparison of the scores themselves (which is your primary question), the cluster bootstrap on raw scores is more appropriate.

### When GEE Is Useful in Your Study

GEE is useful for:
- **Notebook 04 (primary analysis)**: As a sensitivity check showing that the score-outcome association is robust to clustering
- **Reporting cluster-robust p-values** for the score coefficients
- **Checking whether the exchangeable ICC is substantial** (if the estimated ICC is near zero, clustering is negligible)

---

## 6. Published Examples

### Direct Examples of Clustered AUROC Methods in Major Journals

Most published fall-risk validation studies (and EHR-based prediction model validations more broadly) **do not explicitly account for within-patient clustering** in their AUROC comparisons. This is a common gap. Examples:

**Studies that ignored clustering (typical practice):**
- Lindberg et al. (Int J Med Inform 2020): Compared ML models vs Morse Fall Scale using pairwise t-tests on cross-validated AUROCs, without accounting for repeated patient encounters.
- Choi et al. (Am J Health Syst Pharm 2018): Reported AUROCs with standard DeLong CIs; no clustering adjustment mentioned.
- Park et al. (PMC9834835, 2023): Compared gradient boosting vs Morse Fall Score AUROCs without cluster adjustment.

**Studies that addressed clustering:**
- Goksuluk et al. (Ophthalmology tutorial, PMC8586066, 2022): Applied both Obuchowski nonparametric and cluster bootstrap to ophthalmology data (correlated eyes). Found that for low inter-eye correlation (kappa ~0.13), naive and cluster-adjusted CIs were "very similar." For high correlation (kappa ~0.98), naive CIs were "substantially narrower."
- Leisenring et al. (Biometrics 2000; PMC3622772): Extended Mann-Whitney for clustered risk prediction rules in AMD progression, explicitly separating within- and between-cluster discrimination.
- Wong et al. (JAMA Netw Open 2026): Your blueprint paper -- check whether they addressed clustering. Most large EHR validation studies with >50,000 encounters report standard DeLong and note clustering as a limitation.

### The Reviewer-Proof Approach

The most defensible strategy for a JAMA Network Open submission:

1. **Primary analysis**: Standard DeLong (this is what readers expect and can compare to other studies)
2. **Sensitivity analysis**: Cluster bootstrap at the patient level
3. **Report both**: If concordant, state robustness. If discordant, use the cluster-adjusted result as primary.
4. **First-encounter-only analysis**: As an additional sensitivity analysis, restrict to each patient's first encounter, eliminating clustering entirely (already in your plan as notebook 10)

---

## 7. Python Implementation

The implementation file is provided separately at:
`/Users/JCR/Desktop/rushfalla/utils/cluster_auroc.py`

---

## Key References

### Methods

1. **DeLong ER, DeLong DM, Clarke-Pearson DL.** Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*. 1988;44(3):837-845.

2. **Obuchowski NA.** Nonparametric analysis of clustered ROC curve data. *Biometrics*. 1997;53(2):567-578. -- THE key paper for your situation.

3. **Sun X, Xu W.** Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*. 2014;21(11):1389-1393.

4. **Leisenring W, Pepe MS, Longton G.** A marginal regression modelling framework for evaluating medical diagnostic tests. *Stat Med*. 1997;16(11):1263-1281.

5. **Goksuluk D, et al.** Tutorial on Biostatistics: Receiver-Operating Characteristic (ROC) Analysis for Correlated Eye Data. *Ophthalmology*. 2022; PMC8586066. -- Excellent practical tutorial comparing Obuchowski, cluster bootstrap, and naive methods.

6. **Rao JNK, Scott AJ.** A simple method for the analysis of clustered binary data. *Biometrics*. 1992;48:577-585. -- Foundation for the design effect approach.

7. **Cameron AC, Miller DL.** A practitioner's guide to cluster-robust inference. *J Human Resources*. 2015;50(2):317-372. -- General reference for cluster bootstrap methodology.

### Software

8. **R package (Cleveland Clinic)**: Obuchowski's clusteredROC functions: https://www.lerner.ccf.org/quantitative-health/documents/clusteredROC_help.pdf

9. **Python DeLong**: Yandex fast DeLong implementation: https://github.com/yandexdataschool/roc_comparison

### Reporting Guidelines

10. **Collins GS, et al.** TRIPOD+AI statement. *BMJ*. 2024;385:e078378. -- Item 10a: describe how clustering/correlation was handled.
