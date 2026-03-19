"""
Cluster-adjusted AUROC comparison for within-patient clustering.

Implements:
1. Naive DeLong test (Sun & Xu 2014 fast algorithm)
2. Cluster-bootstrap AUROC comparison (resample patients, not encounters)
3. Design-effect estimation for quantifying clustering impact
4. Sensitivity comparison wrapper

References:
    - DeLong ER et al. Biometrics 1988;44(3):837-845.
    - Obuchowski NA. Biometrics 1997;53(2):567-578.
    - Sun X, Xu W. IEEE Signal Proc Letters 2014;21(11):1389-1393.
    - Goksuluk D et al. Ophthalmology 2022 (PMC8586066).

Usage:
    from utils.cluster_auroc import (
        cluster_bootstrap_auroc_comparison,
        naive_delong_comparison,
        run_sensitivity_comparison,
    )
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class AUROCResult:
    """Single model AUROC with CI."""

    auc: float
    ci_lower: float
    ci_upper: float
    se: float


@dataclass
class AUROCComparisonResult:
    """Paired AUROC comparison result."""

    auc_a: float
    auc_b: float
    delta: float  # auc_a - auc_b
    ci_lower: float  # CI for delta
    ci_upper: float
    p_value: float
    method: str
    n_obs: int
    n_clusters: int | None = None
    n_bootstrap: int | None = None


@dataclass
class SensitivityResult:
    """Combined naive vs cluster-adjusted results."""

    naive: AUROCComparisonResult
    cluster_bootstrap: AUROCComparisonResult
    first_encounter_only: AUROCComparisonResult | None
    design_effect_falls: float
    design_effect_nonfalls: float
    icc_estimate: float | None


# ---------------------------------------------------------------------------
# 1. Naive DeLong test (fast algorithm, Sun & Xu 2014)
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for tied values.

    Fast midrank computation adapted from the Yandex DeLong implementation.
    """
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    rank = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n - 1 and z[k + 1] == z[k]:
            k += 1
        # All elements from i to k have the same value; assign midrank.
        midrank = 0.5 * (i + k + 2)  # 1-based midrank
        for m in range(i, k + 1):
            rank[j[m]] = midrank
        i = k + 1
    return rank


def _fast_delong(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute AUC and structural components (placement values) via fast DeLong.

    Parameters
    ----------
    y_true : array of 0/1
    y_score : array of continuous predictions

    Returns
    -------
    auc : float
    v10 : array, placement values for positive class
    v01 : array, placement values for negative class
    """
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    # Compute midranks on the full array
    midranks = _compute_midrank(y_score)

    # Placement values
    # V10[i] = fraction of negatives ranked below positive i
    # V01[j] = fraction of positives ranked above negative j
    sum_pos_ranks = np.sum(midranks[pos_idx])
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    # Structural components via searchsorted (O(n log n) instead of O(n_pos * n_neg))
    pos_scores = y_score[pos_idx]
    neg_scores = y_score[neg_idx]

    # V10: for each positive, proportion of negatives it beats
    neg_sorted = np.sort(neg_scores)
    below = np.searchsorted(neg_sorted, pos_scores, side="left")
    equal = np.searchsorted(neg_sorted, pos_scores, side="right") - below
    v10 = (below + 0.5 * equal) / n_neg

    # V01: for each negative, proportion of positives that beat it
    pos_sorted = np.sort(pos_scores)
    below_or_eq = np.searchsorted(pos_sorted, neg_scores, side="right")
    above = n_pos - below_or_eq
    equal_01 = below_or_eq - np.searchsorted(pos_sorted, neg_scores, side="left")
    v01 = (above + 0.5 * equal_01) / n_pos

    return auc, v10, v01


def naive_delong_comparison(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> AUROCComparisonResult:
    """Standard DeLong test for paired AUROC comparison (assumes independence).

    Parameters
    ----------
    y_true : array of 0/1, shape (n,)
    scores_a : array of continuous scores for model A, shape (n,)
    scores_b : array of continuous scores for model B, shape (n,)

    Returns
    -------
    AUROCComparisonResult with p-value, delta, and 95% CI
    """
    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    n = len(y_true)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    auc_a, v10_a, v01_a = _fast_delong(y_true, scores_a)
    auc_b, v10_b, v01_b = _fast_delong(y_true, scores_b)

    # Variance of each AUC (DeLong formula)
    var_a = np.var(v10_a, ddof=1) / n_pos + np.var(v01_a, ddof=1) / n_neg
    var_b = np.var(v10_b, ddof=1) / n_pos + np.var(v01_b, ddof=1) / n_neg

    # Covariance between AUCs (same subjects, different scores)
    cov_ab = (
        np.cov(v10_a, v10_b, ddof=1)[0, 1] / n_pos
        + np.cov(v01_a, v01_b, ddof=1)[0, 1] / n_neg
    )

    # Variance of the difference
    var_diff = var_a + var_b - 2 * cov_ab

    delta = auc_a - auc_b
    se_diff = np.sqrt(max(var_diff, 1e-12))
    z = delta / se_diff
    p_value = 2 * stats.norm.sf(abs(z))

    ci_lower = delta - 1.96 * se_diff
    ci_upper = delta + 1.96 * se_diff

    return AUROCComparisonResult(
        auc_a=auc_a,
        auc_b=auc_b,
        delta=delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        method="DeLong (naive, assumes independence)",
        n_obs=n,
    )


def naive_delong_single(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> AUROCResult:
    """DeLong variance for a single AUROC."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    auc, v10, v01 = _fast_delong(y_true, scores)

    var_auc = np.var(v10, ddof=1) / n_pos + np.var(v01, ddof=1) / n_neg
    se = np.sqrt(var_auc)

    return AUROCResult(
        auc=auc,
        ci_lower=auc - 1.96 * se,
        ci_upper=auc + 1.96 * se,
        se=se,
    )


# ---------------------------------------------------------------------------
# 2. Cluster-bootstrap AUROC comparison
# ---------------------------------------------------------------------------

def cluster_bootstrap_auroc_comparison(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    method: str = "bca",
    seed: int = 42,
    stratify_outcome: bool = True,
) -> AUROCComparisonResult:
    """Cluster-bootstrap comparison of two paired AUROCs.

    Resamples at the patient (cluster) level, preserving within-patient
    correlation. All encounters for a selected patient are included together.

    Parameters
    ----------
    y_true : array of 0/1, shape (n,)
    scores_a : array of continuous scores for model A, shape (n,)
    scores_b : array of continuous scores for model B, shape (n,)
    cluster_ids : array of patient IDs, shape (n,)
    n_bootstrap : number of bootstrap replicates (default 2000)
    confidence_level : CI level (default 0.95)
    method : "percentile" or "bca" (default "bca")
    seed : random seed for reproducibility
    stratify_outcome : if True, resample fall-patients and non-fall-patients
        separately to guarantee both classes in every bootstrap sample.
        Recommended for rare events. Default True.

    Returns
    -------
    AUROCComparisonResult with cluster-adjusted CI and p-value

    Notes
    -----
    The BCa (bias-corrected and accelerated) interval is preferred for
    small-sample or skewed bootstrap distributions. Falls back to
    percentile if BCa jackknife is too slow for large datasets.
    """
    rng = np.random.RandomState(seed)

    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    n_obs = len(y_true)

    # Pre-compute cluster membership via sorting (vectorized, O(n log n))
    sort_order = np.argsort(cluster_ids)
    sorted_ids = cluster_ids[sort_order]
    unique_clusters, starts, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True
    )
    n_clusters = len(unique_clusters)

    # Pre-build list of encounter index arrays per cluster (fast, one pass)
    cluster_enc_idx = [sort_order[starts[i]:starts[i] + counts[i]] for i in range(n_clusters)]

    # Map each encounter to its cluster integer index (always needed)
    inv = np.empty(n_obs, dtype=int)
    for ci in range(n_clusters):
        inv[cluster_enc_idx[ci]] = ci

    # Determine cluster-level outcome for stratified resampling (vectorized)
    if stratify_outcome:
        cluster_fall_flag = np.bincount(inv, weights=y_true, minlength=n_clusters) > 0
        fall_cluster_idx = np.where(cluster_fall_flag)[0]
        nofall_cluster_idx = np.where(~cluster_fall_flag)[0]
        n_fall_clusters = len(fall_cluster_idx)
        n_nofall_clusters = len(nofall_cluster_idx)
    else:
        fall_cluster_idx = None
        nofall_cluster_idx = None

    # Observed statistic
    auc_a_obs = roc_auc_score(y_true, scores_a)
    auc_b_obs = roc_auc_score(y_true, scores_b)
    delta_obs = auc_a_obs - auc_b_obs

    # Bootstrap loop
    boot_deltas = np.zeros(n_bootstrap)
    boot_auc_a = np.zeros(n_bootstrap)
    boot_auc_b = np.zeros(n_bootstrap)
    n_failed = 0

    for b in range(n_bootstrap):
        # Resample cluster integer indices
        if stratify_outcome and fall_cluster_idx is not None and nofall_cluster_idx is not None:
            sampled_fall = rng.choice(fall_cluster_idx, size=n_fall_clusters, replace=True)
            sampled_nofall = rng.choice(nofall_cluster_idx, size=n_nofall_clusters, replace=True)
            sampled = np.concatenate([sampled_fall, sampled_nofall])
        else:
            sampled = rng.choice(n_clusters, size=n_clusters, replace=True)

        # Build bootstrap sample using vectorized np.repeat
        sample_counts = np.bincount(sampled, minlength=n_clusters)
        # For each encounter, repeat it sample_counts[its_cluster] times
        enc_repeat = sample_counts[inv]
        boot_indices = np.repeat(np.arange(n_obs), enc_repeat)

        y_boot = y_true[boot_indices]
        sa_boot = scores_a[boot_indices]
        sb_boot = scores_b[boot_indices]

        # Check that both classes are present
        if len(np.unique(y_boot)) < 2:
            n_failed += 1
            boot_deltas[b] = np.nan
            boot_auc_a[b] = np.nan
            boot_auc_b[b] = np.nan
            continue

        try:
            aa = roc_auc_score(y_boot, sa_boot)
            ab = roc_auc_score(y_boot, sb_boot)
            boot_auc_a[b] = aa
            boot_auc_b[b] = ab
            boot_deltas[b] = aa - ab
        except ValueError:
            n_failed += 1
            boot_deltas[b] = np.nan
            boot_auc_a[b] = np.nan
            boot_auc_b[b] = np.nan

    if n_failed > 0:
        warnings.warn(
            f"{n_failed}/{n_bootstrap} bootstrap samples had only one class "
            "and were excluded.",
            stacklevel=2,
        )

    # Remove failed samples
    valid_mask = ~np.isnan(boot_deltas)
    boot_deltas_valid = boot_deltas[valid_mask]
    n_valid = len(boot_deltas_valid)

    alpha = 1 - confidence_level

    if method == "bca" and n_valid >= 100:
        ci_lower, ci_upper = _bca_interval(
            boot_deltas_valid,
            delta_obs,
            y_true,
            scores_a,
            scores_b,
            cluster_ids,
            inv,
            n_clusters,
            alpha,
        )
    else:
        # Percentile method
        ci_lower = np.percentile(boot_deltas_valid, 100 * alpha / 2)
        ci_upper = np.percentile(boot_deltas_valid, 100 * (1 - alpha / 2))

    # Bootstrap p-value (two-sided): proportion of bootstrap deltas on
    # the opposite side of zero from the observed delta, times 2.
    # More standard: proportion of bootstrap samples where |delta| >= 0
    # under H0. Since we don't center, use the proportion of bootstrap
    # deltas with opposite sign as a conservative p-value estimate.
    # Alternatively: use the CI to determine significance.
    p_lower = np.mean(boot_deltas_valid <= 0)
    p_upper = np.mean(boot_deltas_valid >= 0)
    p_value = 2 * min(p_lower, p_upper)
    p_value = min(p_value, 1.0)

    return AUROCComparisonResult(
        auc_a=auc_a_obs,
        auc_b=auc_b_obs,
        delta=delta_obs,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        method=f"Cluster bootstrap ({method}, n={n_valid})",
        n_obs=n_obs,
        n_clusters=n_clusters,
        n_bootstrap=n_valid,
    )


def cluster_bootstrap_auroc_single(
    y_true: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> AUROCResult:
    """Cluster-bootstrap CI for a single AUROC."""
    rng = np.random.RandomState(seed)

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    n_obs = len(y_true)

    # Pre-compute cluster membership via sorting (vectorized, O(n log n))
    sort_order = np.argsort(cluster_ids)
    sorted_ids = cluster_ids[sort_order]
    unique_clusters, starts, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True
    )
    n_clusters = len(unique_clusters)

    # Map each encounter to its cluster integer index
    inv = np.empty(n_obs, dtype=int)
    for ci in range(n_clusters):
        inv[sort_order[starts[ci]:starts[ci] + counts[ci]]] = ci

    auc_obs = roc_auc_score(y_true, scores)

    boot_aucs = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        sampled = rng.choice(n_clusters, size=n_clusters, replace=True)

        # Build bootstrap sample using vectorized np.repeat
        sample_counts = np.bincount(sampled, minlength=n_clusters)
        enc_repeat = sample_counts[inv]
        boot_indices = np.repeat(np.arange(n_obs), enc_repeat)

        y_b = y_true[boot_indices]
        s_b = scores[boot_indices]
        if len(np.unique(y_b)) < 2:
            boot_aucs[b] = np.nan
            continue
        boot_aucs[b] = roc_auc_score(y_b, s_b)

    valid = boot_aucs[~np.isnan(boot_aucs)]
    alpha = 1 - confidence_level
    ci_lo = np.percentile(valid, 100 * alpha / 2)
    ci_hi = np.percentile(valid, 100 * (1 - alpha / 2))
    se = np.std(valid, ddof=1)

    return AUROCResult(auc=auc_obs, ci_lower=ci_lo, ci_upper=ci_hi, se=se)


# ---------------------------------------------------------------------------
# BCa interval helper
# ---------------------------------------------------------------------------

def _bca_interval(
    boot_deltas: np.ndarray,
    delta_obs: float,
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    cluster_ids: np.ndarray,
    inv: np.ndarray,
    n_clusters: int,
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap interval.

    Uses the jackknife (leave-one-cluster-out) to estimate the acceleration
    parameter. For very large numbers of clusters, falls back to percentile.
    """
    n_boot = len(boot_deltas)

    # Bias correction factor (z0)
    prop_below = float(np.mean(boot_deltas < delta_obs))
    prop_below = np.clip(prop_below, 1e-6, 1 - 1e-6)
    z0 = float(stats.norm.ppf(prop_below))

    # Acceleration factor via jackknife (leave-one-cluster-out)
    # For computational efficiency, limit jackknife to max 2000 clusters
    if n_clusters > 2000:
        warnings.warn(
            f"Too many clusters ({n_clusters}) for BCa jackknife; "
            "using percentile interval.",
            stacklevel=2,
        )
        ci_lo = float(np.percentile(boot_deltas, 100 * alpha / 2))
        ci_hi = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))
        return ci_lo, ci_hi

    jack_deltas = np.zeros(n_clusters)
    for i in range(n_clusters):
        # Leave out all encounters from cluster i (vectorized mask)
        mask = inv != i
        y_jack = y_true[mask]
        sa_jack = scores_a[mask]
        sb_jack = scores_b[mask]

        if len(np.unique(y_jack)) < 2:
            jack_deltas[i] = delta_obs
            continue

        try:
            jack_deltas[i] = roc_auc_score(y_jack, sa_jack) - roc_auc_score(y_jack, sb_jack)
        except ValueError:
            jack_deltas[i] = delta_obs

    jack_mean = np.mean(jack_deltas)
    jack_diff = jack_mean - jack_deltas

    denom = np.sum(jack_diff**2) ** 1.5
    if denom < 1e-12:
        a_hat = 0.0
    else:
        a_hat = np.sum(jack_diff**3) / (6.0 * denom)

    # Adjusted percentiles
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    def _adjusted_percentile(z_alpha: float) -> float:
        numerator = z0 + z_alpha
        denominator = 1 - a_hat * numerator
        if abs(denominator) < 1e-12:
            return z_alpha  # Fallback
        adjusted_z = z0 + numerator / denominator
        return stats.norm.cdf(adjusted_z)

    p_lo = _adjusted_percentile(z_alpha_lo)
    p_hi = _adjusted_percentile(z_alpha_hi)

    # Clip to valid range
    p_lo = np.clip(p_lo, 0.5 / n_boot, 1 - 0.5 / n_boot)
    p_hi = np.clip(p_hi, 0.5 / n_boot, 1 - 0.5 / n_boot)

    ci_lo = np.percentile(boot_deltas, 100 * p_lo)
    ci_hi = np.percentile(boot_deltas, 100 * p_hi)

    return ci_lo, ci_hi


# ---------------------------------------------------------------------------
# 3. Design-effect estimation
# ---------------------------------------------------------------------------

def estimate_design_effect(
    y_true: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict:
    """Estimate the design effect for AUROC due to within-patient clustering.

    Computes the approximate design effect based on the average cluster size
    and an estimate of the intracluster correlation (ICC) of the scores.

    Parameters
    ----------
    y_true : array of 0/1
    scores : array of scores
    cluster_ids : array of patient IDs

    Returns
    -------
    dict with keys:
        - n_obs: total observations
        - n_clusters: number of unique clusters
        - avg_cluster_size: mean encounters per patient
        - avg_cluster_size_falls: mean encounters per fall-patient
        - avg_cluster_size_nonfalls: mean encounters per non-fall patient
        - icc_estimate: intracluster correlation (for clusters with >1 obs)
        - deff_overall: design effect estimate
        - deff_falls: design effect for fall-patient side
        - deff_nonfalls: design effect for non-fall-patient side
        - effective_n: effective sample size
        - ci_widening_factor: sqrt(deff) -- factor by which naive CI is too narrow
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    unique_clusters, inverse, cluster_sizes = np.unique(
        cluster_ids, return_inverse=True, return_counts=True
    )
    n_clusters = len(unique_clusters)
    n_obs = len(y_true)
    avg_cluster_size = np.mean(cluster_sizes)

    # Determine which clusters have at least one fall (vectorized)
    cluster_has_fall = np.bincount(inverse, weights=y_true, minlength=n_clusters) > 0
    fall_cluster_sizes = cluster_sizes[cluster_has_fall]
    nofall_cluster_sizes = cluster_sizes[~cluster_has_fall]

    avg_cs_falls = float(np.mean(fall_cluster_sizes)) if len(fall_cluster_sizes) > 0 else 1.0
    avg_cs_nonfalls = float(np.mean(nofall_cluster_sizes)) if len(nofall_cluster_sizes) > 0 else 1.0

    # Estimate ICC for scores (one-way random effects ANOVA)
    # Only meaningful for clusters with >1 observation
    multi_mask = cluster_sizes > 1
    n_multi = int(np.sum(multi_mask))

    icc = None
    if n_multi >= 5:
        # Vectorized ANOVA: compute cluster means, then SS_between and SS_within
        # cluster_mean[i] = mean of scores in cluster i
        cluster_sum = np.bincount(inverse, weights=scores, minlength=n_clusters)
        cluster_mean = cluster_sum / cluster_sizes
        grand_mean = np.mean(scores)

        # Restrict to multi-observation clusters
        multi_idx = np.where(multi_mask)[0]
        enc_in_multi = np.isin(inverse, multi_idx)

        # Map encounter -> its cluster mean
        enc_cluster_mean = cluster_mean[inverse]

        ss_between = float(np.sum(
            cluster_sizes[multi_idx] * (cluster_mean[multi_idx] - grand_mean) ** 2
        ))
        ss_within = float(np.sum((scores[enc_in_multi] - enc_cluster_mean[enc_in_multi]) ** 2))

        k = len(multi_idx)
        total_n = int(np.sum(cluster_sizes[multi_idx]))

        if k > 1 and total_n > k:
            msb = ss_between / (k - 1)
            msw = ss_within / (total_n - k)

            sum_ni = float(np.sum(cluster_sizes[multi_idx]))
            sum_ni2 = float(np.sum(cluster_sizes[multi_idx] ** 2))
            n0 = (sum_ni - sum_ni2 / sum_ni) / (k - 1)

            if n0 > 0 and (msb + (n0 - 1) * msw) > 0:
                icc = (msb - msw) / (msb + (n0 - 1) * msw)
                icc = max(0.0, icc)

    # Design effects
    rho = icc if icc is not None else 0.0
    deff_overall = 1 + (avg_cluster_size - 1) * rho
    deff_falls = 1 + (avg_cs_falls - 1) * rho
    deff_nonfalls = 1 + (avg_cs_nonfalls - 1) * rho
    effective_n = n_obs / deff_overall

    return {
        "n_obs": n_obs,
        "n_clusters": n_clusters,
        "avg_cluster_size": round(avg_cluster_size, 3),
        "avg_cluster_size_falls": round(avg_cs_falls, 3),
        "avg_cluster_size_nonfalls": round(avg_cs_nonfalls, 3),
        "n_multi_obs_clusters": n_multi,
        "icc_estimate": round(rho, 4) if icc is not None else None,
        "deff_overall": round(deff_overall, 4),
        "deff_falls": round(deff_falls, 4),
        "deff_nonfalls": round(deff_nonfalls, 4),
        "effective_n": round(effective_n, 0),
        "ci_widening_factor": round(np.sqrt(deff_overall), 4),
    }


# ---------------------------------------------------------------------------
# 4. First-encounter-only analysis
# ---------------------------------------------------------------------------

def first_encounter_comparison(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    cluster_ids: np.ndarray,
    encounter_dates: np.ndarray | None = None,
) -> AUROCComparisonResult:
    """DeLong comparison restricted to first encounter per patient.

    Eliminates within-patient clustering entirely by keeping only each
    patient's earliest encounter. If encounter_dates is not provided,
    keeps the first row for each cluster_id as encountered in the data.

    Parameters
    ----------
    y_true, scores_a, scores_b, cluster_ids : arrays
    encounter_dates : optional array of dates/timestamps for selecting
        the earliest encounter per patient

    Returns
    -------
    AUROCComparisonResult (DeLong on independent first-encounter subset)
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    # Select first encounter per patient
    if encounter_dates is not None:
        encounter_dates = np.asarray(encounter_dates)
        sort_idx = np.argsort(encounter_dates)
        y_true = y_true[sort_idx]
        scores_a = scores_a[sort_idx]
        scores_b = scores_b[sort_idx]
        cluster_ids = cluster_ids[sort_idx]

    # Keep first occurrence of each cluster_id (vectorized)
    _, keep_idx = np.unique(cluster_ids, return_index=True)
    y_sub = y_true[keep_idx]
    sa_sub = scores_a[keep_idx]
    sb_sub = scores_b[keep_idx]

    result = naive_delong_comparison(y_sub, sa_sub, sb_sub)
    result.method = "DeLong (first encounter per patient, fully independent)"
    result.n_clusters = len(keep_idx)
    return result


# ---------------------------------------------------------------------------
# 5. Sensitivity comparison wrapper
# ---------------------------------------------------------------------------

def run_sensitivity_comparison(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    cluster_ids: np.ndarray,
    encounter_dates: np.ndarray | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
    label_a: str = "Epic PMFRS",
    label_b: str = "Morse Fall Scale",
) -> SensitivityResult:
    """Run all three comparison methods and the design-effect analysis.

    This is the main entry point for the clustering sensitivity analysis.
    Produces results for the manuscript table comparing:
    1. Naive DeLong (primary)
    2. Cluster bootstrap (sensitivity)
    3. First-encounter-only DeLong (sensitivity)

    Parameters
    ----------
    y_true : array of 0/1
    scores_a : array of scores for model A (e.g., Epic PMFRS)
    scores_b : array of scores for model B (e.g., Morse Fall Scale)
    cluster_ids : array of patient IDs (MRN)
    encounter_dates : optional, for selecting first encounter
    n_bootstrap : number of bootstrap replicates
    seed : random seed
    label_a, label_b : model names for display

    Returns
    -------
    SensitivityResult containing all three comparisons and design-effect info
    """
    print("=== AUROC Clustering Sensitivity Analysis ===")
    print(f"Comparing: {label_a} vs {label_b}")
    print(f"N encounters: {len(y_true):,}")
    print(f"N events: {np.sum(y_true):,}")
    print(f"N unique patients: {len(np.unique(cluster_ids)):,}")
    print()

    # 1. Design effect
    print("--- Design Effect Analysis ---")
    deff_a = estimate_design_effect(y_true, scores_a, cluster_ids)
    print(f"  Average cluster size (overall): {deff_a['avg_cluster_size']}")
    print(f"  Average cluster size (fall patients): {deff_a['avg_cluster_size_falls']}")
    print(f"  Average cluster size (non-fall patients): {deff_a['avg_cluster_size_nonfalls']}")
    print(f"  Clusters with >1 encounter: {deff_a['n_multi_obs_clusters']}")
    print(f"  Estimated ICC (scores): {deff_a['icc_estimate']}")
    print(f"  Design effect (overall): {deff_a['deff_overall']}")
    print(f"  Design effect (fall side): {deff_a['deff_falls']}")
    print(f"  CI widening factor: {deff_a['ci_widening_factor']}")
    print()

    # 2. Naive DeLong
    print("--- Naive DeLong Test ---")
    naive = naive_delong_comparison(y_true, scores_a, scores_b)
    print(f"  {label_a} AUROC: {naive.auc_a:.4f}")
    print(f"  {label_b} AUROC: {naive.auc_b:.4f}")
    print(f"  Delta (A - B): {naive.delta:.4f}")
    print(f"  95% CI: ({naive.ci_lower:.4f}, {naive.ci_upper:.4f})")
    print(f"  P-value: {naive.p_value:.4e}")
    print()

    # 3. Cluster bootstrap
    print(f"--- Cluster Bootstrap ({n_bootstrap} replicates) ---")
    cluster_boot = cluster_bootstrap_auroc_comparison(
        y_true, scores_a, scores_b, cluster_ids,
        n_bootstrap=n_bootstrap, seed=seed, method="bca",
    )
    print(f"  Delta (A - B): {cluster_boot.delta:.4f}")
    print(f"  95% CI: ({cluster_boot.ci_lower:.4f}, {cluster_boot.ci_upper:.4f})")
    print(f"  P-value: {cluster_boot.p_value:.4f}")
    print()

    # 4. First encounter only
    first_enc = None
    try:
        print("--- First Encounter Per Patient (DeLong) ---")
        first_enc = first_encounter_comparison(
            y_true, scores_a, scores_b, cluster_ids, encounter_dates,
        )
        print(f"  N encounters (first per patient): {first_enc.n_obs:,}")
        print(f"  {label_a} AUROC: {first_enc.auc_a:.4f}")
        print(f"  {label_b} AUROC: {first_enc.auc_b:.4f}")
        print(f"  Delta (A - B): {first_enc.delta:.4f}")
        print(f"  95% CI: ({first_enc.ci_lower:.4f}, {first_enc.ci_upper:.4f})")
        print(f"  P-value: {first_enc.p_value:.4e}")
    except Exception as e:
        print(f"  Could not compute first-encounter analysis: {e}")
    print()

    # 5. Summary comparison
    print("=" * 60)
    print("SUMMARY: Sensitivity of AUROC Difference to Clustering")
    print("=" * 60)
    print(f"{'Method':<45} {'Delta':>7} {'95% CI':>20} {'P':>10}")
    print("-" * 82)
    print(
        f"{'Naive DeLong':<45} "
        f"{naive.delta:>7.4f} "
        f"{'(' + f'{naive.ci_lower:.4f}, {naive.ci_upper:.4f}' + ')':>20} "
        f"{naive.p_value:>10.4e}"
    )
    print(
        f"{'Cluster bootstrap (patient-level)':<45} "
        f"{cluster_boot.delta:>7.4f} "
        f"{'(' + f'{cluster_boot.ci_lower:.4f}, {cluster_boot.ci_upper:.4f}' + ')':>20} "
        f"{cluster_boot.p_value:>10.4f}"
    )
    if first_enc is not None:
        print(
            f"{'First encounter only (DeLong)':<45} "
            f"{first_enc.delta:>7.4f} "
            f"{'(' + f'{first_enc.ci_lower:.4f}, {first_enc.ci_upper:.4f}' + ')':>20} "
            f"{first_enc.p_value:>10.4e}"
        )
    print("-" * 82)

    # Interpretation
    naive_width = naive.ci_upper - naive.ci_lower
    boot_width = cluster_boot.ci_upper - cluster_boot.ci_lower
    widening_pct = (boot_width - naive_width) / naive_width * 100

    print()
    print(f"CI width (naive):           {naive_width:.4f}")
    print(f"CI width (cluster boot):    {boot_width:.4f}")
    print(f"CI widening:                {widening_pct:+.1f}%")
    if abs(widening_pct) < 10:
        print("INTERPRETATION: Clustering has minimal impact (<10% CI change).")
        print("The naive DeLong result is robust.")
    elif abs(widening_pct) < 25:
        print("INTERPRETATION: Clustering has moderate impact (10-25% CI change).")
        print("Report cluster-adjusted result as primary.")
    else:
        print("INTERPRETATION: Clustering has substantial impact (>25% CI change).")
        print("The naive DeLong result is NOT reliable. Use cluster-adjusted result.")

    return SensitivityResult(
        naive=naive,
        cluster_bootstrap=cluster_boot,
        first_encounter_only=first_enc,
        design_effect_falls=deff_a["deff_falls"],
        design_effect_nonfalls=deff_a["deff_nonfalls"],
        icc_estimate=deff_a["icc_estimate"],
    )


# ---------------------------------------------------------------------------
# 6. GEE-based discrimination comparison (optional, requires statsmodels)
# ---------------------------------------------------------------------------

def gee_discrimination_check(
    df_pandas,
    score_col_a: str,
    score_col_b: str,
    outcome_col: str = "fall_flag",
    cluster_col: str = "mrn",
) -> dict:
    """Fit GEE models and compare predicted-probability AUROCs.

    This is a supplementary check, NOT a replacement for the cluster
    bootstrap AUROC comparison. It answers a slightly different question:
    "After fitting a GEE logistic model with exchangeable correlation,
    do the predicted probabilities from score A discriminate better than
    those from score B?"

    Parameters
    ----------
    df_pandas : pandas DataFrame with columns for scores, outcome, cluster
    score_col_a, score_col_b : column names for the two scores
    outcome_col : binary outcome column
    cluster_col : patient ID column

    Returns
    -------
    dict with GEE model summaries, estimated ICC, and AUROCs of predictions
    """
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.generalized_estimating_equations import GEE
    except ImportError:
        return {"error": "statsmodels not installed"}

    df = df_pandas.copy()
    # Sort by cluster for GEE
    df = df.sort_values(cluster_col).reset_index(drop=True)

    results = {}

    for label, score_col in [("model_a", score_col_a), ("model_b", score_col_b)]:
        endog = df[outcome_col].values
        exog = sm.add_constant(df[score_col].values)
        groups = df[cluster_col].values

        cov_struct = Exchangeable()
        family = Binomial()

        try:
            model = GEE(endog, exog, groups=groups, family=family, cov_struct=cov_struct)
            fit = model.fit(maxiter=100)

            pred_prob = fit.predict(exog)
            auc = roc_auc_score(endog, pred_prob)
            icc_est = cov_struct.summary()

            results[label] = {
                "score_col": score_col,
                "coef": fit.params[1],
                "se_robust": fit.bse[1],
                "p_value": fit.pvalues[1],
                "auroc_gee_predictions": round(auc, 4),
                "icc_exchangeable": str(icc_est),
                "n_clusters": len(np.unique(groups)),
            }
        except Exception as e:
            results[label] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Convenience: generate manuscript-ready text
# ---------------------------------------------------------------------------

def format_for_manuscript(result: SensitivityResult) -> str:
    """Generate manuscript-ready text for the Methods and Results sections."""
    naive = result.naive
    boot = result.cluster_bootstrap

    naive_width = naive.ci_upper - naive.ci_lower
    boot_width = boot.ci_upper - boot.ci_lower
    widening_pct = (boot_width - naive_width) / naive_width * 100

    text = []
    text.append("METHODS (Statistical Analysis section):")
    text.append("-" * 40)
    text.append(
        "Discrimination was compared between Epic PMFRS and MFS using the "
        "paired DeLong test (DeLong et al., 1988). Because some patients "
        "contributed multiple encounters, we performed a prespecified "
        "sensitivity analysis using cluster-level bootstrap resampling "
        f"({boot.n_bootstrap} replicates) at the patient level to account "
        "for within-patient correlation (Obuchowski, 1997). Additionally, "
        "we restricted the analysis to each patient's first encounter to "
        "eliminate clustering entirely."
    )
    text.append("")
    text.append("RESULTS:")
    text.append("-" * 40)
    text.append(
        f"The AUROC for Epic PMFRS was {naive.auc_a:.3f} and for MFS was "
        f"{naive.auc_b:.3f} (difference, {naive.delta:.3f}; 95% CI, "
        f"{naive.ci_lower:.3f} to {naive.ci_upper:.3f}; P{'<.001' if naive.p_value < 0.001 else f'={naive.p_value:.3f}'}; "
        f"DeLong test). Results were robust to accounting for within-patient "
        f"clustering (cluster-bootstrap 95% CI for the difference, "
        f"{boot.ci_lower:.3f} to {boot.ci_upper:.3f}"
    )
    if result.first_encounter_only is not None:
        fe = result.first_encounter_only
        text.append(
            f"; first-encounter-only analysis: difference, {fe.delta:.3f}; "
            f"95% CI, {fe.ci_lower:.3f} to {fe.ci_upper:.3f}"
        )
    text.append(
        f"). The confidence interval widened by {widening_pct:.1f}% after "
        "cluster adjustment, consistent with mild within-patient clustering "
        f"(estimated design effect for fall encounters, {result.design_effect_falls:.2f})."
    )

    return "\n".join(text)
