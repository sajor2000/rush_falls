"""Statistical metrics: DeLong AUROC, bootstrap CIs, NRI/IDI, calibration, thresholds."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from utils.constants import ALPHA, COST_PARAMS, N_BOOTSTRAP, RANDOM_SEED

# ═══════════════════════════════════════════════════════════════════════
# DeLong test — Sun & Xu (2014) fast algorithm
# Adapted from yandexdataschool/roc_comparison (MIT)
# ═══════════════════════════════════════════════════════════════════════


def _compute_midrank(x: NDArray) -> NDArray:
    """Compute midranks for tied values."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j_end = i
        while j_end < n and z[j_end] == z[i]:
            j_end += 1
        t[i:j_end] = 0.5 * (i + j_end - 1)
        i = j_end
    t2 = np.empty(n, dtype=float)
    t2[j] = t + 1
    return t2


def _fast_delong(
    predictions_sorted_transposed: NDArray,
    label_1_count: int,
) -> tuple[NDArray, NDArray]:
    """Core fast DeLong computation."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(
    ground_truth: NDArray,
    predictions_one: NDArray,
    predictions_two: NDArray,
) -> float:
    """Two-sided p-value for H0: AUC1 == AUC2 (paired DeLong test)."""
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    predictions_sorted_transposed = np.vstack(
        (predictions_one, predictions_two)
    )[:, order]
    aucs, delongcov = _fast_delong(predictions_sorted_transposed, label_1_count)
    diff = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(diff, delongcov), diff.T))
    log10_pvalue = np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
    return float(10 ** log10_pvalue[0][0])


def delong_roc_variance(
    ground_truth: NDArray,
    predictions: NDArray,
) -> tuple[float, float]:
    """Return (AUC, variance) for a single model using DeLong."""
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = _fast_delong(predictions_sorted_transposed, label_1_count)
    return float(aucs[0]), float(delongcov)


def delong_ci(
    ground_truth: NDArray,
    predictions: NDArray,
    alpha: float = ALPHA,
) -> tuple[float, float, float]:
    """Return (AUC, lower_ci, upper_ci) using DeLong variance."""
    auc, var = delong_roc_variance(ground_truth, predictions)
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(var)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


# ═══════════════════════════════════════════════════════════════════════
# Stratified Bootstrap
# ═══════════════════════════════════════════════════════════════════════


def stratified_bootstrap(
    y_true: NDArray,
    pred_a: NDArray,
    pred_b: NDArray | None = None,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    alpha: float = ALPHA,
) -> dict[str, dict[str, float]]:
    """Stratified bootstrap preserving event ratio.

    Returns dict of metric name → {estimate, ci_lower, ci_upper}.
    If pred_b is provided, also computes difference metrics.
    """
    rng = np.random.RandomState(seed)
    events_idx = np.where(y_true == 1)[0]
    nonevents_idx = np.where(y_true == 0)[0]

    boot: dict[str, list[float]] = {"auc_a": [], "auprc_a": []}
    if pred_b is not None:
        boot["auc_b"] = []
        boot["auc_diff"] = []
        boot["auprc_b"] = []
        boot["auprc_diff"] = []

    for _ in range(n_boot):
        be = rng.choice(events_idx, size=len(events_idx), replace=True)
        bn = rng.choice(nonevents_idx, size=len(nonevents_idx), replace=True)
        idx = np.concatenate([be, bn])

        y_b = y_true[idx]
        pa_b = pred_a[idx]

        auc_a = float(roc_auc_score(y_b, pa_b))
        auprc_a = float(average_precision_score(y_b, pa_b))
        boot["auc_a"].append(auc_a)
        boot["auprc_a"].append(auprc_a)

        if pred_b is not None:
            pb_b = pred_b[idx]
            auc_b = float(roc_auc_score(y_b, pb_b))
            auprc_b = float(average_precision_score(y_b, pb_b))
            boot["auc_b"].append(auc_b)
            boot["auc_diff"].append(auc_a - auc_b)
            boot["auprc_b"].append(auprc_b)
            boot["auprc_diff"].append(auprc_a - auprc_b)

    results: dict[str, dict[str, float]] = {}
    for key, vals in boot.items():
        arr = np.array(vals)
        results[key] = {
            "estimate": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
# Classification metrics at a given threshold
# ═══════════════════════════════════════════════════════════════════════


def classification_metrics_at_threshold(
    y_true: NDArray,
    y_score: NDArray,
    threshold: float,
) -> dict[str, float]:
    """Compute sens, spec, PPV, NPV, NNE at a single threshold."""
    pred_pos = y_score >= threshold
    tp = int(np.sum(pred_pos & (y_true == 1)))
    fp = int(np.sum(pred_pos & (y_true == 0)))
    fn = int(np.sum(~pred_pos & (y_true == 1)))
    tn = int(np.sum(~pred_pos & (y_true == 0)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    nne = 1 / ppv if ppv > 0 else float("inf")

    total = tp + fp + fn + tn
    flag_rate = (tp + fp) / total * 100 if total > 0 else 0.0

    return {
        "threshold": threshold,
        "sensitivity": sens,
        "specificity": spec,
        "flag_rate": flag_rate,
        "ppv": ppv,
        "npv": npv,
        "nne": nne,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ═══════════════════════════════════════════════════════════════════════
# Threshold selection methods
# ═══════════════════════════════════════════════════════════════════════


def youden_threshold(y_true: NDArray, y_score: NDArray) -> float:
    """Youden index: max(sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def closest_topleft_threshold(y_true: NDArray, y_score: NDArray) -> float:
    """Closest to (0,1) corner on ROC space."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    dist = np.sqrt((1 - tpr) ** 2 + fpr**2)
    return float(thresholds[np.argmin(dist)])


def fixed_sensitivity_threshold(
    y_true: NDArray,
    y_score: NDArray,
    target_sens: float,
) -> float:
    """Find threshold achieving at least target_sens sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = tpr >= target_sens
    if not valid.any():
        return float(thresholds[0])
    # Among thresholds meeting sensitivity, pick highest (most specific)
    idx = np.where(valid)[0]
    return float(thresholds[idx[-1]])


def value_optimizing_threshold(
    y_true: NDArray,
    y_pred_prob: NDArray,
    cost_params: dict | None = None,
    threshold_grid: NDArray | None = None,
) -> tuple[float, float, list[dict[str, float]]]:
    """Find threshold maximizing expected NMB (Parsons et al. JAMIA 2023).

    Returns (best_threshold, best_nmb, nmb_by_threshold).
    """
    cp = cost_params or COST_PARAMS
    if threshold_grid is None:
        threshold_grid = np.arange(0.001, 0.20, 0.001)

    best_nmb = -np.inf
    best_threshold = 0.5
    nmb_by_threshold: list[dict[str, float]] = []
    n = len(y_true)

    for t in threshold_grid:
        pred_pos = y_pred_prob >= t
        tp = np.sum(pred_pos & (y_true == 1))
        fp = np.sum(pred_pos & (y_true == 0))
        tn = np.sum(~pred_pos & (y_true == 0))
        fn = np.sum(~pred_pos & (y_true == 1))

        eff = cp["effectiveness_alpha"] / (
            cp["effectiveness_alpha"] + cp["effectiveness_beta"]
        )
        cost_fall = cp["cost_fall_mean"]
        cost_int = cp["cost_intervention_mean"]
        qaly = cp["qaly_loss_mean"]
        wtp = cp["wtp"]

        nmb_tp = -(cost_int + cost_fall * (1 - eff) + qaly * (1 - eff) * wtp)
        nmb_fp = -cost_int
        nmb_tn = 0.0
        nmb_fn = -(cost_fall + qaly * wtp)

        total_nmb = (tp * nmb_tp + fp * nmb_fp + tn * nmb_tn + fn * nmb_fn) / n
        nmb_by_threshold.append({"threshold": float(t), "nmb": float(total_nmb)})

        if total_nmb > best_nmb:
            best_nmb = float(total_nmb)
            best_threshold = float(t)

    return best_threshold, best_nmb, nmb_by_threshold


def extract_dca_threshold_range(
    df_dca: Any,
    model_name: str,
    treat_all_name: str = "all",
    treat_none_name: str = "none",
) -> dict[str, float | None]:
    """Extract threshold range where model net benefit > treat-all AND > treat-none.

    Parameters
    ----------
    df_dca : pandas DataFrame
        Long-format output from ``dcurves.dca()``, with columns
        ``model``, ``threshold``, ``net_benefit``.
    model_name : str
        Name of the model column to evaluate (e.g. ``"epic_prob"``).
    treat_all_name : str
        Name of the treat-all reference in the ``model`` column.
    treat_none_name : str
        Name of the treat-none reference in the ``model`` column.

    Returns
    -------
    dict with keys ``lower``, ``upper`` (probability scale), or ``None``
    if no threshold range meets both conditions.
    """
    import pandas as pd

    def _extract_nb(model: str, name: str) -> "pd.Series":
        s = df_dca[df_dca["model"] == model].copy()
        s["threshold"] = s["threshold"].round(8)
        return s.set_index("threshold")["net_benefit"].rename(name)

    nb_model = _extract_nb(model_name, "nb_model")
    nb_all = _extract_nb(treat_all_name, "nb_all")
    nb_none = _extract_nb(treat_none_name, "nb_none")

    merged = pd.concat([nb_model, nb_all, nb_none], axis=1).dropna()
    valid = merged[(merged["nb_model"] > merged["nb_all"]) & (merged["nb_model"] > merged["nb_none"])]

    if valid.empty:
        return {"lower": None, "upper": None}

    return {
        "lower": float(valid.index.min()),
        "upper": float(valid.index.max()),
    }


# ═══════════════════════════════════════════════════════════════════════
# NRI / IDI
# ═══════════════════════════════════════════════════════════════════════


def compute_nri_idi(
    y_true: NDArray,
    prob_ref: NDArray,
    prob_new: NDArray,
    threshold: float | None = None,
) -> dict[str, float | None]:
    """Compute continuous NRI (event/non-event) and IDI.

    If threshold provided, also computes category-based NRI.
    """
    events = y_true == 1
    nonevents = y_true == 0

    # Continuous NRI
    event_up = float(np.mean(prob_new[events] > prob_ref[events]))
    event_down = float(np.mean(prob_new[events] < prob_ref[events]))
    nri_events = event_up - event_down

    nonevent_down = float(np.mean(prob_new[nonevents] < prob_ref[nonevents]))
    nonevent_up = float(np.mean(prob_new[nonevents] > prob_ref[nonevents]))
    nri_nonevents = nonevent_down - nonevent_up

    nri_continuous = nri_events + nri_nonevents

    # Category-based NRI
    nri_categorical: float | None = None
    if threshold is not None:
        ref_class = (prob_ref >= threshold).astype(int)
        new_class = (prob_new >= threshold).astype(int)
        nri_events_cat = float(
            np.mean(new_class[events] > ref_class[events])
            - np.mean(new_class[events] < ref_class[events])
        )
        nri_nonevents_cat = float(
            np.mean(new_class[nonevents] < ref_class[nonevents])
            - np.mean(new_class[nonevents] > ref_class[nonevents])
        )
        nri_categorical = nri_events_cat + nri_nonevents_cat

    # IDI
    idi_events = float(np.mean(prob_new[events]) - np.mean(prob_ref[events]))
    idi_nonevents = float(np.mean(prob_new[nonevents]) - np.mean(prob_ref[nonevents]))
    idi = idi_events - idi_nonevents

    return {
        "nri_continuous": nri_continuous,
        "nri_events": nri_events,
        "nri_nonevents": nri_nonevents,
        "nri_categorical": nri_categorical,
        "idi": idi,
        "idi_events": idi_events,
        "idi_nonevents": idi_nonevents,
    }


def compute_categorical_nri(
    y_true: NDArray,
    prob_ref: NDArray,
    prob_new: NDArray,
    threshold: float,
) -> float:
    """Compute category-based NRI at a single threshold (no IDI, no continuous NRI)."""
    events = y_true == 1
    nonevents = y_true == 0
    ref_class = (prob_ref >= threshold).astype(int)
    new_class = (prob_new >= threshold).astype(int)
    nri_events_cat = float(
        np.mean(new_class[events] > ref_class[events])
        - np.mean(new_class[events] < ref_class[events])
    )
    nri_nonevents_cat = float(
        np.mean(new_class[nonevents] < ref_class[nonevents])
        - np.mean(new_class[nonevents] > ref_class[nonevents])
    )
    return nri_events_cat + nri_nonevents_cat


# ═══════════════════════════════════════════════════════════════════════
# Logistic recalibration
# ═══════════════════════════════════════════════════════════════════════


def logistic_recalibration(
    scores: NDArray,
    y_true: NDArray,
    max_iter: int = 1000,
    random_state: int = RANDOM_SEED,
) -> tuple[NDArray, Any]:
    """Recalibrate ordinal scores to probabilities via single-predictor logistic regression.

    Returns (probability_array, fitted_model). Use the fitted model to compute
    probability equivalents at specific score cutoffs::

        prob_at_cutoff = float(model.predict_proba([[cutoff]])[0, 1])
    """
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(max_iter=max_iter, random_state=random_state)
    lr.fit(scores.reshape(-1, 1), y_true)
    prob = lr.predict_proba(scores.reshape(-1, 1))[:, 1]
    return prob, lr


# ═══════════════════════════════════════════════════════════════════════
# Calibration metrics
# ═══════════════════════════════════════════════════════════════════════


def calibration_metrics(
    y_true: NDArray,
    y_pred_prob: NDArray,
    lowess_frac: float = 0.3,
) -> dict[str, float]:
    """Compute CITL, calibration slope, and ICI.

    y_pred_prob must be predicted probabilities (not ordinal scores).
    """
    import statsmodels.api as sm
    from scipy.interpolate import interp1d

    # Clip to avoid log(0)
    eps = 1e-10
    y_pred_clipped = np.clip(y_pred_prob, eps, 1 - eps)

    # CITL
    observed_prev = y_true.mean()
    logit_obs = np.log(observed_prev / (1 - observed_prev))
    logit_pred = np.log(y_pred_clipped / (1 - y_pred_clipped))
    citl = float(logit_obs - logit_pred.mean())

    # Calibration slope (Cox's method)
    x = sm.add_constant(logit_pred)
    model = sm.GLM(y_true, x, family=sm.families.Binomial())
    result = model.fit()
    cal_intercept = float(result.params[0])
    cal_slope = float(result.params[1])

    # ICI with LOWESS
    lowess_result = sm.nonparametric.lowess(
        y_true, y_pred_clipped, frac=lowess_frac, it=3, return_sorted=True
    )
    lowess_func = interp1d(
        lowess_result[:, 0],
        lowess_result[:, 1],
        bounds_error=False,
        fill_value="extrapolate",
    )
    smoothed = lowess_func(y_pred_clipped)
    ici = float(np.mean(np.abs(y_pred_clipped - smoothed)))

    return {
        "citl": citl,
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
        "ici": ici,
    }
