"""
Smoke test for cluster_auroc.py with simulated data mimicking
the rushfalla study structure.

Run: python utils/test_cluster_auroc.py
"""
import sys
import numpy as np

# Ensure the parent directory is on the path
sys.path.insert(0, ".")

from utils.cluster_auroc import (
    naive_delong_comparison,
    naive_delong_single,
    cluster_bootstrap_auroc_comparison,
    cluster_bootstrap_auroc_single,
    estimate_design_effect,
    first_encounter_comparison,
    run_sensitivity_comparison,
    format_for_manuscript,
)


def simulate_rushfalla_like_data(
    n_fall_patients: int = 778,
    n_nofall_patients: int = 10000,
    extra_fall_encounters: int = 83,
    mean_nofall_encounters: float = 1.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate data resembling the rushfalla study.

    Returns (y_true, epic_scores, morse_scores, patient_ids).
    """
    rng = np.random.RandomState(seed)

    y_list = []
    epic_list = []
    morse_list = []
    pid_list = []

    # Fall patients: most have 1 encounter, ~83 have 2
    for i in range(n_fall_patients):
        n_enc = 2 if i < extra_fall_encounters else 1
        base_epic = rng.normal(55, 15)
        base_morse = rng.normal(50, 12)
        for _ in range(n_enc):
            y_list.append(1)
            epic_list.append(base_epic + rng.normal(0, 5))
            morse_list.append(base_morse + rng.normal(0, 4))
            pid_list.append(f"FALL_{i:05d}")

    # Non-fall patients
    for i in range(n_nofall_patients):
        n_enc = max(1, int(rng.exponential(mean_nofall_encounters - 1)) + 1)
        base_epic = rng.normal(30, 15)
        base_morse = rng.normal(25, 12)
        for _ in range(n_enc):
            y_list.append(0)
            epic_list.append(base_epic + rng.normal(0, 5))
            morse_list.append(base_morse + rng.normal(0, 4))
            pid_list.append(f"NOFALL_{i:06d}")

    return (
        np.array(y_list),
        np.clip(np.array(epic_list), 0, 100),
        np.clip(np.array(morse_list), 0, 100),
        np.array(pid_list),
    )


def main():
    print("Simulating rushfalla-like data...")
    y, epic, morse, pids = simulate_rushfalla_like_data()
    print(f"  Total encounters: {len(y):,}")
    print(f"  Falls: {np.sum(y):,}")
    print(f"  Unique patients: {len(np.unique(pids)):,}")
    print()

    # Run the full sensitivity analysis
    result = run_sensitivity_comparison(
        y_true=y,
        scores_a=epic,
        scores_b=morse,
        cluster_ids=pids,
        n_bootstrap=500,  # Reduced for speed in test
        seed=42,
        label_a="Epic PMFRS (simulated)",
        label_b="Morse Fall Scale (simulated)",
    )

    print()
    print("=" * 60)
    print("MANUSCRIPT TEXT:")
    print("=" * 60)
    print(format_for_manuscript(result))

    # Test individual functions
    print()
    print("--- Individual AUROC CIs ---")
    epic_single = naive_delong_single(y, epic)
    print(f"Epic (DeLong):           {epic_single.auc:.4f} ({epic_single.ci_lower:.4f}, {epic_single.ci_upper:.4f})")

    epic_boot = cluster_bootstrap_auroc_single(y, epic, pids, n_bootstrap=500)
    print(f"Epic (cluster boot):     {epic_boot.auc:.4f} ({epic_boot.ci_lower:.4f}, {epic_boot.ci_upper:.4f})")

    # Design effect
    print()
    deff = estimate_design_effect(y, epic, pids)
    print("--- Design Effect ---")
    for k, v in deff.items():
        print(f"  {k}: {v}")

    print()
    print("ALL TESTS PASSED.")


if __name__ == "__main__":
    main()
