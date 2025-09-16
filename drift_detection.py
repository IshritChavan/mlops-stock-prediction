"""
Drift detection utilities using KL divergence and Population Stability Index (PSI).

Data drift occurs when the statistical properties of features change over time.
KL divergence quantifies the dissimilarity between two probability distributions【180918425037921†L82-L92】,
whereas PSI measures how much a distribution has shifted relative to a baseline and is
commonly used in finance【596254147784958†L162-L169】.  Higher values indicate more drift.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def _histogram(a: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram counts and bin edges for array `a` with a fixed number of bins.
    Excludes NaN values.
    """
    a = a[~np.isnan(a)]
    counts, edges = np.histogram(a, bins=bins, density=False)
    return counts.astype(float), edges


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Compute the Kullback–Leibler divergence between two probability distributions.

    Parameters
    ----------
    p : np.ndarray
        Probabilities of the reference distribution.
    q : np.ndarray
        Probabilities of the comparison distribution.
    epsilon : float
        Small constant to avoid division by zero or log of zero.

    Returns
    -------
    float
        KL divergence (non‑negative).  Values closer to 0 indicate similar distributions.
    """
    p = p + epsilon
    q = q + epsilon
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)
    return float(np.sum(p_norm * np.log(p_norm / q_norm)))


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute the Population Stability Index (PSI) between two samples.

    The PSI sums over bins the difference between expected and actual proportions multiplied
    by the log of their ratio.  Higher PSI values indicate larger shifts【596254147784958†L162-L169】.

    Parameters
    ----------
    expected : np.ndarray
        Baseline feature values.
    actual : np.ndarray
        Current feature values.
    bins : int
        Number of histogram bins.

    Returns
    -------
    float
        PSI value.
    """
    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    # Determine common bin edges based on expected distribution
    counts_expected, bin_edges = np.histogram(expected, bins=bins)
    counts_actual, _ = np.histogram(actual, bins=bin_edges)
    # Convert counts to proportions
    expected_perc = counts_expected / max(np.sum(counts_expected), 1)
    actual_perc = counts_actual / max(np.sum(counts_actual), 1)
    psi_values = []
    for e, a in zip(expected_perc, actual_perc):
        if e == 0 or a == 0:
            continue
        psi_values.append((a - e) * np.log(a / e))
    return float(np.sum(psi_values))


def detect_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: Dict[str, str],
    bins: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate drift metrics for each numeric feature.

    Parameters
    ----------
    baseline : pd.DataFrame
        Historical baseline dataset.
    current : pd.DataFrame
        New dataset to compare against baseline.
    numeric_features : dict
        Mapping of feature names to more descriptive names (unused but kept for
        future enhancements).
    bins : int
        Number of bins for PSI/​KL calculations.

    Returns
    -------
    dict
        Dictionary where each key is a feature and the value is another dict
        containing the PSI and KL divergence.
    """
    metrics: Dict[str, Dict[str, float]] = {}
    for feature in numeric_features:
        expected_values = baseline[feature].values
        actual_values = current[feature].values
        # Histogram for KL divergence using same bin edges
        p_counts, edges = _histogram(expected_values, bins)
        q_counts, _ = np.histogram(actual_values, bins=edges)
        kl = kl_divergence(p_counts, q_counts)
        psi = population_stability_index(expected_values, actual_values, bins=bins)
        metrics[feature] = {"kl_divergence": kl, "psi": psi}
    return metrics