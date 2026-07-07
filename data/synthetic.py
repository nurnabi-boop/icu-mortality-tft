"""
Synthetic ICU dataset for development / smoke-testing.

Generates physiologically plausible vital-sign time series that mimic the
class-conditional distributions seen in MIMIC-III. Mortality patients show
higher heart rate, lower SpO2, and more extreme blood pressures.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


VITAL_COLS = ["hr", "spo2", "sbp", "dbp", "mbp"]
WINDOW_HOURS = 48


def _ar1(n: int, mu: float, sigma: float, phi: float = 0.85, rng=None) -> np.ndarray:
    """AR(1) process around mean mu, noise sigma, persistence phi."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.empty(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(0, sigma)
    return x


def generate_synthetic(
    n_samples: int = 3000,
    mortality_rate: float = 0.13,
    window: int = WINDOW_HOURS,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : float32 array (n_samples, window, 5)  — hr, spo2, sbp, dbp, mbp
    y : int64 array  (n_samples,)              — 0 = survived, 1 = died
    """
    rng = np.random.default_rng(seed)
    n_mort = int(n_samples * mortality_rate)
    labels = np.array([1] * n_mort + [0] * (n_samples - n_mort))
    rng.shuffle(labels)

    # (mean, std) for (survived, died) per vital
    params = {
        #       survived          died
        "hr":  ((78,  12),  (102, 18)),
        "spo2":((97,   1),   (92,  3)),
        "sbp": ((122, 18),  (100, 22)),
        "dbp": ((74,  12),   (60, 14)),
        "mbp": ((90,  14),   (75, 16)),
    }

    X = np.zeros((n_samples, window, len(VITAL_COLS)), dtype=np.float32)

    for i, label in enumerate(labels):
        for j, vital in enumerate(VITAL_COLS):
            (mu0, s0), (mu1, s1) = params[vital]
            mu, sigma = (mu1, s1) if label == 1 else (mu0, s0)

            # Add a slow drift (deterioration) for mortality patients
            drift = 0.0
            if label == 1:
                direction = -1 if vital in ("spo2", "sbp", "dbp", "mbp") else 1
                drift = direction * rng.uniform(0.05, 0.2)  # per hour

            ts = _ar1(window, mu, sigma * 0.4, phi=0.88, rng=rng)
            ts += np.arange(window) * drift

            # Random missingness (5 %) → forward-fill
            missing = rng.random(window) < 0.05
            for t in range(window):
                if missing[t] and t > 0:
                    ts[t] = ts[t - 1]

            X[i, :, j] = ts.astype(np.float32)

    # Clip to physiological ranges
    bounds = [(0, 300), (50, 100), (60, 280), (30, 180), (50, 220)]
    for j, (lo, hi) in enumerate(bounds):
        X[:, :, j] = np.clip(X[:, :, j], lo, hi)

    return X, labels.astype(np.int64)


def normalise(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std
