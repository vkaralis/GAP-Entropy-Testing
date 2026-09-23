"""Core implementation of GET-V, GET-Plus, and GET-Omega."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class TestResult:
    """Statistic and upper-tail permutation p-value for one GET procedure."""

    statistic: float
    pvalue: float


@dataclass(frozen=True)
class GETComponents:
    """Observed unstandardized components used by the GET procedures."""

    location: float
    spacing_entropy: float
    energy: float


@dataclass(frozen=True)
class GETResult:
    """Results returned by :func:`get_test`."""

    get_v: TestResult
    get_plus: TestResult
    get_omega: TestResult
    components: GETComponents
    permutations: int
    seed: int | None


def _as_sample(values: ArrayLike | Iterable[float], name: str) -> NDArray[np.float64]:
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sample.")
    if sample.size < 5:
        raise ValueError(f"{name} must contain at least five observations.")
    if not np.all(np.isfinite(sample)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    if np.unique(sample).size != sample.size:
        raise ValueError(
            f"{name} contains ties. GET spacing inference requires continuous, "
            "tie-free observations; do not add arbitrary jitter."
        )
    return sample


def _vasicek_entropy(values: NDArray[np.float64]) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    n = ordered.size
    m = max(1, int(math.sqrt(n)))
    if 2 * m >= n:
        m = max(1, (n - 1) // 2)

    indices = np.arange(n)
    left = np.maximum(indices - m, 0)
    right = np.minimum(indices + m, n - 1)
    spacings = np.maximum(ordered[right] - ordered[left], 1e-12)
    return float(np.mean(np.log((n / (2.0 * m)) * spacings)))


def spacing_entropy(sample: ArrayLike | Iterable[float]) -> float:
    """Return the entropy estimate of log1p-transformed normalized gaps.

    The statistic is invariant to translation and positive rescaling. Exact
    ties are rejected because GET spacing inference assumes continuous data.
    """

    values = _as_sample(sample, "sample")
    gaps = np.diff(np.sort(values))
    median_gap = float(np.median(gaps))
    if median_gap <= 0.0:
        raise ValueError("The median adjacent gap must be positive.")
    transformed = np.log1p(gaps / median_gap)
    return _vasicek_entropy(transformed)


def _energy_statistic(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    x_column = x[:, None]
    y_column = y[:, None]
    between = cdist(x_column, y_column, metric="euclidean").mean()
    within_x = cdist(x_column, x_column, metric="euclidean").mean()
    within_y = cdist(y_column, y_column, metric="euclidean").mean()
    return float(2.0 * between - within_x - within_y)


def _components(x: NDArray[np.float64], y: NDArray[np.float64]) -> GETComponents:
    return GETComponents(
        location=abs(float(np.median(x) - np.median(y))),
        spacing_entropy=abs(spacing_entropy(x) - spacing_entropy(y)),
        energy=_energy_statistic(x, y),
    )


def _robust_standardize(
    observed: float, permuted: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    center = float(np.median(permuted))
    spread = float(stats.median_abs_deviation(permuted, scale="normal"))
    if spread <= 1e-12:
        spread = float(np.std(permuted, ddof=1)) if permuted.size > 1 else 1.0
    if spread <= 1e-12:
        spread = 1.0
    return float((observed - center) / spread), (permuted - center) / spread


def _permutation_pvalue(observed: float, permuted: NDArray[np.float64]) -> float:
    return float((np.count_nonzero(permuted >= observed) + 1) / (permuted.size + 1))


def get_test(
    x: ArrayLike | Iterable[float],
    y: ArrayLike | Iterable[float],
    *,
    permutations: int = 999,
    seed: int | None = None,
) -> GETResult:
    """Run GET-V, GET-Plus, and GET-Omega on two independent samples.

    Parameters
    ----------
    x, y:
        One-dimensional, finite, tie-free numeric samples with at least five
        observations each.
    permutations:
        Number of individual-label permutations.
    seed:
        Seed for NumPy's random generator. Supply a seed for reproducibility.

    Notes
    -----
    The permutation p-values require a common continuous i.i.d.
    exchangeability null. GET-V compares spacing entropy. GET-Plus takes the
    maximum of robustly standardized spacing and median-location components.
    GET-Omega additionally includes the energy-distance component.
    """

    sample_x = _as_sample(x, "x")
    sample_y = _as_sample(y, "y")
    combined = np.concatenate([sample_x, sample_y])
    if np.unique(combined).size != combined.size:
        raise ValueError(
            "The pooled samples contain ties. All observations across x and y "
            "must be unique for GET spacing permutation inference."
        )
    if isinstance(permutations, bool) or not isinstance(permutations, (int, np.integer)):
        raise TypeError("permutations must be an integer.")
    if permutations < 1:
        raise ValueError("permutations must be at least 1.")

    observed = _components(sample_x, sample_y)
    n_x = sample_x.size
    rng = np.random.default_rng(seed)

    perm_location = np.empty(permutations)
    perm_spacing = np.empty(permutations)
    perm_energy = np.empty(permutations)

    for index in range(permutations):
        shuffled = rng.permutation(combined)
        permuted = _components(shuffled[:n_x], shuffled[n_x:])
        perm_location[index] = permuted.location
        perm_spacing[index] = permuted.spacing_entropy
        perm_energy[index] = permuted.energy

    z_location, z_location_perm = _robust_standardize(
        observed.location, perm_location
    )
    z_spacing, z_spacing_perm = _robust_standardize(
        observed.spacing_entropy, perm_spacing
    )
    z_energy, z_energy_perm = _robust_standardize(observed.energy, perm_energy)

    plus_observed = max(z_location, z_spacing)
    plus_permuted = np.maximum(z_location_perm, z_spacing_perm)
    omega_observed = max(z_location, z_spacing, z_energy)
    omega_permuted = np.maximum.reduce(
        [z_location_perm, z_spacing_perm, z_energy_perm]
    )

    return GETResult(
        get_v=TestResult(
            statistic=observed.spacing_entropy,
            pvalue=_permutation_pvalue(observed.spacing_entropy, perm_spacing),
        ),
        get_plus=TestResult(
            statistic=plus_observed,
            pvalue=_permutation_pvalue(plus_observed, plus_permuted),
        ),
        get_omega=TestResult(
            statistic=omega_observed,
            pvalue=_permutation_pvalue(omega_observed, omega_permuted),
        ),
        components=observed,
        permutations=int(permutations),
        seed=seed,
    )
