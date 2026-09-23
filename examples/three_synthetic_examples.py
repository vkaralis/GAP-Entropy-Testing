"""Three synthetic examples illustrating the complementary GET procedures."""

from __future__ import annotations

import numpy as np

from get_testing import get_test


def spacing_regularity_example(n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Irregular uniform sample versus a near-regular lattice stress test."""

    rng = np.random.default_rng(101)
    irregular = rng.uniform(0.0, 1.0, n)
    lattice = (np.arange(n) + 0.5) / n
    near_regular = np.clip(lattice + rng.normal(0.0, 0.18 / n, n), 0.0, 1.0)
    return irregular, near_regular


def location_shift_example(n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Two normal samples differing primarily in location."""

    rng = np.random.default_rng(202)
    return rng.normal(0.0, 1.0, n), rng.normal(0.65, 1.0, n)


def scale_difference_example(n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Two centered normal samples with different scales."""

    rng = np.random.default_rng(303)
    return rng.normal(0.0, 1.0, n), rng.normal(0.0, 2.0, n)


def main() -> None:
    examples = [
        ("Spacing regularity (GET-V)", spacing_regularity_example()),
        ("Location shift (GET-Plus)", location_shift_example()),
        ("Scale difference (GET-Omega)", scale_difference_example()),
    ]

    print(f"{'Example':34} {'GET-V':>9} {'GET-Plus':>10} {'GET-Omega':>11}")
    print("-" * 68)
    for label, (x, y) in examples:
        result = get_test(x, y, permutations=999, seed=42)
        print(
            f"{label:34} "
            f"{result.get_v.pvalue:9.3f} "
            f"{result.get_plus.pvalue:10.3f} "
            f"{result.get_omega.pvalue:11.3f}"
        )

    print(
        "\nThe spacing-regularity example is a constructed stress test outside "
        "the common i.i.d. null; its permutation p-values are descriptive."
    )


if __name__ == "__main__":
    main()

