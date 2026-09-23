import numpy as np
import pytest

from get_testing import get_test, spacing_entropy


def test_spacing_entropy_is_translation_and_positive_scale_invariant():
    rng = np.random.default_rng(7)
    sample = rng.normal(size=40)
    expected = spacing_entropy(sample)
    assert spacing_entropy(sample + 12.5) == pytest.approx(expected, abs=1e-12)
    assert spacing_entropy(sample * 3.2) == pytest.approx(expected, abs=1e-12)


def test_get_test_is_reproducible_with_seed():
    rng = np.random.default_rng(8)
    x = rng.normal(size=25)
    y = rng.normal(0.4, size=25)
    first = get_test(x, y, permutations=39, seed=123)
    second = get_test(x, y, permutations=39, seed=123)
    assert first == second


def test_ties_are_rejected():
    x = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    y = np.array([0.1, 1.1, 2.1, 3.1, 4.1])
    with pytest.raises(ValueError, match="contains ties"):
        get_test(x, y, permutations=9, seed=1)


def test_ties_across_pooled_samples_are_rejected():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.1, 2.1, 3.1, 4.1])
    with pytest.raises(ValueError, match="pooled samples contain ties"):
        get_test(x, y, permutations=9, seed=1)


def test_invalid_sample_shape_is_rejected():
    x = np.arange(10.0).reshape(2, 5)
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        get_test(x, y, permutations=9, seed=1)
