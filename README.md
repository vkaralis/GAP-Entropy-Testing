# Gap-Entropy Testing (GET)

This repository provides a compact reference implementation of the three
procedures introduced in:

> Karalis, V. D. (2026). Gap-Entropy Testing (GET): A Framework for Structural
> and Distributional Differences. *Applied Mathematics and Statistics*, 3(2),
> 19. <https://doi.org/10.53941/ams.2026.100019>

It contains the method itself, a reproducible synthetic demonstration, and
basic tests. It is not the complete reproduction archive for every simulation,
table, and figure in the article.

**Software author:** Vangelis D. Karalis

## The three procedures

- **GET-V** compares entropy estimates of transformed adjacent spacings.
- **GET-Plus** combines the spacing component with a median-location component.
- **GET-Omega** additionally incorporates energy distance as an omnibus
  distributional component.

All p-values are obtained by individual-label permutation. Their formal
interpretation requires a common continuous i.i.d. exchangeability null.
Measurements must be numeric, finite, and tie-free across the pooled samples.
The implementation rejects ties instead of applying arbitrary jitter.

## Installation

Clone the repository and install it from its root directory:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
pytest
```

## Basic use

```python
import numpy as np
from get_testing import get_test

rng = np.random.default_rng(42)
x = rng.normal(0.0, 1.0, 100)
y = rng.normal(0.5, 1.0, 100)

result = get_test(x, y, permutations=999, seed=42)

print(result.get_v.pvalue)
print(result.get_plus.pvalue)
print(result.get_omega.pvalue)
print(result.components)
```

Increase `permutations` for final analyses. A fixed `seed` makes the Monte Carlo
permutation results reproducible.

## Three synthetic examples

Run:

```bash
python examples/three_synthetic_examples.py
```

The examples are deliberately complementary:

1. **Spacing regularity:** an i.i.d. uniform sample is compared with a
   near-regular lattice. GET-V detects the large gap-entropy contrast while
   energy distance is nearly unchanged. This is a constructed spacing stress
   test outside the common i.i.d. null, so the label-permutation p-values are
   descriptive rather than a generally valid point-process test.
2. **Location shift:** two normal samples share the same scale but have
   different centers. GET-V remains non-significant, while GET-Plus detects the
   added median-location difference.
3. **Scale difference:** two centered normal samples have different scales.
   Median-gap normalization makes GET-V insensitive to positive scale alone;
   GET-Omega detects the distributional difference through energy distance.

With the included seeds and 999 permutations, a typical run is:

```text
Example                                GET-V   GET-Plus   GET-Omega
--------------------------------------------------------------------
Spacing regularity (GET-V)              0.001      0.001       0.001
Location shift (GET-Plus)               0.275      0.001       0.001
Scale difference (GET-Omega)            0.810      0.856       0.001
```

GET-Plus and GET-Omega are nested composite procedures. Therefore, a strong
spacing signal can also make both composites significant, and a strong location
signal can also make GET-Omega significant. The examples identify the component
responsible for the result rather than treating the three tests as competitors.

## Method summary

For each sample, observations are sorted and adjacent gaps are divided by their
sample median gap. The normalized gaps are transformed with `log1p` and
summarized using a Vasicek-type entropy estimator. GET-V is the absolute
difference between the two estimates.

For GET-Plus and GET-Omega, each observed component is standardized using the
median and normal-consistent MAD of its own permutation distribution. GET-Plus
uses the maximum standardized spacing and location components. GET-Omega uses
the maximum standardized spacing, location, and energy components. Permutation
p-values use the finite-sample correction `(exceedances + 1) / (B + 1)`.

## Important limitations

- Individual-label permutation is not generally valid for dependent point
  processes. A process-aware resampling scheme may be required.
- GET-V is translation- and positive-scale-invariant, but it is not invariant
  to marginal shape.
- Discrete or rounded measurements with ties are not eligible for GET spacing
  inference in this implementation.
- Report the individual components together with the composite result so the
  source of a rejection remains interpretable.

## License and citation

The code is released under the MIT License. Citation metadata is provided in
`CITATION.cff`.

## Support

This repository is provided as a reference implementation accompanying the
published article and is made available as-is. Individual technical support,
implementation consulting, and guaranteed response times are not provided.
