"""Public interface for Gap-Entropy Testing."""

from .core import GETComponents, GETResult, TestResult, get_test, spacing_entropy

__all__ = [
    "GETComponents",
    "GETResult",
    "TestResult",
    "get_test",
    "spacing_entropy",
]

__version__ = "0.1.0"

