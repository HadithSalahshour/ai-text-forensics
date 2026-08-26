"""Calibration: corpus management, dataset building, training, and the fitted
detector used at inference.

The corpus lives under `data/corpus/` and is gitignored. The human half is
expected to be the user's own writing; it stays on this machine.
"""

from . import corpus, dataset, fitted, train

__all__ = ["corpus", "dataset", "fitted", "train"]
