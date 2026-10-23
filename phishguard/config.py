"""Frozen model choice for ``python train.py``.

These two values are what ``python compare.py`` measured. The comparison
picks a threshold on a validation slice, picks the model from those
validation scores, and only then scores the test set. ``comparison.json``
is the record of that run. Change these only by re-running it.
"""

from __future__ import annotations

DEFAULT_MODEL = "random_forest"
DEFAULT_THRESHOLD = 0.35
