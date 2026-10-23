"""Score a URL for phishing risk without opening the page.

Callers usually need only two types. ``Pipeline`` extracts features and scores
a URL. ``Verdict`` is the result: a probability, a risk band, and short reasons.
"""

from phishguard.pipeline import Pipeline
from phishguard.verdict import Verdict

__all__ = ["Pipeline", "Verdict"]
__version__ = "0.1.0"
