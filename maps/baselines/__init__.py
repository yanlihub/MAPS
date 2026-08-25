"""Post-processing baselines that MAPS is compared against.

Every baseline consumes the same synthetic pool and returns a subset of the same target
size, so differences between them reflect the selection rule alone.
"""

from .authenticity import alaa_authenticity_filter
from .moment_matching import wang_moment_matching
from .qde import qde_utility_filter

__all__ = [
    "alaa_authenticity_filter",
    "wang_moment_matching",
    "qde_utility_filter",
]
