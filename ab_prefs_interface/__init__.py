from ab_prefs_interface.matching import build_comparison_units
from ab_prefs_interface.sampling import SAMPLING_STRATEGIES, build_session_queue
from ab_prefs_interface.scoring import score_units

__all__ = [
    "SAMPLING_STRATEGIES",
    "build_comparison_units",
    "build_session_queue",
    "score_units",
]


def __getattr__(name: str):
    if name == "NotebookPreferenceInterface":
        from ab_prefs_interface.interface_notebook import NotebookPreferenceInterface
        return NotebookPreferenceInterface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
