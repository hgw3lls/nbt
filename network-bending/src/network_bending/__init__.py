"""Network Bending: a media-theoretical neural bending package."""

# Import bends to populate registry at package import time
from . import bends  # noqa: F401
from .toolbox import apply_bend, list_bend_summaries
<<<<<<< ours

__all__ = ["apply_bend", "list_bend_summaries", "bends"]
=======
from .gui import launch_gui

__all__ = ["apply_bend", "list_bend_summaries", "launch_gui", "bends"]
>>>>>>> theirs
