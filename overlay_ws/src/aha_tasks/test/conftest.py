"""Host runs: import aha_perception from the source tree when it is not installed."""

import importlib.util
import os
import sys

if importlib.util.find_spec("aha_perception") is None:
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "aha_perception",
        ),
    )
