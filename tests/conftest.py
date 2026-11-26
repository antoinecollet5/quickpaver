# SPDX-License-Identifier: BSD-3-Clause
"""Shared pytest configuration."""

import os

# Tests never need a GUI: use matplotlib's non-interactive Agg backend. Without
# it matplotlib may pick TkAgg, which needs a display and a working Tcl/Tk
# install (missing or flaky on headless CI runners, e.g. Windows + uv Python).
# Must be set before matplotlib.pyplot is first imported.
os.environ["MPLBACKEND"] = "Agg"
