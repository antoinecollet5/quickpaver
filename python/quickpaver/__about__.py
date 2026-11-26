# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Get the metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    # The version lives in Cargo.toml (single source of truth, read by maturin).
    __version__ = version("quickpaver")
except PackageNotFoundError:  # pragma: no cover - running from a raw checkout
    __version__ = "0.0.0+unknown"
__author__ = "Antoine Collet"
__email__ = "antoine.collet5@gmail.com"
