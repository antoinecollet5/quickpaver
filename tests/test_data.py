# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET
"""Tests for the test data import."""

import quickpaver


def test_load_data() -> None:
    quickpaver.data.load_corsica_contour()
    quickpaver.data.load_france_and_corsica_contour()
    quickpaver.load_france_contour()
