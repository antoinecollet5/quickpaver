# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Provide utils to work with list."""

from __future__ import annotations

from enum import Enum
from typing import List, Sequence, Union

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int64]
NDArrayBool = npt.NDArray[np.bool_]
Int = Union[int, NDArrayInt, Sequence[int]]
ArrayLike = npt.ArrayLike


class StrEnum(str, Enum):
    """Hashable string Enum that can be used for pd.DataFrame column names."""

    def __str__(self) -> str:
        """Return instance value."""
        return self.value

    def __hash__(self) -> int:
        """Return the hash of the value."""
        return hash(self.value)

    def __eq__(self, other) -> bool:
        """Return if two instances are equal."""
        if not isinstance(other, StrEnum) and not isinstance(other, str):
            return False
        return self.value == other

    @classmethod
    def to_list(cls) -> List[StrEnum]:
        """Return all enums as a list."""
        return list(cls)
