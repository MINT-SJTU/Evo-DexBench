"""Coordinate transforms shared by tracked-hand teleoperation backends."""

from __future__ import annotations

import numpy as np


OPERATOR2MANO_RIGHT = np.asarray(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
OPERATOR2MANO_LEFT = np.asarray(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)
