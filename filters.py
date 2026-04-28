"""
One Euro Filter – adaptive low-pass filter for real-time signal smoothing.

Reference: Casiez, Roussel, Vogel – CHI 2012
https://cristal.univ-lille.fr/~casiez/1euro/

Smooths heavily when still (kills jitter), follows quickly when moving (low lag).
"""

import math
import time
from collections import deque


class OneEuroFilter:
    """One Euro Filter for a single scalar signal."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._prev_filtered_x: float | None = None
        self._prev_filtered_dx: float = 0.0
        self._last_time: float | None = None

    def __call__(self, x: float, t: float | None = None) -> float:
        """Filter a new sample. If `t` is None, uses wall-clock time."""
        if t is None:
            t = time.monotonic()

        if self._last_time is not None:
            dt = max(t - self._last_time, 1e-6)
        else:
            dt = 1.0 / 30.0  # assume ~30 fps on first frame
        self._last_time = t

        # 1. Raw derivative
        if self._prev_filtered_x is not None:
            dx = (x - self._prev_filtered_x) / dt
        else:
            dx = 0.0

        # 2. Smooth the derivative
        alpha_dx = self._smoothing_factor(dt, self.d_cutoff)
        filtered_dx = alpha_dx * dx + (1 - alpha_dx) * self._prev_filtered_dx
        self._prev_filtered_dx = filtered_dx

        # 3. Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(filtered_dx)

        # 4. Smooth the signal
        alpha_x = self._smoothing_factor(dt, cutoff)
        if self._prev_filtered_x is not None:
            filtered_x = alpha_x * x + (1 - alpha_x) * self._prev_filtered_x
        else:
            filtered_x = x
        self._prev_filtered_x = filtered_x

        return filtered_x

    def reset(self) -> None:
        self._prev_filtered_x = None
        self._prev_filtered_dx = 0.0
        self._last_time = None

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class OneEuroFilter2D:
    """Pair of OneEuroFilters for a 2D point (x, y)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def __call__(self, x: float, y: float, t: float | None = None) -> tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()


# ---------------------------------------------------------------------------
# Temporal debouncing for finger count
# ---------------------------------------------------------------------------

class FingerCountDebouncer:
    """Rolling-window majority-vote debouncer for finger-count values.

    Feed the raw count each frame; read `value` for the debounced result.
    """

    def __init__(self, window: int = 5, threshold: int = 3):
        self._buf: deque[int] = deque(maxlen=window)
        self._threshold = threshold
        self.value: int = 0

    def update(self, raw_count: int) -> int:
        self._buf.append(raw_count)
        # Majority vote: pick the count that appears >= threshold times
        for candidate in set(self._buf):
            if self._buf.count(candidate) >= self._threshold:
                self.value = candidate
                return self.value
        # No majority → keep previous value
        return self.value

    def reset(self) -> None:
        self._buf.clear()
        self.value = 0
