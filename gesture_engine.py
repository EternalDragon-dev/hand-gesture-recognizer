"""
Gesture Engine – detects dynamic gestures from a rolling history of hand landmarks.

Supports: swipe (4 directions), pinch, and point gestures.
"""

import math
import time
from enum import Enum


class Gesture(Enum):
    NONE        = "None"
    SWIPE_LEFT  = "\u2190 Swipe Left"
    SWIPE_RIGHT = "Swipe Right \u2192"
    SWIPE_UP    = "\u2191 Swipe Up"
    SWIPE_DOWN  = "Swipe Down \u2193"
    PINCH       = "Pinch"
    POINT       = "Point"


class GestureEngine:
    """Analyses a rolling history of hand landmarks to detect dynamic gestures.

    Create one instance per tracked hand.
    """

    def __init__(
        self,
        swipe_threshold: float = 0.12,
        swipe_window: int = 18,
        pinch_threshold: float = 0.045,
    ):
        self.swipe_threshold = swipe_threshold
        self.swipe_window = swipe_window
        self.pinch_threshold = pinch_threshold

        self.gesture: Gesture = Gesture.NONE
        self.cursor: tuple[float, float] | None = None  # normalised (x, y)
        self.is_clicking: bool = False
        self.pinch_scale: float = 1.0

        self._history: list[dict] = []
        self._max_history = 25
        self._pinch_start_dist: float | None = None
        self._swipe_cooldown: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, landmarks, fingers_extended: int) -> Gesture:
        """Feed the current frame's landmarks and finger count.

        `landmarks` is the MediaPipe NormalizedLandmarkList (list-like, 21 items).
        Returns the detected gesture.
        """
        index_tip = landmarks[8]
        thumb_tip = landmarks[4]

        self._history.append({
            "ix": index_tip.x, "iy": index_tip.y,
            "tx": thumb_tip.x, "ty": thumb_tip.y,
            "fingers": fingers_extended,
            "t": time.monotonic(),
        })
        if len(self._history) > self._max_history:
            self._history.pop(0)

        if self._swipe_cooldown > 0:
            self._swipe_cooldown -= 1

        thumb_index_dist = math.hypot(
            thumb_tip.x - index_tip.x,
            thumb_tip.y - index_tip.y,
        )

        # --- Priority: Pinch > Point > Swipe > None ---

        # 1. Pinch
        if thumb_index_dist < self.pinch_threshold:
            if self._pinch_start_dist is None:
                self._pinch_start_dist = max(thumb_index_dist, 0.001)
            self.pinch_scale = thumb_index_dist / self._pinch_start_dist

            if fingers_extended <= 2:
                self.gesture = Gesture.PINCH
                self.is_clicking = True
                self.cursor = (index_tip.x, index_tip.y)
            else:
                self.gesture = Gesture.PINCH
                self.is_clicking = False
                self.cursor = None
            return self.gesture

        # Pinch ended
        self._pinch_start_dist = None
        self.pinch_scale = 1.0
        self.is_clicking = False

        # 2. Point (only index extended)
        if fingers_extended == 1 and index_tip.y < landmarks[6].y:
            # MediaPipe: y increases downward, tip above PIP = extended
            self.gesture = Gesture.POINT
            self.cursor = (index_tip.x, index_tip.y)
            return self.gesture

        self.cursor = None

        # 3. Swipe (open hand, enough history, no cooldown)
        if (
            self._swipe_cooldown == 0
            and fingers_extended >= 3
            and len(self._history) >= self.swipe_window
        ):
            swipe = self._detect_swipe()
            if swipe is not None:
                self.gesture = swipe
                self._swipe_cooldown = self.swipe_window
                self._history.clear()
                return self.gesture

        # 4. Nothing
        self.gesture = Gesture.NONE
        return self.gesture

    def reset(self) -> None:
        self._history.clear()
        self.gesture = Gesture.NONE
        self.cursor = None
        self.is_clicking = False
        self.pinch_scale = 1.0
        self._pinch_start_dist = None
        self._swipe_cooldown = 0

    # ------------------------------------------------------------------
    # Swipe detection
    # ------------------------------------------------------------------

    def _detect_swipe(self) -> Gesture | None:
        window = self._history[-self.swipe_window:]
        first, last = window[0], window[-1]

        dx = last["ix"] - first["ix"]
        dy = last["iy"] - first["iy"]  # MediaPipe: y increases downward

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # Horizontal
        if abs_dx > self.swipe_threshold and abs_dx > abs_dy * 1.4:
            return Gesture.SWIPE_RIGHT if dx > 0 else Gesture.SWIPE_LEFT

        # Vertical (note: MediaPipe y is inverted vs screen "up")
        if abs_dy > self.swipe_threshold and abs_dy > abs_dx * 1.4:
            return Gesture.SWIPE_DOWN if dy > 0 else Gesture.SWIPE_UP

        return None
