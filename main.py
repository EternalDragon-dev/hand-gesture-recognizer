#!/usr/bin/env python3
"""Hand Gesture Recognizer – multi-threaded pipeline with world coordinates.

Architecture:
  Camera Thread  → frame_queue →  Inference Thread  → result_queue →  Render Thread
  (60 fps cap)                    (MediaPipe + filters)                (OpenCV display)

World coordinates (hand_world_landmarks) are used for joint angle computation —
they provide metric-scale 3D positions (metres) centred on the palm, independent
of camera perspective.  Image coordinates are still used for all drawing.
"""

import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

from filters import OneEuroFilter2D, FingerCountDebouncer
from gesture_engine import GestureEngine, Gesture
from joint_angles import compute_joint_angles

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Landmark constants
FINGERTIPS = {4: "THUMB", 8: "INDEX", 12: "MIDDLE", 16: "RING", 20: "PINKY"}
PALM_INDICES = [0, 1, 5, 9, 13, 17]

# Colours (BGR)
COLOR_LANDMARK   = (0, 255, 255)
COLOR_FINGERTIP  = (0, 255, 0)
COLOR_CENTER     = (255, 0, 255)
COLOR_CONNECTION = (255, 200, 0)
COLOR_LABEL      = (255, 255, 255)

GESTURE_COLORS = {
    Gesture.NONE:        (255, 200, 0),
    Gesture.PINCH:       (0, 140, 255),
    Gesture.POINT:       (255, 100, 0),
    Gesture.SWIPE_LEFT:  (0, 255, 0),
    Gesture.SWIPE_RIGHT: (0, 255, 0),
    Gesture.SWIPE_UP:    (0, 255, 0),
    Gesture.SWIPE_DOWN:  (0, 255, 0),
}


# ---------------------------------------------------------------------------
# Data passed between threads
# ---------------------------------------------------------------------------
@dataclass
class HandResult:
    """Per-hand analysis result produced by the inference thread."""
    hand_lm: object                      # MediaPipe NormalizedLandmarkList (for drawing)
    label: str                           # "Left" / "Right"
    fingers: int                         # debounced finger count
    gesture: Gesture                     # detected gesture
    angles: dict                         # joint angles (from world coords when available)
    conn_color: tuple                    # gesture-aware skeleton colour
    cursor: tuple | None                 # normalised cursor position
    is_clicking: bool                    # pinch-while-pointing
    trail: list                          # pixel trail points


@dataclass
class FrameResult:
    """Full frame result passed from inference → render thread."""
    frame: np.ndarray
    w: int
    h: int
    hands: list = field(default_factory=list)
    gesture_toast: str = ""
    toast_expiry: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_palm_center(landmarks, w: int, h: int) -> tuple[int, int]:
    xs = [landmarks[i].x * w for i in PALM_INDICES]
    ys = [landmarks[i].y * h for i in PALM_INDICES]
    return int(np.mean(xs)), int(np.mean(ys))


def count_fingers_up(landmarks) -> int:
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    count = 0
    if landmarks[4].x < landmarks[3].x:
        count += 1
    for tip, pip_ in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip_].y:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Thread 1: Camera capture
# ---------------------------------------------------------------------------
def camera_thread(frame_q: queue.Queue, stop_event: threading.Event) -> None:
    """Reads frames from the webcam and pushes them to the queue."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break

        frame = cv2.flip(frame, 1)

        # Drop old frames so inference always works on the latest
        try:
            frame_q.put_nowait((frame, time.monotonic()))
        except queue.Full:
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
            frame_q.put_nowait((frame, time.monotonic()))

    cap.release()


# ---------------------------------------------------------------------------
# Thread 2: ML inference + analysis
# ---------------------------------------------------------------------------
def inference_thread(
    frame_q: queue.Queue,
    result_q: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Runs MediaPipe hand pose + all analysis (filters, gestures, angles)."""

    hand_filters: list[dict[int, OneEuroFilter2D]] = [{}, {}]
    finger_debouncers = [FingerCountDebouncer(), FingerCountDebouncer()]
    gesture_engines = [GestureEngine(), GestureEngine()]
    trail_buffers: list[list[tuple[int, int]]] = [[], []]
    frames_without_hand = [0, 0]
    TRAIL_MAX = 12

    gesture_toast = ""
    toast_expiry = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while not stop_event.is_set():
            try:
                frame, t = frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_results: list[HandResult] = []

            if results.multi_hand_landmarks:
                # World landmarks (metric 3D, palm-centred) — used for angles
                world_lms_list = getattr(results, "multi_hand_world_landmarks", None) or []

                for hand_idx, (hand_lm, hand_info) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    if hand_idx >= 2:
                        break

                    # Smooth image landmarks
                    for i, lm in enumerate(hand_lm.landmark):
                        if i not in hand_filters[hand_idx]:
                            hand_filters[hand_idx][i] = OneEuroFilter2D(
                                min_cutoff=1.0, beta=0.007, d_cutoff=1.0
                            )
                        lm.x, lm.y = hand_filters[hand_idx][i](lm.x, lm.y, t)

                    # Gesture detection
                    raw_fingers = count_fingers_up(hand_lm.landmark)
                    fingers = finger_debouncers[hand_idx].update(raw_fingers)
                    detected = gesture_engines[hand_idx].update(hand_lm.landmark, fingers)

                    # Joint angles — prefer world coordinates (metric 3D)
                    if hand_idx < len(world_lms_list) and world_lms_list[hand_idx] is not None:
                        angles = compute_joint_angles(world_lms_list[hand_idx].landmark)
                    else:
                        angles = compute_joint_angles(hand_lm.landmark)

                    # Trail
                    ix_tip = hand_lm.landmark[8]
                    trail_buffers[hand_idx].append((int(ix_tip.x * w), int(ix_tip.y * h)))
                    if len(trail_buffers[hand_idx]) > TRAIL_MAX:
                        trail_buffers[hand_idx].pop(0)
                    frames_without_hand[hand_idx] = 0

                    # Toast
                    if detected != Gesture.NONE:
                        gesture_toast = detected.value
                        toast_expiry = time.monotonic() + 1.0

                    hand_results.append(HandResult(
                        hand_lm=hand_lm,
                        label=hand_info.classification[0].label,
                        fingers=fingers,
                        gesture=detected,
                        angles=angles,
                        conn_color=GESTURE_COLORS.get(detected, COLOR_CONNECTION),
                        cursor=gesture_engines[hand_idx].cursor,
                        is_clicking=gesture_engines[hand_idx].is_clicking,
                        trail=list(trail_buffers[hand_idx]),
                    ))

                # Reset unused slots
                active = len(results.multi_hand_landmarks)
                for slot in range(active, 2):
                    hand_filters[slot].clear()
                    finger_debouncers[slot].reset()
                    gesture_engines[slot].reset()
                    frames_without_hand[slot] += 1
                    if frames_without_hand[slot] > 5:
                        trail_buffers[slot].clear()
            else:
                for slot in range(2):
                    hand_filters[slot].clear()
                    finger_debouncers[slot].reset()
                    gesture_engines[slot].reset()
                    frames_without_hand[slot] += 1
                    if frames_without_hand[slot] > 5:
                        trail_buffers[slot].clear()

            fr = FrameResult(
                frame=frame, w=w, h=h,
                hands=hand_results,
                gesture_toast=gesture_toast,
                toast_expiry=toast_expiry,
            )

            # Drop stale results
            try:
                result_q.put_nowait(fr)
            except queue.Full:
                try:
                    result_q.get_nowait()
                except queue.Empty:
                    pass
                result_q.put_nowait(fr)


# ---------------------------------------------------------------------------
# Thread 3: Rendering (runs on main thread for OpenCV GUI compatibility)
# ---------------------------------------------------------------------------
def render_loop(result_q: queue.Queue, stop_event: threading.Event) -> None:
    """Draws overlays and displays the frame. Must run on the main thread."""

    while not stop_event.is_set():
        try:
            fr: FrameResult = result_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame, w, h = fr.frame, fr.w, fr.h

        for hr in fr.hands:
            # Skeleton with gesture-aware colour
            mp_drawing.draw_landmarks(
                frame, hr.hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=COLOR_LANDMARK, thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=hr.conn_color, thickness=2),
            )

            # Fingertip labels
            for idx, name in FINGERTIPS.items():
                lm = hr.hand_lm.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 8, COLOR_FINGERTIP, cv2.FILLED)
                cv2.putText(frame, name, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LABEL, 1, cv2.LINE_AA)

            # Palm center
            pcx, pcy = compute_palm_center(hr.hand_lm.landmark, w, h)
            cv2.circle(frame, (pcx, pcy), 10, COLOR_CENTER, cv2.FILLED)
            cv2.putText(frame, "CENTER", (pcx + 12, pcy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CENTER, 1, cv2.LINE_AA)

            # Info text
            info = f"{hr.label} hand | Fingers: {hr.fingers}"
            if hr.gesture != Gesture.NONE:
                info += f" | {hr.gesture.value}"
            cv2.putText(frame, info, (pcx - 80, pcy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 2, cv2.LINE_AA)

            # Joint angle sample (world-coord based when available)
            if "index_pip" in hr.angles:
                cv2.putText(frame, f"Index PIP: {hr.angles['index_pip']:.0f} deg",
                            (pcx - 60, pcy + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Fingertip trail
            trail = hr.trail
            if len(trail) >= 2:
                for ti in range(1, len(trail)):
                    progress = ti / len(trail)
                    thickness = max(1, int(1 + progress * 3))
                    overlay = frame.copy()
                    cv2.line(overlay, trail[ti - 1], trail[ti],
                             (255, 200, 80), thickness, cv2.LINE_AA)
                    cv2.addWeighted(overlay, progress, frame, 1 - progress * 0.3, 0, frame)
                cv2.circle(frame, trail[-1], 5, (255, 200, 80), cv2.FILLED)

            # Cursor reticle
            if hr.cursor is not None:
                cx = int(hr.cursor[0] * w)
                cy = int(hr.cursor[1] * h)
                clr = (0, 140, 255) if hr.gesture == Gesture.PINCH else (255, 100, 0)
                cv2.circle(frame, (cx, cy), 18, clr, 2)
                cv2.line(frame, (cx - 26, cy), (cx - 14, cy), clr, 2)
                cv2.line(frame, (cx + 14, cy), (cx + 26, cy), clr, 2)
                cv2.line(frame, (cx, cy - 26), (cx, cy - 14), clr, 2)
                cv2.line(frame, (cx, cy + 14), (cx, cy + 26), clr, 2)
                if hr.gesture == Gesture.PINCH:
                    cv2.circle(frame, (cx, cy), 5, (0, 140, 255), cv2.FILLED)

        # Gesture toast
        if fr.gesture_toast and time.monotonic() < fr.toast_expiry:
            cv2.putText(frame, fr.gesture_toast, (w // 2 - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_LABEL, 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Recognizer (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Hand Gesture Recognizer – multi-threaded pipeline")
    print("  Thread 1: Camera capture")
    print("  Thread 2: ML inference + analysis (world coords for angles)")
    print("  Thread 3: Rendering (this thread)")
    print("Press 'q' to quit.\n")

    stop = threading.Event()
    frame_q = queue.Queue(maxsize=2)
    result_q = queue.Queue(maxsize=2)

    cam = threading.Thread(target=camera_thread, args=(frame_q, stop), daemon=True)
    inf = threading.Thread(target=inference_thread, args=(frame_q, result_q, stop), daemon=True)
    cam.start()
    inf.start()

    # Render on main thread (OpenCV requires it for GUI)
    render_loop(result_q, stop)

    stop.set()
    cam.join(timeout=2)
    inf.join(timeout=2)


if __name__ == "__main__":
    main()
