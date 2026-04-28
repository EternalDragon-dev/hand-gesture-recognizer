#!/usr/bin/env python3
"""Hand Gesture Recognizer – maps fingers and palm center in real-time."""

import time

import cv2
import mediapipe as mp
import numpy as np

from filters import OneEuroFilter2D, FingerCountDebouncer
from gesture_engine import GestureEngine, Gesture
from joint_angles import compute_joint_angles

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Fingertip landmark indices and labels
FINGERTIPS = {
    4: "THUMB",
    8: "INDEX",
    12: "MIDDLE",
    16: "RING",
    20: "PINKY",
}

# Palm landmarks used to compute center (wrist + base of each finger)
PALM_INDICES = [0, 1, 5, 9, 13, 17]

# Colors (BGR)
COLOR_LANDMARK = (0, 255, 255)    # yellow – regular landmarks
COLOR_FINGERTIP = (0, 255, 0)     # green  – fingertips
COLOR_CENTER = (255, 0, 255)      # magenta – palm center
COLOR_CONNECTION = (255, 200, 0)  # cyan-ish – bone connections (idle)
COLOR_LABEL = (255, 255, 255)     # white – text

# Gesture-aware skeleton colours (BGR)
GESTURE_COLORS = {
    Gesture.NONE:        (255, 200, 0),   # cyan (default)
    Gesture.PINCH:       (0, 140, 255),   # orange
    Gesture.POINT:       (255, 100, 0),   # blue
    Gesture.SWIPE_LEFT:  (0, 255, 0),     # green
    Gesture.SWIPE_RIGHT: (0, 255, 0),
    Gesture.SWIPE_UP:    (0, 255, 0),
    Gesture.SWIPE_DOWN:  (0, 255, 0),
}


def compute_palm_center(landmarks, w: int, h: int) -> tuple[int, int]:
    """Return the (x, y) pixel coords of the palm center."""
    xs = [landmarks[i].x * w for i in PALM_INDICES]
    ys = [landmarks[i].y * h for i in PALM_INDICES]
    return int(np.mean(xs)), int(np.mean(ys))


def draw_hand(image, hand_landmarks, w: int, h: int, conn_color=None) -> None:
    """Draw landmarks, connections, fingertip labels, and palm center."""
    landmarks = hand_landmarks.landmark
    color = conn_color if conn_color is not None else COLOR_CONNECTION

    # Draw bone connections
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=COLOR_LANDMARK, thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=color, thickness=2),
    )

    # Highlight and label each fingertip
    for idx, name in FINGERTIPS.items():
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 8, COLOR_FINGERTIP, cv2.FILLED)
        cv2.putText(image, name, (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LABEL, 1, cv2.LINE_AA)

    # Draw palm center
    pcx, pcy = compute_palm_center(landmarks, w, h)
    cv2.circle(image, (pcx, pcy), 10, COLOR_CENTER, cv2.FILLED)
    cv2.putText(image, "CENTER", (pcx + 12, pcy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CENTER, 1, cv2.LINE_AA)


def count_fingers_up(landmarks) -> int:
    """Count how many fingers are extended (simple heuristic)."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    count = 0

    # Thumb: compare x of tip vs IP joint (works for right hand facing camera)
    if landmarks[4].x < landmarks[3].x:
        count += 1

    # Other four fingers: tip above PIP joint means extended
    for tip, pip_ in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip_].y:
            count += 1

    return count


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Hand Gesture Recognizer – press 'q' to quit.")

    # Per-hand filters: dict of {landmark_index: OneEuroFilter2D}
    hand_filters: list[dict[int, OneEuroFilter2D]] = [{}, {}]
    finger_debouncers = [FingerCountDebouncer(), FingerCountDebouncer()]
    gesture_engines = [GestureEngine(), GestureEngine()]

    # Gesture toast state
    gesture_toast = ""
    toast_expiry = 0.0

    # Fingertip trail: list of (x_px, y_px) per hand slot
    trail_buffers: list[list[tuple[int, int]]] = [[], []]
    TRAIL_MAX = 12
    frames_without_hand = [0, 0]

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip horizontally for a mirror-like view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            t = time.monotonic()

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_idx, (hand_lm, hand_info) in enumerate(
                    zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness,
                    )
                ):
                    if hand_idx >= 2:
                        break

                    # Smooth each landmark with OneEuroFilter
                    for i, lm in enumerate(hand_lm.landmark):
                        if i not in hand_filters[hand_idx]:
                            hand_filters[hand_idx][i] = OneEuroFilter2D(
                                min_cutoff=1.0, beta=0.007, d_cutoff=1.0
                            )
                        lm.x, lm.y = hand_filters[hand_idx][i](lm.x, lm.y, t)

                    # Detect gesture
                    raw_fingers = count_fingers_up(hand_lm.landmark)
                    fingers = finger_debouncers[hand_idx].update(raw_fingers)
                    detected = gesture_engines[hand_idx].update(
                        hand_lm.landmark, fingers
                    )

                    # Use gesture-aware skeleton colour
                    conn_color = GESTURE_COLORS.get(detected, COLOR_CONNECTION)
                    draw_hand(frame, hand_lm, w, h, conn_color=conn_color)

                    # Compute joint angles (available for robotics / logging)
                    angles = compute_joint_angles(hand_lm.landmark)

                    # Show debounced finger count, handedness, and gesture
                    label = hand_info.classification[0].label  # "Left" / "Right"
                    pcx, pcy = compute_palm_center(hand_lm.landmark, w, h)
                    info = f"{label} hand | Fingers: {fingers}"
                    if detected != Gesture.NONE:
                        info += f" | {detected.value}"
                    cv2.putText(
                        frame, info,
                        (pcx - 80, pcy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        COLOR_LABEL,
                        2,
                        cv2.LINE_AA,
                    )

                    # Show index PIP angle as a sample (useful for servo mapping)
                    if "index_pip" in angles:
                        angle_text = f"Index PIP: {angles['index_pip']:.0f} deg"
                        cv2.putText(
                            frame, angle_text,
                            (pcx - 60, pcy + 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (200, 200, 200), 1, cv2.LINE_AA,
                        )

                    # Gesture toast (top of screen)
                    if detected != Gesture.NONE:
                        gesture_toast = detected.value
                        toast_expiry = time.monotonic() + 1.0

                    # Update fingertip trail
                    ix_tip = hand_lm.landmark[8]
                    trail_buffers[hand_idx].append(
                        (int(ix_tip.x * w), int(ix_tip.y * h))
                    )
                    if len(trail_buffers[hand_idx]) > TRAIL_MAX:
                        trail_buffers[hand_idx].pop(0)
                    frames_without_hand[hand_idx] = 0

                    # Draw fingertip trail (fading polyline)
                    trail = trail_buffers[hand_idx]
                    if len(trail) >= 2:
                        for ti in range(1, len(trail)):
                            progress = ti / len(trail)  # 0→1
                            alpha = int(25 + progress * 230)  # 25→255
                            thickness = max(1, int(1 + progress * 3))
                            # Light blue trail colour with alpha via overlay
                            overlay = frame.copy()
                            cv2.line(overlay, trail[ti - 1], trail[ti],
                                     (255, 200, 80), thickness, cv2.LINE_AA)
                            cv2.addWeighted(overlay, progress, frame,
                                            1 - progress * 0.3, 0, frame)
                        # Glow dot at tip
                        cv2.circle(frame, trail[-1], 5, (255, 200, 80), cv2.FILLED)

                    # Draw cursor for point / pinch
                    if gesture_engines[hand_idx].cursor is not None:
                        cx = int(gesture_engines[hand_idx].cursor[0] * w)
                        cy = int(gesture_engines[hand_idx].cursor[1] * h)
                        color = (0, 140, 255) if detected == Gesture.PINCH else (255, 100, 0)
                        cv2.circle(frame, (cx, cy), 18, color, 2)
                        # Crosshair
                        cv2.line(frame, (cx - 26, cy), (cx - 14, cy), color, 2)
                        cv2.line(frame, (cx + 14, cy), (cx + 26, cy), color, 2)
                        cv2.line(frame, (cx, cy - 26), (cx, cy - 14), color, 2)
                        cv2.line(frame, (cx, cy + 14), (cx, cy + 26), color, 2)
                        if detected == Gesture.PINCH:
                            cv2.circle(frame, (cx, cy), 5, (0, 140, 255), cv2.FILLED)

                # Reset filters for hands that disappeared
                active = len(results.multi_hand_landmarks)
                for slot in range(active, 2):
                    hand_filters[slot].clear()
                    finger_debouncers[slot].reset()
                    gesture_engines[slot].reset()
                    frames_without_hand[slot] += 1
                    if frames_without_hand[slot] > 5:
                        trail_buffers[slot].clear()
            else:
                # No hands — reset all filters
                for slot in range(2):
                    hand_filters[slot].clear()
                    finger_debouncers[slot].reset()
                    gesture_engines[slot].reset()
                    frames_without_hand[slot] += 1
                    if frames_without_hand[slot] > 5:
                        trail_buffers[slot].clear()

            # Draw gesture toast at top of screen
            if gesture_toast and time.monotonic() < toast_expiry:
                cv2.putText(
                    frame, gesture_toast,
                    (w // 2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA,
                )

            cv2.imshow("Hand Gesture Recognizer (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
