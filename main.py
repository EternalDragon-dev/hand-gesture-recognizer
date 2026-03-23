#!/usr/bin/env python3
"""Hand Gesture Recognizer – maps fingers and palm center in real-time."""

import cv2
import mediapipe as mp
import numpy as np

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
COLOR_CONNECTION = (255, 200, 0)  # cyan-ish – bone connections
COLOR_LABEL = (255, 255, 255)     # white – text


def compute_palm_center(landmarks, w: int, h: int) -> tuple[int, int]:
    """Return the (x, y) pixel coords of the palm center."""
    xs = [landmarks[i].x * w for i in PALM_INDICES]
    ys = [landmarks[i].y * h for i in PALM_INDICES]
    return int(np.mean(xs)), int(np.mean(ys))


def draw_hand(image, hand_landmarks, w: int, h: int) -> None:
    """Draw landmarks, connections, fingertip labels, and palm center."""
    landmarks = hand_landmarks.landmark

    # Draw bone connections
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=COLOR_LANDMARK, thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=COLOR_CONNECTION, thickness=2),
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

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_lm, hand_info in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):
                    draw_hand(frame, hand_lm, w, h)

                    # Show finger count and handedness
                    label = hand_info.classification[0].label  # "Left" / "Right"
                    fingers = count_fingers_up(hand_lm.landmark)
                    pcx, pcy = compute_palm_center(hand_lm.landmark, w, h)
                    cv2.putText(
                        frame,
                        f"{label} hand | Fingers: {fingers}",
                        (pcx - 60, pcy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        COLOR_LABEL,
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow("Hand Gesture Recognizer (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
