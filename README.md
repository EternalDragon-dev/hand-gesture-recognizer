# Hand Gesture Recognizer

A real-time hand tracking application that uses your webcam to detect hands, map all 21 landmarks per hand, label each fingertip, display the palm center, and count extended fingers.

## Features

- **21-point hand landmark detection** — full skeletal map of every joint.
- **Fingertip labels** — THUMB, INDEX, MIDDLE, RING, PINKY highlighted in green.
- **Palm center point** — computed from wrist + finger base joints, shown in magenta.
- **Finger counting** — detects how many fingers are extended per hand.
- **Handedness detection** — identifies Left vs. Right hand.
- **Multi-hand support** — tracks up to 2 hands simultaneously.
- **Mirror view** — frame is flipped so the display matches your natural perspective.

## Project Structure

```
hand_gesture_recognizer/
├── main.py            # Full application (single file)
└── requirements.txt   # Python dependencies
```

## Prerequisites

- Python 3.9+
- A working webcam

## Setup

```bash
cd hand_gesture_recognizer
pip install -r requirements.txt
```

> **Note:** Unlike `dlib`, MediaPipe ships pre-built wheels — installation is fast.

### Dependency compatibility

MediaPipe may pin numpy to `<2.0`. If you also have `opencv-python` installed from a numpy 2.x build, you'll get import errors. Fix:

```bash
pip install "opencv-python<4.11"
```

## Usage

```bash
python main.py
```

Hold your hand(s) in front of the webcam. You'll see:

- Yellow dots + cyan lines for all 21 landmarks and bone connections.
- Green circles on each fingertip with labels.
- A magenta dot at the palm center labelled "CENTER".
- Text showing handedness and finger count.

Press **q** to quit.

## Dependencies

| Package | Purpose |
|---|---|
| `mediapipe` | Hand landmark detection (21 points, ML pipeline) |
| `opencv-python` | Webcam capture, drawing, display |
| `numpy` | Palm center computation (mean of coordinates) |

## How It Works (Brief)

1. Each webcam frame is flipped (mirror) and converted to RGB.
2. MediaPipe's hand detection model locates hand bounding boxes.
3. A landmark model predicts 21 3D keypoints per detected hand.
4. The app draws connections, highlights fingertips, computes the palm center as the centroid of the wrist + 5 finger base joints, and counts extended fingers using a simple tip-vs-joint heuristic.

See [TUTORIAL.md](TUTORIAL.md) for a full technical walkthrough.
