# Hand Gesture Recognizer — Complete Technical Breakdown

A line-by-line, concept-by-concept analysis of a real-time hand landmark detection and finger-state estimation system. Written for engineers who need to understand not just *what* the code does, but *why* each decision was made and how every piece connects — particularly if you intend to integrate this into a robotics pipeline (teleoperation, HRI, gesture-driven control, etc.).

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [System Requirements and Dependencies](#2-system-requirements-and-dependencies)
3. [The ML Pipeline — How MediaPipe Hands Works](#3-the-ml-pipeline--how-mediapipe-hands-works)
4. [Hand Anatomy and the 21-Landmark Model](#4-hand-anatomy-and-the-21-landmark-model)
5. [Coordinate Systems and Transformations](#5-coordinate-systems-and-transformations)
6. [Complete Source Code with Annotations](#6-complete-source-code-with-annotations)
7. [Function-by-Function Deep Dive](#7-function-by-function-deep-dive)
8. [The Main Loop — Frame Acquisition to Display](#8-the-main-loop--frame-acquisition-to-display)
9. [Finger Extension Detection — Geometry and Limitations](#9-finger-extension-detection--geometry-and-limitations)
10. [Palm Center Estimation](#10-palm-center-estimation)
11. [Rendering Pipeline](#11-rendering-pipeline)
12. [Building This From Scratch — Step by Step](#12-building-this-from-scratch--step-by-step)
13. [Latency Budget and Real-Time Constraints](#13-latency-budget-and-real-time-constraints)
14. [Robotics Integration Considerations](#14-robotics-integration-considerations)
15. [Failure Modes and Edge Cases](#15-failure-modes-and-edge-cases)
16. [Extensions and Next Steps](#16-extensions-and-next-steps)

---

## 1. Purpose and Scope

This application captures video from a webcam, detects up to two human hands per frame, estimates 21 skeletal keypoints per hand, and renders an annotated overlay in real-time. It also performs:

- **Fingertip identification** — labels each of the five fingertips by name.
- **Palm center estimation** — computes a stable centroid from the rigid palm structure.
- **Finger extension counting** — determines how many fingers are open using a joint-angle heuristic.
- **Handedness classification** — reports whether each hand is left or right.

**What it is not:** This is not a gesture classifier. It provides the *landmark data* that a gesture classifier would consume. Think of it as the perception layer in a perception → decision → actuation pipeline.

---

## 2. System Requirements and Dependencies

### Runtime

- **Python 3.9+** (uses `tuple[int, int]` return type syntax from PEP 585).
- **Webcam** accessible at device index 0 (`/dev/video0` on Linux, default camera on macOS/Windows).

### Libraries

```
opencv-python   — Video capture (V4L2/AVFoundation/DirectShow backend), image manipulation, GUI rendering.
mediapipe       — Hand detection and landmark regression ML models, inference runtime.
numpy           — Vectorised arithmetic for centroid computation.
```

**Install:**

```bash
pip install opencv-python mediapipe numpy
```

### Compatibility note

MediaPipe 0.10.x pins `numpy<2.0`. If your environment also has `opencv-python>=4.11` (built against numpy 2.x), OpenCV will fail to import with `numpy._core.multiarray` errors. Fix:

```bash
pip install "opencv-python<4.11"
```

This is a binary ABI mismatch, not a code issue. It happens because the OpenCV wheel's compiled C extensions link against numpy 2.x symbols that don't exist in numpy 1.x.

### Hardware

- **CPU only.** MediaPipe Hands is optimised for CPU inference. On macOS it can use Apple's ML Compute backend. No GPU or CUDA required.
- Expect ~25–30 fps on a modern laptop CPU (Intel i5/M1 class) at 720p input.

---

## 3. The ML Pipeline — How MediaPipe Hands Works

MediaPipe Hands uses a **two-stage architecture** designed for real-time inference on commodity hardware.

### Stage 1: Palm Detection (BlazePalm)

A **Single Shot Detector (SSD)** variant called BlazePalm scans the full frame for palm bounding boxes.

**Why palms, not hands?**
- Palms are approximately rigid and square-shaped. This makes them well-suited for anchor-box-based detectors, which assume roughly fixed aspect ratios.
- Full hands are highly articulated — fingers can be extended, curled, crossed, or occluded. A detector trained on "hand" would need far more anchor configurations.
- Palms have fewer self-occlusion patterns and are visible in most hand poses.

The palm detector outputs an oriented bounding box (position + rotation) that is used to crop and align the hand region for stage 2.

**When does it run?**
- On the **first frame** where a hand appears.
- Whenever **tracking confidence** drops below the `min_tracking_confidence` threshold.
- In steady-state tracking, the palm detector is **skipped entirely** — this is the main source of MediaPipe's speed advantage.

### Stage 2: Hand Landmark Model

Given the cropped, aligned hand region from stage 1, a **regression CNN** predicts 21 keypoints in 3D (x, y, z). It also outputs a handedness score and a "hand presence" confidence that drives the re-detection trigger.

The model was trained on ~30,000 real-world hand images plus synthetic data, with keypoints manually annotated.

**Output format per landmark:**
- `x`: horizontal position, normalised to [0.0, 1.0] relative to image width.
- `y`: vertical position, normalised to [0.0, 1.0] relative to image height.
- `z`: depth relative to the wrist, in roughly the same scale as `x`. Negative values = closer to camera.

### Pipeline data flow

```
Webcam frame (BGR)
    │
    ▼
cv2.cvtColor → RGB frame
    │
    ▼
hands.process(rgb)
    │
    ├──► [Palm Detector]  (runs if no active track or confidence drop)
    │         │
    │         ▼
    │    Oriented bounding box(es)
    │         │
    └──► [Landmark Model]  (runs every frame on cropped hand region)
              │
              ▼
         21 keypoints (x, y, z) per hand
         Handedness classification
         Hand presence confidence
```

---

## 4. Hand Anatomy and the 21-Landmark Model

### Landmark index map

```
             THUMB    INDEX    MIDDLE    RING    PINKY
  TIP          4        8       12       16       20
  DIP          3        7       11       15       19
  PIP          2        6       10       14       18
  MCP          1        5        9       13       17
                         WRIST: 0
```

### Joint terminology

| Abbreviation | Full Name | Anatomical Meaning |
|---|---|---|
| **MCP** | Metacarpophalangeal | Knuckle — where the finger meets the palm. |
| **PIP** | Proximal Interphalangeal | First bend of the finger (middle joint). |
| **DIP** | Distal Interphalangeal | Second bend (closest to fingertip). |
| **TIP** | — | The very end of the finger. |
| **CMC** | Carpometacarpal | Thumb base joint (landmark 1 in this model). |
| **IP** | Interphalangeal | The thumb's single interphalangeal joint (landmark 3). |

### Why 21 points?

Each of the 5 fingers has 4 landmarks (MCP, PIP, DIP, TIP), giving 20. The 21st is the **wrist** (landmark 0), which anchors the entire skeleton. The thumb is a special case: its "PIP" is actually the CMC joint and its "DIP" is the IP joint, because the thumb has only 2 phalanges (not 3 like the other fingers).

### Landmark connectivity (bones)

MediaPipe defines these as `mp_hands.HAND_CONNECTIONS` — a frozenset of `(start_idx, end_idx)` tuples:

```
Wrist → Thumb:   (0,1), (1,2), (2,3), (3,4)
Wrist → Index:   (0,5), (5,6), (6,7), (7,8)
Wrist → Middle:  (0,9), (9,10), (10,11), (11,12)
Wrist → Ring:    (0,13), (13,14), (14,15), (15,16)
Wrist → Pinky:   (0,17), (17,18), (18,19), (19,20)
Palm bridge:     (5,9), (9,13), (13,17)
Thumb-index gap: (0,5) — shared with index chain
```

The "palm bridge" connections (5→9→13→17) link the MCP joints across the palm, forming the rigid palm quadrilateral. These are important because they give the rendered skeleton a physically plausible structure rather than a star-burst from the wrist.

---

## 5. Coordinate Systems and Transformations

Understanding the coordinate system is critical for robotics applications.

### MediaPipe normalised coordinates

All landmark positions are returned as normalised floats:

```
x ∈ [0.0, 1.0]   — left edge = 0.0, right edge = 1.0
y ∈ [0.0, 1.0]   — top edge = 0.0, bottom edge = 1.0
z ∈ (unbounded)   — depth relative to wrist, roughly same scale as x
```

### Pixel coordinate conversion

To draw on an image of size (w, h):

```python
pixel_x = int(landmark.x * w)
pixel_y = int(landmark.y * h)
```

This is a simple linear mapping. There is no lens distortion correction — MediaPipe assumes a pinhole camera model. If you're using a wide-angle or fisheye lens (common on robotics platforms), you should undistort the frame *before* passing it to MediaPipe.

### Mirror flip

```python
frame = cv2.flip(frame, 1)
```

This flips the image horizontally before processing. Consequences:

- The display acts as a mirror — intuitive for a user facing the camera.
- MediaPipe's handedness classifier labels based on the *flipped* image. A user's right hand appears on the right side of the mirror and is classified as "Right." Without the flip, it would be classified as "Left" because it appears on the left side of the raw frame.
- **For robotics:** if your camera is not user-facing (e.g. mounted on a robot looking at an operator), you may want to remove the flip. In that case, swap the handedness labels or handle the x-axis inversion in your control mapping.

### OpenCV BGR vs RGB

OpenCV captures and displays in **BGR** channel order. MediaPipe expects **RGB**. The conversion:

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

This is a pure channel swap (no data copy on most platforms) and adds negligible latency. However, it's a common source of bugs — if you forget this conversion, the model will still produce landmarks, but accuracy degrades because the neural network was trained on RGB data.

---

## 6. Complete Source Code with Annotations

```python
#!/usr/bin/env python3
"""Hand Gesture Recognizer – maps fingers and palm center in real-time."""

import cv2                          # OpenCV: capture, draw, display
import mediapipe as mp              # ML pipeline: hand detection + landmarks
import numpy as np                  # Numeric: centroid computation

# ── MediaPipe module aliases ──────────────────────────────────────────────
mp_hands = mp.solutions.hands       # Hand detection/landmark solution
mp_drawing = mp.solutions.drawing_utils  # Built-in skeleton renderer

# ── Fingertip landmark indices ────────────────────────────────────────────
# Only the TIP landmarks (end of each finger) get explicit labels.
FINGERTIPS = {
    4: "THUMB",
    8: "INDEX",
    12: "MIDDLE",
    16: "RING",
    20: "PINKY",
}

# ── Palm centroid landmarks ───────────────────────────────────────────────
# Wrist (0) + MCP base of each finger (1, 5, 9, 13, 17).
# These form the rigid palm structure whose centroid is stable
# regardless of finger pose.
PALM_INDICES = [0, 1, 5, 9, 13, 17]

# ── BGR colour palette ────────────────────────────────────────────────────
COLOR_LANDMARK   = (0, 255, 255)    # Yellow:  all 21 joint dots
COLOR_FINGERTIP  = (0, 255, 0)      # Green:   the 5 fingertip circles
COLOR_CENTER     = (255, 0, 255)    # Magenta: palm center dot
COLOR_CONNECTION = (255, 200, 0)    # Cyan:    bone lines
COLOR_LABEL      = (255, 255, 255)  # White:   text overlays
```

The remaining functions and main loop are detailed in Section 7 and 8 below.

---

## 7. Function-by-Function Deep Dive

### 7.1 `compute_palm_center(landmarks, w, h) → (int, int)`

```python
def compute_palm_center(landmarks, w: int, h: int) -> tuple[int, int]:
    xs = [landmarks[i].x * w for i in PALM_INDICES]
    ys = [landmarks[i].y * h for i in PALM_INDICES]
    return int(np.mean(xs)), int(np.mean(ys))
```

**Purpose:** Estimate the geometric center of the palm in pixel coordinates.

**Inputs:**
- `landmarks` — the `.landmark` attribute of a MediaPipe hand result. A list-like of 21 `NormalizedLandmark` objects.
- `w`, `h` — frame width and height in pixels (needed to denormalise).

**Algorithm:**
1. For each of the 6 palm indices, convert normalised x/y to pixel coordinates.
2. Compute the arithmetic mean of the 6 x-coordinates and 6 y-coordinates independently.
3. Cast to `int` for pixel-space rendering.

**Why these 6 points?**
Landmarks 0, 1, 5, 9, 13, 17 are the wrist and the five MCP (knuckle-base) joints. Together they form the rigid palm structure — these points move together as a unit regardless of finger articulation. If you averaged all 21 landmarks, the centroid would shift toward extended fingers, making it unstable as a spatial reference.

**Robotics relevance:**
The palm center is a useful reference frame for:
- **Teleop mapping** — map the palm center to an end-effector position.
- **Gesture anchor** — finger states can be defined relative to the palm center rather than absolute image coordinates.
- **Depth estimation** — the z-values of palm landmarks can give a rough estimate of hand distance from the camera (though monocular depth from MediaPipe is noisy).

### 7.2 `draw_hand(image, hand_landmarks, w, h) → None`

```python
def draw_hand(image, hand_landmarks, w: int, h: int) -> None:
    landmarks = hand_landmarks.landmark

    # Layer 1: Skeleton (all 21 joints + bone connections)
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=COLOR_LANDMARK, thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=COLOR_CONNECTION, thickness=2),
    )

    # Layer 2: Fingertip highlights
    for idx, name in FINGERTIPS.items():
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 8, COLOR_FINGERTIP, cv2.FILLED)
        cv2.putText(image, name, (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LABEL, 1, cv2.LINE_AA)

    # Layer 3: Palm center
    pcx, pcy = compute_palm_center(landmarks, w, h)
    cv2.circle(image, (pcx, pcy), 10, COLOR_CENTER, cv2.FILLED)
    cv2.putText(image, "CENTER", (pcx + 12, pcy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CENTER, 1, cv2.LINE_AA)
```

**Purpose:** Render the full hand visualisation as three stacked layers.

**Rendering order matters.** Layer 1 (skeleton) is drawn first so that the larger fingertip circles (layer 2) and the palm center (layer 3) are drawn *on top* and remain visible.

**`mp_drawing.draw_landmarks()` internals:**
This MediaPipe utility iterates over `HAND_CONNECTIONS`, converts normalised coordinates to pixel space, and draws:
- A filled circle at each of the 21 landmarks.
- A line segment for each connection.

The two `DrawingSpec` arguments control the style for landmarks (dots) and connections (lines) respectively.

**`cv2.LINE_AA`:** Anti-aliased line drawing. Without this flag, text and circles have jagged pixelated edges. The cost is marginal (~5% slower than `cv2.LINE_8`) and well worth it for visual clarity.

**`cv2.FILLED` vs thickness:** `cv2.FILLED` fills the circle entirely. Using a positive thickness value would draw only the outline.

### 7.3 `count_fingers_up(landmarks) → int`

```python
def count_fingers_up(landmarks) -> int:
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    count = 0

    # Thumb
    if landmarks[4].x < landmarks[3].x:
        count += 1

    # Index through pinky
    for tip, pip_ in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip_].y:
            count += 1

    return count
```

**Purpose:** Estimate how many fingers are extended (open).

**Algorithm — four fingers (index, middle, ring, pinky):**

```
finger is extended  ⟺  tip.y < pip.y
```

In image coordinates, y=0 is the top of the frame and increases downward. When a finger is extended (pointing up), its TIP landmark is *above* (lower y-value than) its PIP joint. When the finger is curled, the TIP drops below the PIP.

This comparison uses the **PIP** joint (indices 6, 10, 14, 18) rather than the MCP because:
- MCP is at the knuckle base and doesn't move much when fingers curl.
- PIP is the first bend joint — it's the clearest inflection point between "open" and "closed."
- Using DIP would be too sensitive to slight finger curl.

**Algorithm — thumb:**

```
thumb is extended  ⟺  tip.x < ip.x
```

The thumb extends *laterally* (side-to-side) rather than vertically. Comparing y-coordinates would fail because the thumb tip and IP joint can be at similar heights in many poses. The x-comparison works because:
- When the thumb is open, the tip (landmark 4) extends outward, away from the palm.
- When closed, the tip tucks inward, past the IP joint (landmark 3).

**Critical caveat:** The `<` comparison assumes the hand is a **right hand facing the camera in the mirrored (flipped) view**. For a left hand, the thumb extends in the opposite x-direction. The code does not account for this — it will miscount the thumb on left hands in some orientations. A robust version would check `hand_info.classification[0].label` and flip the comparison accordingly.

**Accuracy envelope:**
- Front-facing, palm toward camera, fingers pointing up: ~95% accurate.
- Angled views (30°+ rotation): degrades to ~70%.
- Fist with one finger extended: works well.
- Sideways hand (knife-edge): fails — all y-comparisons break down.

---

## 8. The Main Loop — Frame Acquisition to Display

```python
def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
```

`cv2.VideoCapture(0)` opens the default camera. The `0` is the device index. On systems with multiple cameras, you may need `1`, `2`, etc. On Linux, this maps to `/dev/video0`. OpenCV selects the backend automatically (V4L2 on Linux, AVFoundation on macOS, DirectShow on Windows).

`isOpened()` check: cameras can fail to open if another process holds the device, the device doesn't exist, or permissions are insufficient. Always guard against this.

```python
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
```

**`static_image_mode=False`** — the most important parameter for real-time use. When False, MediaPipe enters **tracking mode**:
- Frame N: palm detector runs → finds hand → landmark model runs → outputs keypoints + confidence.
- Frame N+1: if confidence ≥ `min_tracking_confidence`, **skip** palm detection → run landmark model on the region predicted from frame N's landmarks.
- If confidence drops: re-run palm detection.

This means the expensive palm detector runs only on the first detection and on re-detections, while the cheaper landmark model runs every frame. This is what makes 30 fps possible on CPU.

**`max_num_hands=2`** — caps the number of simultaneously tracked hands. Each additional hand adds ~5ms of landmark inference. For robotics teleoperation with a single arm, set to 1.

**`min_detection_confidence=0.7`** — the palm detector must be at least 70% confident to register a new hand. Higher values reduce false detections but may miss hands at unusual angles.

**`min_tracking_confidence=0.5`** — if the landmark model's "hand presence" score drops below 50%, tracking is abandoned and the palm detector re-runs. Lower values keep tracking longer (fewer re-detections, smoother) but may track ghost hands. Higher values are more responsive but cause more jitter during re-detection transitions.

```python
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
```

**Frame acquisition:** `cap.read()` returns `(success_bool, frame_array)`. The frame is a numpy array of shape `(height, width, 3)` in BGR uint8 format.

**`hands.process(rgb)`** runs the full two-stage pipeline. The input **must** be RGB. The method is synchronous — it blocks until inference completes.

**Return value (`results`):**
- `results.multi_hand_landmarks` — list of `NormalizedHandLandmarks`, one per detected hand. `None` if no hands found.
- `results.multi_handedness` — list of `Classification` objects with `.label` ("Left"/"Right") and `.score` (confidence). Parallel with `multi_hand_landmarks`.

```python
            if results.multi_hand_landmarks:
                for hand_lm, hand_info in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):
                    draw_hand(frame, hand_lm, w, h)

                    label = hand_info.classification[0].label
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
```

The `None` guard on `multi_hand_landmarks` is essential — without it, the `zip()` would throw a `TypeError` on frames where no hands are detected.

The handedness label text is positioned relative to the palm center (offset 60px left, 30px below) so it stays near the hand it refers to, even when multiple hands are present.

```python
            cv2.imshow("Hand Gesture Recognizer (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
```

**`cv2.waitKey(1)`:** Waits 1 millisecond for a keypress. This does two things:
1. Processes the GUI event loop (without this call, `imshow` windows won't render or respond).
2. Returns the pressed key code, or -1 if no key was pressed.

**`& 0xFF`:** On some platforms, `waitKey` returns a 32-bit value with platform-specific flags in the upper bits. The bitmask isolates the lower 8 bits (the actual ASCII code).

**Cleanup:** `cap.release()` frees the camera device. `destroyAllWindows()` closes all OpenCV GUI windows. These run even if the loop exits due to a camera read failure (not just `q` press), because they're outside the loop but inside `main()`.

---

## 9. Finger Extension Detection — Geometry and Limitations

### The geometric model

The finger extension heuristic treats each finger as a 2D kinematic chain and uses a single inequality to determine state:

```
                TIP (8)
                 │
                DIP (7)     ← not used in the comparison
                 │
                PIP (6)     ← reference joint
                 │
                MCP (5)     ← not used in the comparison
```

**Extended:** TIP is above PIP in image space (tip.y < pip.y).
**Curled:** TIP is below PIP in image space (tip.y ≥ pip.y).

### Why this works

When you extend your index finger and point it upward, the tip is physically higher than the PIP joint. In image coordinates (where y increases downward), "higher" means a smaller y-value. The inequality `tip.y < pip.y` captures exactly this spatial relationship.

### Why the thumb is different

The thumb's range of motion is dominated by **abduction/adduction** (opening/closing in the plane of the palm) rather than **flexion/extension** (curling). In a front-facing hand:
- An open thumb extends laterally away from the index finger.
- A closed thumb tucks across or beside the palm.

The lateral motion means the y-coordinates of the thumb tip and IP joint are often similar regardless of state. The x-coordinate is the discriminating axis.

### Failure cases

1. **Hand rotation:** If the hand rotates 90° (knife-edge), all fingers move laterally rather than vertically. The y-comparison for the four fingers breaks down entirely.
2. **Hand inversion:** If the hand is upside down (fingers pointing downward), extended fingers have *higher* y-values than their PIP joints — the heuristic reports them as curled.
3. **Thumb on left hand:** The x-comparison assumes the thumb extends in the negative-x direction. For a left hand (in mirrored view), the thumb extends in the positive-x direction. The comparison needs to be flipped.
4. **Partial occlusion:** If one finger crosses behind another, the landmark model may estimate incorrect positions, causing transient miscounts.

### How to improve

For a production robotics system, replace the y/x heuristic with **joint angle calculation:**

```python
# Angle at PIP joint between MCP→PIP and PIP→TIP vectors
vec1 = (mcp.x - pip.x, mcp.y - pip.y)
vec2 = (tip.x - pip.x, tip.y - pip.y)
angle = math.degrees(math.atan2(
    vec1[0]*vec2[1] - vec1[1]*vec2[0],
    vec1[0]*vec2[0] + vec1[1]*vec2[1]
))
is_extended = angle > threshold
```

This approach is rotation-invariant because it measures the angle *between bones* rather than absolute screen position.

---

## 10. Palm Center Estimation

### Point selection rationale

```python
PALM_INDICES = [0, 1, 5, 9, 13, 17]
```

These 6 points define the rigid palm:

```
         5 ─── 9 ─── 13 ─── 17     ← MCP bridge (knuckle line)
        ╱                      ╲
       1                        ╲
        ╲                        │
         0 ───────────────────────   ← Wrist
```

**Why not use all 21 landmarks?** If you average all 21 points, the centroid shifts toward whichever fingers are extended. Make a fist: centroid is near the palm. Open all fingers: centroid jumps toward the fingertips. This makes it unreliable as a spatial anchor.

The 6 palm points are biomechanically rigid — they move as a unit regardless of finger articulation. Their centroid stays in approximately the same place whether the hand is open or closed.

### Centroid vs. geometric center

The arithmetic mean of the 6 points is a **centroid**, not a true geometric center of the palm surface. It's biased slightly toward the thumb side because landmark 1 (thumb CMC) is included. For most applications this bias is negligible, but if you need the true palm center, you could:
- Use only the four MCP joints (5, 9, 13, 17) for a knuckle-line centroid.
- Compute the centroid of the polygon formed by (0, 5, 17) for a triangular palm approximation.

---

## 11. Rendering Pipeline

The visual output is composed of 4 layers, drawn in order:

```
Layer 1: Skeleton        — mp_drawing.draw_landmarks() — 21 yellow dots + cyan bone lines
Layer 2: Fingertip dots  — cv2.circle() × 5            — green filled circles (radius 8)
Layer 3: Palm center     — cv2.circle() × 1            — magenta filled circle (radius 10)
Layer 4: Text overlays   — cv2.putText() × 7           — 5 fingertip names + "CENTER" + hand info
```

Later layers occlude earlier layers. This is why fingertip dots (layer 2) are drawn after the skeleton (layer 1) — the larger green circle covers the smaller yellow dot, creating a clear visual hierarchy.

### Drawing functions reference

**`cv2.circle(image, center, radius, color, thickness)`**
- `center`: (x, y) in pixels.
- `radius`: in pixels.
- `thickness`: positive integer for outline; `cv2.FILLED` (-1) for solid.
- Modifies `image` in-place (no return value needed).

**`cv2.putText(image, text, origin, font, scale, color, thickness, line_type)`**
- `origin`: bottom-left corner of the text string.
- `scale`: font scale factor (1.0 = base size, which is font-dependent).
- `line_type`: `cv2.LINE_AA` for anti-aliased, `cv2.LINE_8` for aliased.
- Text is rendered into the image buffer directly — there is no background. If the text overlaps a busy region, it may be hard to read. A production version might draw a semi-transparent rectangle behind text.

**`cv2.rectangle(image, pt1, pt2, color, thickness)`**
- Not used in this project, but commonly used in companion face-detection projects.

---

## 12. Building This From Scratch — Step by Step

### Phase 1: Minimal webcam loop (10 min)

Get a live camera feed with MediaPipe hand skeleton:

```python
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS
                )
        cv2.imshow("Hands", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
```

At this point you have a working hand tracker. Everything else is post-processing on the landmark data.

### Phase 2: Add mirror flip (1 min)

Add `frame = cv2.flip(frame, 1)` immediately after `cap.read()`. This makes the display intuitive.

### Phase 3: Label fingertips (10 min)

Define the `FINGERTIPS` dict. Loop over it, convert normalised coords to pixels, draw circles and text. This forces you to understand the landmark indexing scheme.

### Phase 4: Compute palm center (5 min)

Define `PALM_INDICES`. Write `compute_palm_center()`. Draw the result. This teaches you about the rigid vs. articulated parts of the hand model.

### Phase 5: Add finger counting (15 min)

Write `count_fingers_up()`. This is where you confront the coordinate system (y-down, x-right) and the thumb's lateral motion. Test with different hand poses. You'll quickly discover the failure cases listed in Section 9.

### Phase 6: Add handedness display (5 min)

Access `results.multi_handedness`, extract the label, render it near the palm center.

### Phase 7: Polish (15 min)

- Define a colour palette.
- Tune confidence thresholds.
- Customise `DrawingSpec` for line thickness and dot radius.
- Add the `isOpened()` guard and cleanup.

**Total from zero to finished: ~1 hour** for an engineer familiar with Python and OpenCV.

---

## 13. Latency Budget and Real-Time Constraints

For a 30 fps target, the total per-frame budget is **33.3 ms**.

Approximate breakdown on a modern laptop CPU (Intel i5 / Apple M1):

```
Camera read + decode:          2–5 ms
cv2.flip + cvtColor:           < 1 ms
Palm detection (when active):  8–12 ms
Landmark inference:            5–8 ms
Post-processing + drawing:     1–2 ms
cv2.imshow + waitKey:          1–2 ms
──────────────────────────────────────
Steady state (tracking):       10–18 ms  → 55–100 fps achievable
Re-detection frame:            18–30 ms  → may drop to 33 fps briefly
```

The key insight: **the palm detector is the bottleneck**, and tracking mode (`static_image_mode=False`) means it runs infrequently. This is why MediaPipe achieves real-time performance without a GPU.

### Latency vs. throughput

For robotics control, **latency** (time from photon to data) matters more than throughput (frames per second). The camera itself adds 33–66 ms of latency (1–2 frame buffers). MediaPipe adds 5–12 ms. OpenCV's display pipeline adds 1–2 ms. Total glass-to-glass: ~50–80 ms. This is acceptable for teleoperation but may be too slow for high-frequency closed-loop control.

---

## 14. Robotics Integration Considerations

### Decoupling perception from rendering

For a robotics use case, you don't need the OpenCV display at all. Strip out the rendering and extract raw landmark data:

```python
results = hands.process(rgb)
if results.multi_hand_landmarks:
    for hand_lm in results.multi_hand_landmarks:
        for i, lm in enumerate(hand_lm.landmark):
            print(f"Landmark {i}: x={lm.x:.4f} y={lm.y:.4f} z={lm.z:.4f}")
```

Publish this data to your control system (ROS topic, serial port, shared memory, WebSocket, etc.).

### Mapping to robot hand/gripper

**Simple gripper (1 DoF):** Map finger count to gripper aperture: 0 fingers = closed, 5 = fully open, linear interpolation in between.

**Multi-finger robot hand (5+ DoF):** Map each finger's extension angle (see Section 9 improvement) to the corresponding robot finger servo angle.

**Arm teleoperation:** Map the palm center (x, y) to end-effector position in the robot's workspace. The z-coordinate from MediaPipe can give coarse depth, but monocular depth is noisy — consider using a stereo camera or depth sensor (RealSense, ZED) for the z-axis.

### Frame rate and control frequency

MediaPipe at 30 fps gives a 33 ms control update period. For arm teleoperation this is adequate. For dexterous manipulation, you may need to interpolate between frames or use a predictive filter (Kalman, EMA) to smooth the landmark trajectory and run the control loop at a higher frequency.

### Filtering and smoothing

Raw MediaPipe landmarks exhibit frame-to-frame jitter (~2–5 pixels). For smooth control signals:

- **Exponential Moving Average (EMA):** `smoothed = α * new + (1 - α) * smoothed` with α ≈ 0.3–0.5.
- **One Euro Filter:** Adaptive low-pass that increases smoothing at low velocity and decreases it at high velocity. Ideal for hand tracking.
- **Kalman Filter:** If you need predicted state (position + velocity) for feedforward control.

MediaPipe has a built-in `SolutionOutputs` smoothing for video mode, but adding an explicit filter gives you control over the latency/smoothing tradeoff.

### Multi-camera setups

If your robot has multiple cameras, run a separate MediaPipe pipeline per camera. MediaPipe is not thread-safe — use multiprocessing (not threading) for parallel pipelines. Fuse the 2D landmarks from multiple views using triangulation to get true 3D hand pose.

---

## 15. Failure Modes and Edge Cases

| Scenario | Symptom | Cause | Mitigation |
|---|---|---|---|
| No hands detected | `multi_hand_landmarks` is None | Hand outside frame, bad lighting, low confidence | Lower `min_detection_confidence` to 0.5 |
| Jittery landmarks | Landmarks oscillate between frames | Marginal tracking confidence triggering re-detection | Lower `min_tracking_confidence` to 0.3; add smoothing filter |
| Wrong handedness | Left hand labelled "Right" | Mirror flip confuses classification; or hand is partially occluded | Check if flip is appropriate for your camera orientation |
| Thumb miscount | Reports 1 finger when thumb is closed (or vice versa) | x-comparison assumes right hand orientation | Use handedness label to flip thumb comparison |
| Ghost hand | Landmarks appear where no hand exists | False positive from palm detector on hand-like objects | Raise `min_detection_confidence` to 0.8 |
| Slow frame rate | < 15 fps | Palm detector re-running every frame | Ensure `static_image_mode=False`; check for CPU throttling |
| Camera won't open | `isOpened()` returns False | Device busy, wrong index, permission denied | Try different device index; check `ls /dev/video*` on Linux |
| Colour mismatch | Landmarks are inaccurate but present | Forgot BGR→RGB conversion | Ensure `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before `.process()` |

---

## 16. Extensions and Next Steps

### For robotics engineers specifically

1. **Joint angle extraction:** Compute the angle at each joint using the 3-point angle formula. Publish as a vector of 15 angles (3 per finger) for direct mapping to a robot hand.

2. **Wrist orientation:** Use landmarks 0, 5, and 17 to define a coordinate frame (origin at wrist, x-axis along palm, y-axis up). Extract roll, pitch, yaw for wrist-mounted end-effector control.

3. **Gesture state machine:** Define named gestures (fist, open, point, pinch, peace, etc.) as landmark-angle predicates. Run a simple state machine to debounce transitions and fire events.

4. **Depth integration:** Replace the monocular camera with an RGBD sensor. Use the depth channel for true z-position of each landmark. This eliminates the noisy monocular z-estimate.

5. **ROS integration:** Wrap the pipeline in a ROS node. Publish landmarks as a `sensor_msgs/JointState` or custom message. Subscribe to camera topics instead of using `cv2.VideoCapture`.

6. **Dual-hand coordination:** Track both hands and compute relative pose (distance between palm centers, synchronised finger states) for bimanual manipulation.

7. **Latency optimisation:** Run MediaPipe in a separate process, pipe landmarks via shared memory or ZMQ, and decouple the control loop from the camera frame rate.
