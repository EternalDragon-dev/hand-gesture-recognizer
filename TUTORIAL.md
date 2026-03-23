# Tutorial — Building a Hand Gesture Recognizer in Python

This tutorial walks through the architecture, ML pipeline, and implementation of the hand gesture recognizer. It assumes familiarity with Python and basic computer vision concepts.

---

## 1. Architecture Overview

This project is a single-file application (`main.py`) since all logic is tightly coupled around one real-time loop. The code is organised into four logical sections:

```
Constants & config     →  Landmark indices, colour palette
compute_palm_center()  →  Geometric calculation
draw_hand()            →  Rendering (landmarks, labels, center)
count_fingers_up()     →  Gesture heuristic
main()                 →  Webcam loop + MediaPipe integration
```

---

## 2. Core Concepts

### 2.1 MediaPipe Hands Pipeline

MediaPipe Hands is a two-stage ML pipeline:

**Stage 1 — Palm Detection:**
A lightweight SSD (Single Shot Detector) model locates hand bounding boxes in the frame. It's trained to detect palms rather than full hands because palms are rigid, roughly square, and easier to anchor-box than articulated fingers.

**Stage 2 — Hand Landmark Model:**
Given a cropped hand region, a regression model predicts **21 3D keypoints** (x, y, z for each). The model outputs normalised coordinates (0.0–1.0 relative to image dimensions).

The two-stage design is efficient: palm detection runs only when tracking is lost, while the landmark model runs every frame on the already-localised hand region.

### 2.2 The 21 Landmarks

```
        THUMB   INDEX   MIDDLE   RING   PINKY
TIP       4       8      12      16      20
DIP       3       7      11      15      19
PIP       2       6      10      14      18
MCP       1       5       9      13      17
                    WRIST: 0
```

- **MCP** = metacarpophalangeal (knuckle base)
- **PIP** = proximal interphalangeal (first bend)
- **DIP** = distal interphalangeal (second bend)
- **TIP** = fingertip

Each landmark has `.x`, `.y` (normalised to image size) and `.z` (depth relative to wrist).

### 2.3 Finger Extension Heuristic

To determine if a finger is "up" (extended), we use a simple geometric test:

**For the four fingers (index through pinky):**
```
finger is up  ⟺  tip.y < pip.y
```
In image coordinates, y increases downward, so a fingertip above its PIP joint means the finger is extended.

**For the thumb:**
```
thumb is up  ⟺  tip.x < ip.x    (for right hand facing camera)
```
The thumb extends laterally rather than vertically, so we compare x-coordinates instead. This works for a right hand facing the camera; the mirrored view means MediaPipe's handedness label handles the flip.

---

## 3. Code Walkthrough

### 3.1 Constants

```python
FINGERTIPS = {
    4: "THUMB",
    8: "INDEX",
    12: "MIDDLE",
    16: "RING",
    20: "PINKY",
}

PALM_INDICES = [0, 1, 5, 9, 13, 17]
```

`FINGERTIPS` maps landmark indices to display names — only the five tip landmarks need labels.

`PALM_INDICES` identifies the 6 points that define the palm: the wrist (0) and the MCP joint at the base of each finger (1, 5, 9, 13, 17). Their centroid gives a stable estimate of the palm center.

### 3.2 `compute_palm_center(landmarks, w, h)`

```python
xs = [landmarks[i].x * w for i in PALM_INDICES]
ys = [landmarks[i].y * h for i in PALM_INDICES]
return int(np.mean(xs)), int(np.mean(ys))
```

MediaPipe returns normalised coordinates (0.0–1.0). Multiplying by frame width/height converts to pixel coordinates. The mean of the 6 palm points gives a center that stays stable even when fingers move.

Why these 6 points? They form the rigid palm structure. Using all 21 points would bias the center toward whichever fingers are extended.

### 3.3 `draw_hand(image, hand_landmarks, w, h)`

This function layers three visual elements:

**1. Skeleton (landmarks + connections):**
```python
mp_drawing.draw_landmarks(
    image,
    hand_landmarks,
    mp_hands.HAND_CONNECTIONS,
    mp_drawing.DrawingSpec(color=COLOR_LANDMARK, thickness=2, circle_radius=3),
    mp_drawing.DrawingSpec(color=COLOR_CONNECTION, thickness=2),
)
```

`mp_hands.HAND_CONNECTIONS` is a predefined set of `(start, end)` index pairs representing the bones between joints. MediaPipe's drawing utility handles the conversion from normalised to pixel coordinates internally.

**2. Fingertip highlights:**
```python
for idx, name in FINGERTIPS.items():
    lm = landmarks[idx]
    cx, cy = int(lm.x * w), int(lm.y * h)
    cv2.circle(image, (cx, cy), 8, COLOR_FINGERTIP, cv2.FILLED)
    cv2.putText(image, name, (cx + 10, cy - 10), ...)
```

Larger green circles drawn on top of the skeleton at each fingertip, with text labels offset to the upper-right.

**3. Palm center:**
A magenta filled circle at the computed centroid, labelled "CENTER".

### 3.4 `count_fingers_up(landmarks)`

```python
# Thumb: lateral comparison
if landmarks[4].x < landmarks[3].x:
    count += 1

# Other fingers: vertical comparison
tips = [8, 12, 16, 20]
pips = [6, 10, 14, 18]
for tip, pip_ in zip(tips, pips):
    if landmarks[tip].y < landmarks[pip_].y:
        count += 1
```

This is intentionally simple. It works well for front-facing hands but can miscount for rotated or sideways hand poses. A production system would use the z-coordinate and hand orientation for more robust detection.

### 3.5 `main()` — The Webcam Loop

```python
cap = cv2.VideoCapture(0)
```
Opens the default webcam (device index 0).

```python
frame = cv2.flip(frame, 1)
```
Horizontal flip creates a mirror effect — when you move your right hand right, it moves right on screen. Without this, the display is counter-intuitive.

```python
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
) as hands:
```

Key parameters:
- `static_image_mode=False` — enables tracking mode. The palm detector only runs when confidence drops below `min_tracking_confidence`; otherwise, it reuses the previous detection and only runs the landmark model. This is much faster.
- `max_num_hands=2` — limits detection to 2 hands (reduces compute).
- `min_detection_confidence=0.7` — palm detection must be ≥70% confident to register a new hand.
- `min_tracking_confidence=0.5` — if tracking confidence drops below 50%, re-run palm detection.

```python
results = hands.process(rgb)
```
Runs the full ML pipeline on the RGB frame. Returns `multi_hand_landmarks` (list of 21-point landmark sets) and `multi_handedness` (Left/Right classification for each hand).

```python
for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
    draw_hand(frame, hand_lm, w, h)
    label = hand_info.classification[0].label
    fingers = count_fingers_up(hand_lm.landmark)
```

Each detected hand gets its own landmark set and handedness classification. We iterate over them in parallel with `zip`.

---

## 4. Building This From Scratch

### Step 1: Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

MediaPipe ships pre-built wheels — no compilation needed.

### Step 2: Get a basic webcam + MediaPipe loop working

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

This gives you the full skeleton with default styling in ~20 lines.

### Step 3: Add fingertip labels

Loop over indices `{4, 8, 12, 16, 20}`, convert normalised coords to pixels, draw circles and `putText`.

### Step 4: Add palm center

Average the normalised (x, y) of landmarks `[0, 1, 5, 9, 13, 17]`, convert to pixels, draw.

### Step 5: Add finger counting

Compare tip.y vs pip.y for each finger (thumb uses x). Display the count as overlay text.

### Step 6: Polish

- Add mirror flip (`cv2.flip`).
- Customise colours.
- Add handedness labels.
- Tune confidence thresholds.

---

## 5. Performance Considerations

- **No downscaling needed:** Unlike the face recognition project, MediaPipe is already optimised for real-time use. It runs its own internal resizing.
- **Tracking vs. detection:** The `static_image_mode=False` setting is critical. It means the expensive palm detector only runs occasionally (when tracking is lost), while the lightweight landmark model runs every frame.
- **GPU:** MediaPipe uses CPU by default but can leverage GPU on supported platforms via its internal delegate system. On macOS, it uses Apple's ML Compute when available.
- **Frame rate:** Expect 25–30+ fps on modern hardware. If performance is an issue, reduce `max_num_hands` to 1.

---

## 6. Known Limitations

- **Thumb detection:** The x-based heuristic for the thumb assumes a front-facing hand. It can misfire for rotated or profile-view hands.
- **Self-occlusion:** When fingers overlap (e.g. making a fist with one finger partially visible), the landmark model can produce noisy estimates.
- **Lighting:** Extreme backlighting or very dark environments degrade palm detection accuracy.
- **No gesture classification:** The app counts fingers but doesn't classify gestures (e.g. "peace sign", "thumbs up"). You'd need to add a gesture classifier on top of the landmarks — either rule-based (angle calculations) or ML-based (train on labeled landmark sequences).

---

## 7. Extension Ideas

- **Gesture classification:** Map landmark positions to named gestures using angle-based rules or a small neural network.
- **Air drawing:** Track the index fingertip across frames to draw on screen.
- **Volume/brightness control:** Map finger count or hand position to system controls.
- **Sign language recognition:** Train a model on landmark sequences for ASL/BSL alphabet letters.
