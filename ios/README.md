# Hand Gesture Recognizer – iOS

> Native iOS port of the [Python hand-gesture-recognizer](https://github.com/EternalDragon-dev/hand-gesture-recognizer), built with SwiftUI, AVFoundation, and Apple's Vision framework.

[![Swift](https://img.shields.io/badge/Swift-5.0-orange)](https://swift.org)
[![iOS](https://img.shields.io/badge/iOS-17.0%2B-blue)](https://developer.apple.com/ios/)
[![Xcode](https://img.shields.io/badge/Xcode-15%2B-blue)](https://developer.apple.com/xcode/)

## Features

All features from the Python version, running natively on iOS:

- **21-point hand landmark detection** — full skeletal map via Apple Vision framework
- **Fingertip labels** — THUMB, INDEX, MIDDLE, RING, PINKY highlighted in green
- **Palm center point** — computed from wrist + finger base joints, shown in magenta
- **Finger counting** — detects how many fingers are extended per hand
- **Handedness detection** — identifies Left vs. Right hand (Vision `chirality`)
- **Multi-hand support** — tracks up to 2 hands simultaneously
- **Mirror view** — front camera with natural mirror perspective
- **Status indicator** — live detection status at the bottom of the screen

### iOS-specific additions

- No external dependencies — uses only Apple frameworks (Vision, AVFoundation, SwiftUI)
- Proper `.resizeAspectFill` coordinate mapping so overlays align perfectly with the camera feed
- Camera permission handling with user-friendly status messages
- Runs on iPhone and iPad

## Requirements

- **Xcode 15** or later
- **iOS 17.0+** device (camera required — the Simulator does not have a camera)
- An Apple Developer account (free tier works for on-device testing)

## Setup

1. **Open the project in Xcode:**
   ```
   open HandGestureRecognizer.xcodeproj
   ```

2. **Select your development team:**
   - Open the project settings (click the blue project icon in the navigator)
   - Select the **HandGestureRecognizer** target
   - Under **Signing & Capabilities**, choose your team from the dropdown

3. **Connect your iPhone/iPad** and select it as the run destination.

4. **Build & Run** (`Cmd + R`).

5. **Grant camera access** when prompted.

Hold your hand(s) in front of the camera — you'll see coloured landmarks, bone connections, fingertip labels, a palm center dot, and a finger count overlay.

## Project Structure

```
HandGestureRecognizer/
├── HandGestureRecognizerApp.swift  # @main entry point
├── ContentView.swift               # Root view: camera + overlay + UI chrome
├── CameraManager.swift             # AVCaptureSession, front camera, frame delivery
├── HandPoseDetector.swift          # Vision hand pose detection + finger counting
├── CameraPreviewView.swift         # UIViewRepresentable for camera preview layer
├── HandOverlayView.swift           # SwiftUI Canvas drawing landmarks & labels
├── HandData.swift                  # Data model: landmarks, chirality, connections
└── Assets.xcassets/                # App icon & accent colour
```

## Architecture

```
┌───────────────┐   CMSampleBuffer   ┌──────────────────┐
│ CameraManager │ ─────────────────▶ │ HandPoseDetector  │
│ (AVFoundation)│                    │ (Vision framework)│
└───────────────┘                    └────────┬─────────┘
                                              │ @Published [HandData]
        ┌─────────────────────────────────────┘
        ▼
┌───────────────────────────────────────────────────┐
│                   ContentView                      │
│  ┌─────────────────┐  ┌────────────────────────┐  │
│  │ CameraPreviewView│  │    HandOverlayView     │  │
│  │ (preview layer)  │  │ (Canvas: bones, tips,  │  │
│  │                  │  │  center, finger count) │  │
│  └─────────────────┘  └────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

1. **CameraManager** captures frames from the front camera and delivers them via a callback.
2. **HandPoseDetector** runs `VNDetectHumanHandPoseRequest` on each frame, extracts 21 landmarks per hand, determines chirality, and counts extended fingers.
3. **HandOverlayView** converts normalised Vision coordinates to screen coordinates (accounting for `.resizeAspectFill` cropping) and draws everything using SwiftUI `Canvas`.

## Python → iOS Mapping

| Python (MediaPipe)              | iOS (Vision)                                     |
|---------------------------------|--------------------------------------------------|
| `mp.solutions.hands`            | `VNDetectHumanHandPoseRequest`                   |
| `mp_drawing.draw_landmarks`     | Custom `Canvas` drawing in `HandOverlayView`     |
| `cv2.VideoCapture(0)`           | `AVCaptureSession` with front camera             |
| `results.multi_handedness`      | `VNHumanHandPoseObservation.chirality`           |
| `count_fingers_up()` heuristic  | Same tip-vs-joint logic in `HandPoseDetector`    |
| `compute_palm_center()`         | `HandData.palmCenter` computed property          |
| OpenCV window + `waitKey`       | SwiftUI `ZStack` with live preview               |

## Customisation

### Detection confidence
In `HandPoseDetector.swift`, adjust the confidence threshold (default `0.3`):
```swift
point.confidence > 0.3
```

### Colours
In `HandOverlayView.swift`, change the overlay colours to match your preference:
```swift
private let landmarkColor   = Color.yellow
private let fingertipColor  = Color.green
private let centerColor     = Color(red: 1, green: 0, blue: 1)
private let connectionColor = Color.cyan
```

### Max hands
In `HandPoseDetector.swift`:
```swift
request.maximumHandCount = 2  // change to 1 for single-hand only
```

## Troubleshooting

**"Camera unavailable" on Simulator**
Vision hand pose detection requires a real camera. Use a physical device.

**Landmarks misaligned**
Make sure the camera preview and overlay both use `.ignoresSafeArea()`. The coordinate conversion in `HandOverlayView.toScreen()` accounts for aspect-fill cropping.

**Slow detection**
Vision hand pose runs on the Neural Engine and should be fast (~30 fps). If it's slow, check that you're running a Release build and not in the debugger.

## License

MIT License — feel free to use and modify!

## See Also

- [Python hand-gesture-recognizer](https://github.com/EternalDragon-dev/hand-gesture-recognizer) — the original project
- [Apple Vision documentation](https://developer.apple.com/documentation/vision/detecting_hand_poses_with_vision)
