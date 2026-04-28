import SwiftUI

struct ContentView: View {

    @StateObject private var cameraManager = CameraManager()
    @StateObject private var handDetector  = HandPoseDetector()
    @StateObject private var gestureEngine = GestureEngine()

    // Gesture toast state
    @State private var lastGestureLabel: String = ""
    @State private var lastGestureIcon: String = ""
    @State private var showGestureToast = false

    // Fingertip trail buffer (normalised Vision coords)
    @State private var trailPoints: [CGPoint] = []
    private let maxTrailLength = 12
    @State private var framesWithoutHand = 0

    var body: some View {
        ZStack {
            // Live camera feed
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            // Hand landmark overlay (drawn on top of the camera)
            HandOverlayView(hands: handDetector.hands,
                            frameSize: handDetector.frameSize,
                            gesture: gestureEngine.gesture,
                            cursorPosition: gestureEngine.cursorPosition,
                            trailPoints: trailPoints)
                .ignoresSafeArea()

            // UI chrome
            VStack {
                titleBar

                // Gesture toast with SF Symbol icon
                if showGestureToast {
                    HStack(spacing: 8) {
                        if !lastGestureIcon.isEmpty {
                            Image(systemName: lastGestureIcon)
                                .font(.title3)
                        }
                        Text(lastGestureLabel)
                            .font(.title3.bold())
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color.black.opacity(0.7))
                    .clipShape(Capsule())
                    .transition(.scale.combined(with: .opacity))
                }

                Spacer()
                statusBar
            }
        }
        .onAppear {
            cameraManager.onFrameCaptured = { [weak handDetector] buffer in
                handDetector?.processFrame(buffer)
            }
            cameraManager.start()
        }
        .onDisappear {
            cameraManager.onFrameCaptured = nil
            cameraManager.stop()
        }
        .onChange(of: handDetector.hands) { _, newHands in
            // Feed the first detected hand into the gesture engine
            if let hand = newHands.first {
                gestureEngine.update(with: hand)
                framesWithoutHand = 0

                // Update fingertip trail with index tip position
                if let indexTip = hand.landmarks[.indexTip] {
                    trailPoints.append(indexTip)
                    if trailPoints.count > maxTrailLength {
                        trailPoints.removeFirst()
                    }
                }
            } else {
                framesWithoutHand += 1
                if framesWithoutHand > 5 {
                    trailPoints.removeAll()
                }
                gestureEngine.reset()
            }
        }
        .onChange(of: gestureEngine.gesture) { _, newGesture in
            guard newGesture != .none else { return }
            lastGestureLabel = newGesture.rawValue
            lastGestureIcon = gestureIcon(for: newGesture)
            withAnimation(.spring(duration: 0.25)) { showGestureToast = true }
            // Auto-dismiss after 1.2 seconds
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
                withAnimation(.easeOut(duration: 0.3)) { showGestureToast = false }
            }
        }
        .statusBarHidden()
    }

    // MARK: - Helpers

    private func gestureIcon(for gesture: HandGesture) -> String {
        switch gesture {
        case .swipeLeft:  return "arrow.left"
        case .swipeRight: return "arrow.right"
        case .swipeUp:    return "arrow.up"
        case .swipeDown:  return "arrow.down"
        case .pinch:      return "hand.pinch"
        case .point:      return "hand.point.up.left"
        case .none:       return ""
        }
    }

    // MARK: - Sub-views

    private var titleBar: some View {
        HStack(spacing: 6) {
            Image(systemName: "hand.raised.fingers.spread")
            Text("Hand Gesture Recognizer")
                .font(.headline)
        }
        .foregroundColor(.white)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.top, 8)
    }

    private var statusBar: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(handDetector.hands.isEmpty ? Color.red : Color.green)
                .frame(width: 8, height: 8)

            if cameraManager.cameraUnavailable {
                Text("Camera unavailable")
                    .font(.caption)
            } else if !cameraManager.isAuthorized {
                Text("Camera permission required")
                    .font(.caption)
            } else if handDetector.hands.isEmpty {
                Text("No hands detected")
                    .font(.caption)
            } else {
                let gLabel = gestureEngine.gesture == .none
                    ? "" : " · \(gestureEngine.gesture.rawValue)"
                Text("\(handDetector.hands.count) hand(s)\(gLabel)")
                    .font(.caption)
            }
        }
        .foregroundColor(.white)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.bottom, 8)
    }
}
