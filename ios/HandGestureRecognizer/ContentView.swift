import SwiftUI

struct ContentView: View {

    @StateObject private var cameraManager = CameraManager()
    @StateObject private var handDetector  = HandPoseDetector()
    @StateObject private var gestureEngine = GestureEngine()

    // Auto-dismiss timer for gesture toast
    @State private var lastGestureLabel: String = ""
    @State private var showGestureToast = false

    var body: some View {
        ZStack {
            // Live camera feed
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            // Hand landmark overlay (drawn on top of the camera)
            HandOverlayView(hands: handDetector.hands,
                            frameSize: handDetector.frameSize,
                            gesture: gestureEngine.gesture,
                            cursorPosition: gestureEngine.cursorPosition)
                .ignoresSafeArea()

            // UI chrome
            VStack {
                titleBar

                // Gesture toast
                if showGestureToast {
                    Text(lastGestureLabel)
                        .font(.title3.bold())
                        .foregroundColor(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(Color.black.opacity(0.7))
                        .clipShape(Capsule())
                        .transition(.move(edge: .top).combined(with: .opacity))
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
            } else {
                gestureEngine.reset()
            }
        }
        .onChange(of: gestureEngine.gesture) { _, newGesture in
            guard newGesture != .none else { return }
            lastGestureLabel = newGesture.rawValue
            withAnimation(.easeInOut(duration: 0.2)) { showGestureToast = true }
            // Auto-dismiss after 1 second
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                withAnimation(.easeInOut(duration: 0.3)) { showGestureToast = false }
            }
        }
        .statusBarHidden()
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
