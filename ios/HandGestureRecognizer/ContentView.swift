import SwiftUI

struct ContentView: View {

    @StateObject private var cameraManager = CameraManager()
    @StateObject private var handDetector  = HandPoseDetector()

    var body: some View {
        ZStack {
            // Live camera feed
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            // Hand landmark overlay (drawn on top of the camera)
            HandOverlayView(hands: handDetector.hands,
                            frameSize: handDetector.frameSize)
                .ignoresSafeArea()

            // UI chrome
            VStack {
                titleBar
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
                Text("\(handDetector.hands.count) hand(s) detected")
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
