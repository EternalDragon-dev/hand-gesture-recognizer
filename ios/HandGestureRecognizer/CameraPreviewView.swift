import SwiftUI
import AVFoundation

/// Wraps an `AVCaptureVideoPreviewLayer` for use in SwiftUI.
struct CameraPreviewView: UIViewRepresentable {

    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        // Session binding is set once in makeUIView; nothing to update.
    }

    // MARK: - Backing UIView

    /// A plain `UIView` whose `layerClass` is `AVCaptureVideoPreviewLayer`,
    /// so the preview automatically resizes with Auto Layout.
    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer {
            // swiftlint:disable:next force_cast
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}
