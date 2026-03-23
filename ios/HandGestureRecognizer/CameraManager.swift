import AVFoundation
import Combine

/// Manages the camera capture session and delivers video frames via a callback.
final class CameraManager: NSObject, ObservableObject {

    let session = AVCaptureSession()
    @Published var isAuthorized = false
    @Published var cameraUnavailable = false

    /// Called on a background queue for each captured video frame.
    var onFrameCaptured: ((CMSampleBuffer) -> Void)?

    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.handgesture.camera", qos: .userInteractive)

    // MARK: - Lifecycle

    func start() {
        checkAuthorization { [weak self] authorized in
            guard authorized, let self else { return }
            self.sessionQueue.async {
                self.configureSession()
                self.session.startRunning()
            }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
        }
    }

    // MARK: - Authorization

    private func checkAuthorization(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            DispatchQueue.main.async { self.isAuthorized = true }
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async { self?.isAuthorized = granted }
                completion(granted)
            }
        default:
            DispatchQueue.main.async {
                self.isAuthorized = false
                self.cameraUnavailable = true
            }
            completion(false)
        }
    }

    // MARK: - Session configuration

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .high

        // Front camera input
        guard let camera = AVCaptureDevice.default(
                .builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: camera)
        else {
            DispatchQueue.main.async { self.cameraUnavailable = true }
            session.commitConfiguration()
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        // Video output
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        // Rotate to portrait and mirror to match the preview layer
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = true
            }
        }

        session.commitConfiguration()
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        onFrameCaptured?(sampleBuffer)
    }
}
