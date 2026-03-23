import Vision
import AVFoundation
import Combine

/// Processes camera frames with Vision's hand pose detection and publishes results.
final class HandPoseDetector: ObservableObject {

    @Published var hands: [HandData] = []

    /// Frame dimensions (portrait, post-rotation) for coordinate conversion.
    @Published var frameSize: CGSize = .zero

    // All 21 joints we extract from each observation.
    private static let allJoints: [VNHumanHandPoseObservation.JointName] = [
        .wrist,
        .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
        .indexMCP, .indexPIP, .indexDIP, .indexTip,
        .middleMCP, .middlePIP, .middleDIP, .middleTip,
        .ringMCP, .ringPIP, .ringDIP, .ringTip,
        .littleMCP, .littlePIP, .littleDIP, .littleTip,
    ]

    // MARK: - Frame processing

    /// Call this from the camera callback (background queue).
    func processFrame(_ sampleBuffer: CMSampleBuffer) {
        // Read frame dimensions for coordinate mapping
        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let w = CVPixelBufferGetWidth(pixelBuffer)
            let h = CVPixelBufferGetHeight(pixelBuffer)
            let newSize = CGSize(width: w, height: h)
            if newSize != frameSize {
                DispatchQueue.main.async { [weak self] in
                    self?.frameSize = newSize
                }
            }
        }

        // Run hand pose request
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 2

        let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: .up)
        do {
            try handler.perform([request])
        } catch {
            DispatchQueue.main.async { [weak self] in self?.hands = [] }
            return
        }

        guard let observations = request.results, !observations.isEmpty else {
            DispatchQueue.main.async { [weak self] in self?.hands = [] }
            return
        }

        let detected = observations.compactMap { extractHandData(from: $0) }

        DispatchQueue.main.async { [weak self] in
            self?.hands = detected
        }
    }

    // MARK: - Extraction

    private func extractHandData(from observation: VNHumanHandPoseObservation) -> HandData? {
        var landmarks: [VNHumanHandPoseObservation.JointName: CGPoint] = [:]

        for joint in Self.allJoints {
            guard let point = try? observation.recognizedPoint(joint),
                  point.confidence > 0.3 else { continue }
            landmarks[joint] = point.location   // normalized, bottom-left origin
        }

        // Need at least 15 of 21 landmarks for a usable detection
        guard landmarks.count >= 15 else { return nil }

        let chirality: HandData.Chirality
        switch observation.chirality {
        case .left:    chirality = .left
        case .right:   chirality = .right
        default:       chirality = .unknown
        }

        let fingersUp = countFingers(landmarks: landmarks)

        return HandData(
            landmarks: landmarks,
            chirality: chirality,
            fingersExtended: fingersUp
        )
    }

    // MARK: - Finger counting

    /// Heuristic finger-extension count (mirrors Python `count_fingers_up`).
    private func countFingers(landmarks: [VNHumanHandPoseObservation.JointName: CGPoint]) -> Int {
        var count = 0

        // Thumb: extended when tip is further from wrist than IP joint.
        // This is orientation-independent (works regardless of mirroring/chirality).
        if let thumbTip = landmarks[.thumbTip],
           let thumbIP  = landmarks[.thumbIP],
           let wrist    = landmarks[.wrist] {
            let tipDist = hypot(thumbTip.x - wrist.x, thumbTip.y - wrist.y)
            let ipDist  = hypot(thumbIP.x  - wrist.x, thumbIP.y  - wrist.y)
            if tipDist > ipDist * 1.15 { count += 1 }
        }

        // Other four fingers: tip above PIP means extended.
        // In Vision coordinates y increases upward, so tip.y > pip.y = extended.
        let fingerPairs: [(VNHumanHandPoseObservation.JointName,
                           VNHumanHandPoseObservation.JointName)] = [
            (.indexTip,  .indexPIP),
            (.middleTip, .middlePIP),
            (.ringTip,   .ringPIP),
            (.littleTip, .littlePIP),
        ]

        for (tip, pip) in fingerPairs {
            if let tipPt = landmarks[tip], let pipPt = landmarks[pip] {
                if tipPt.y > pipPt.y { count += 1 }
            }
        }

        return count
    }
}
