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

    // MARK: - Smoothing (OneEuroFilter per joint, per hand slot)

    /// Two hand slots (index 0 and 1), each mapping joint → filter.
    private var jointFilters: [[VNHumanHandPoseObservation.JointName: OneEuroFilter2D]] = [[:], [:]]

    // MARK: - Temporal debouncing for finger count

    private var fingerCountBuffers: [[Int]] = [[], []]  // rolling window per hand slot
    private let debounceWindow = 5
    private let debounceThreshold = 3

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

        let timestamp = CACurrentMediaTime()

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

        var detected: [HandData] = []
        for (index, observation) in observations.prefix(2).enumerated() {
            if let hand = extractHandData(from: observation, slotIndex: index, timestamp: timestamp) {
                detected.append(hand)
            }
        }

        // Reset filters for unused slots
        if observations.count < 2 {
            resetFilters(for: 1)
        }

        DispatchQueue.main.async { [weak self] in
            self?.hands = detected
        }
    }

    // MARK: - Extraction

    private func extractHandData(
        from observation: VNHumanHandPoseObservation,
        slotIndex: Int,
        timestamp: Double
    ) -> HandData? {
        var landmarks: [VNHumanHandPoseObservation.JointName: CGPoint] = [:]
        var zValues: [Float] = []

        for joint in Self.allJoints {
            guard let point = try? observation.recognizedPoint(joint),
                  point.confidence > 0.3 else { continue }

            // Apply OneEuroFilter to smooth each joint (x, y only)
            let raw = point.location
            if jointFilters[slotIndex][joint] == nil {
                jointFilters[slotIndex][joint] = OneEuroFilter2D(minCutoff: 1.0, beta: 0.007, dCutoff: 1.0)
            }
            let smoothed = jointFilters[slotIndex][joint]!.filter(raw, at: timestamp)
            landmarks[joint] = smoothed

            // Collect z-depth (relative, from Vision)
            zValues.append(Float(point.location.x))  // Vision z is on the point
            // Vision's VNRecognizedPoint doesn't expose z directly in 2D mode,
            // but we can approximate depth from the 3D hand pose observation.
        }

        // Need at least 15 of 21 landmarks for a usable detection
        guard landmarks.count >= 15 else { return nil }

        let chirality: HandData.Chirality
        switch observation.chirality {
        case .left:    chirality = .left
        case .right:   chirality = .right
        default:       chirality = .unknown
        }

        let rawFingers = countFingers(landmarks: landmarks)
        let debouncedFingers = debounceFingersCount(rawFingers, slot: slotIndex)

        // Compute joint angles
        let angles = computeJointAngles(from: landmarks)

        // Compute average z-depth from the observation's 3D points
        let avgDepth = extractAverageDepth(from: observation)

        return HandData(
            landmarks: landmarks,
            chirality: chirality,
            fingersExtended: debouncedFingers,
            jointAngles: angles,
            averageDepth: avgDepth
        )
    }

    /// Extract average relative z-depth from the observation.
    /// VNRecognizedPoint provides (x, y) in 2D but the hand pose observation
    /// can give us z via recognizedPoints. We use the z component from each point.
    private func extractAverageDepth(from observation: VNHumanHandPoseObservation) -> Float {
        var totalZ: Float = 0
        var count: Float = 0
        for joint in Self.allJoints {
            guard let point = try? observation.recognizedPoint(joint),
                  point.confidence > 0.3 else { continue }
            // VNRecognizedPoint.location is (x,y). The z is available via the
            // point's underlying data — for VNHumanHandPoseObservation in iOS 17+,
            // we approximate depth using the spread of landmarks as a proxy.
            // True 3D z requires ARKit. Here we use wrist-to-middle-tip distance
            // as a rough depth proxy (closer hand = larger spread).
            count += 1
        }
        // Proxy: use the bounding spread of landmarks as a depth indicator.
        // Larger spread = closer to camera (negative z), smaller = farther.
        if count > 0 {
            let xs = Self.allJoints.compactMap { try? observation.recognizedPoint($0) }
                .filter { $0.confidence > 0.3 }
                .map { Float($0.location.x) }
            let ys = Self.allJoints.compactMap { try? observation.recognizedPoint($0) }
                .filter { $0.confidence > 0.3 }
                .map { Float($0.location.y) }
            guard let minX = xs.min(), let maxX = xs.max(),
                  let minY = ys.min(), let maxY = ys.max() else { return 0 }
            let spread = max(maxX - minX, maxY - minY)
            // Map spread to depth: 0.4+ = very close (near), 0.15-0.4 = active, <0.15 = far
            // Convert to a z-like value where negative = close
            if spread > 0.35      { return -0.1 }   // near
            else if spread < 0.12 { return 0.1 }    // far
            else                  { return 0.0 }    // active
        }
        return 0
    }

    // MARK: - Filter management

    private func resetFilters(for slot: Int) {
        for key in jointFilters[slot].keys {
            jointFilters[slot][key]?.reset()
        }
        jointFilters[slot].removeAll()
        fingerCountBuffers[slot].removeAll()
    }

    // MARK: - Temporal debouncing

    private func debounceFingersCount(_ raw: Int, slot: Int) -> Int {
        fingerCountBuffers[slot].append(raw)
        if fingerCountBuffers[slot].count > debounceWindow {
            fingerCountBuffers[slot].removeFirst()
        }

        // Majority vote: return the count that appears >= threshold times
        let buf = fingerCountBuffers[slot]
        for candidate in Set(buf) {
            if buf.filter({ $0 == candidate }).count >= debounceThreshold {
                return candidate
            }
        }
        // No majority → return raw (will settle within a few frames)
        return raw
    }

    // MARK: - Finger counting

    /// Heuristic finger-extension count (mirrors Python `count_fingers_up`).
    private func countFingers(landmarks: [VNHumanHandPoseObservation.JointName: CGPoint]) -> Int {
        var count = 0

        // Thumb: extended when tip is further from wrist than IP joint.
        if let thumbTip = landmarks[.thumbTip],
           let thumbIP  = landmarks[.thumbIP],
           let wrist    = landmarks[.wrist] {
            let tipDist = hypot(thumbTip.x - wrist.x, thumbTip.y - wrist.y)
            let ipDist  = hypot(thumbIP.x  - wrist.x, thumbIP.y  - wrist.y)
            if tipDist > ipDist * 1.15 { count += 1 }
        }

        // Other four fingers: tip above PIP means extended.
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
