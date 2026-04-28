import Foundation
import Vision

// MARK: - Depth zone

enum DepthZone: String {
    case near   = "Near"    // hand very close to camera
    case active = "Active"  // optimal interaction range
    case far    = "Far"     // hand too far for reliable gestures
}

/// Represents a single detected hand with landmark positions and metadata.
struct HandData: Identifiable {
    let id = UUID()

    /// Landmark positions in normalized Vision coordinates (0–1, origin bottom-left).
    let landmarks: [VNHumanHandPoseObservation.JointName: CGPoint]

    /// Left / Right / Unknown.
    let chirality: Chirality

    /// Number of fingers currently extended.
    let fingersExtended: Int

    /// Flexion angles (degrees, 0–180) for up to 15 joints. Keyed by name, e.g. "index_pip".
    let jointAngles: [String: Double]

    /// Average relative z-depth across all detected landmarks (from Vision).
    /// Negative = closer to camera, positive = farther.
    let averageDepth: Float

    /// Depth zone derived from averageDepth.
    var depthZone: DepthZone {
        if averageDepth < -0.05      { return .near }
        else if averageDepth > 0.05  { return .far }
        else                         { return .active }
    }

    // MARK: - Chirality

    enum Chirality: String {
        case left = "Left"
        case right = "Right"
        case unknown = "Unknown"
    }

    // MARK: - Fingertip definitions (matches Python FINGERTIPS dict)

    static let fingertips: [(joint: VNHumanHandPoseObservation.JointName, label: String)] = [
        (.thumbTip,  "THUMB"),
        (.indexTip,  "INDEX"),
        (.middleTip, "MIDDLE"),
        (.ringTip,   "RING"),
        (.littleTip, "PINKY"),
    ]

    // MARK: - Palm center joints (matches Python PALM_INDICES)

    /// Wrist + base of each finger, used to compute the palm centroid.
    static let palmJoints: [VNHumanHandPoseObservation.JointName] = [
        .wrist, .thumbCMC, .indexMCP, .middleMCP, .ringMCP, .littleMCP,
    ]

    // MARK: - Bone connections for skeleton drawing

    static let boneConnections: [(VNHumanHandPoseObservation.JointName,
                                  VNHumanHandPoseObservation.JointName)] = [
        // Thumb
        (.wrist, .thumbCMC), (.thumbCMC, .thumbMP), (.thumbMP, .thumbIP), (.thumbIP, .thumbTip),
        // Index
        (.wrist, .indexMCP), (.indexMCP, .indexPIP), (.indexPIP, .indexDIP), (.indexDIP, .indexTip),
        // Middle
        (.wrist, .middleMCP), (.middleMCP, .middlePIP), (.middlePIP, .middleDIP), (.middleDIP, .middleTip),
        // Ring
        (.wrist, .ringMCP), (.ringMCP, .ringPIP), (.ringPIP, .ringDIP), (.ringDIP, .ringTip),
        // Little
        (.wrist, .littleMCP), (.littleMCP, .littlePIP), (.littlePIP, .littleDIP), (.littleDIP, .littleTip),
        // Palm bridge
        (.indexMCP, .middleMCP), (.middleMCP, .ringMCP), (.ringMCP, .littleMCP),
    ]

    // MARK: - Computed helpers

    /// Palm center in normalized coordinates, or `nil` if not enough landmarks.
    var palmCenter: CGPoint? {
        let points = Self.palmJoints.compactMap { landmarks[$0] }
        guard points.count == Self.palmJoints.count else { return nil }
        let avgX = points.map(\.x).reduce(0, +) / CGFloat(points.count)
        let avgY = points.map(\.y).reduce(0, +) / CGFloat(points.count)
        return CGPoint(x: avgX, y: avgY)
    }
}
