import Foundation
import Vision

// MARK: - Joint angle definitions

/// Each finger has 3 measurable flexion angles (MCP, PIP, DIP).
/// The thumb uses CMC, MP, IP instead — same idea, different bone names.
///
/// The angle at a joint is computed from the three landmarks that bracket it:
///   proximal → joint → distal
/// using the law of cosines on vectors (proximal→joint) and (distal→joint).
struct JointAngleDefinition {
    let name: String
    let proximal: VNHumanHandPoseObservation.JointName
    let center:   VNHumanHandPoseObservation.JointName
    let distal:   VNHumanHandPoseObservation.JointName
}

/// All 15 joint angle definitions (3 per finger).
let allJointAngleDefinitions: [JointAngleDefinition] = [
    // Thumb (CMC → MP → IP → Tip)
    JointAngleDefinition(name: "thumb_cmc",  proximal: .wrist,    center: .thumbCMC, distal: .thumbMP),
    JointAngleDefinition(name: "thumb_mp",   proximal: .thumbCMC, center: .thumbMP,  distal: .thumbIP),
    JointAngleDefinition(name: "thumb_ip",   proximal: .thumbMP,  center: .thumbIP,  distal: .thumbTip),
    // Index (MCP → PIP → DIP → Tip)
    JointAngleDefinition(name: "index_mcp",  proximal: .wrist,    center: .indexMCP, distal: .indexPIP),
    JointAngleDefinition(name: "index_pip",  proximal: .indexMCP, center: .indexPIP, distal: .indexDIP),
    JointAngleDefinition(name: "index_dip",  proximal: .indexPIP, center: .indexDIP, distal: .indexTip),
    // Middle
    JointAngleDefinition(name: "middle_mcp", proximal: .wrist,      center: .middleMCP, distal: .middlePIP),
    JointAngleDefinition(name: "middle_pip", proximal: .middleMCP,  center: .middlePIP, distal: .middleDIP),
    JointAngleDefinition(name: "middle_dip", proximal: .middlePIP,  center: .middleDIP, distal: .middleTip),
    // Ring
    JointAngleDefinition(name: "ring_mcp",   proximal: .wrist,    center: .ringMCP, distal: .ringPIP),
    JointAngleDefinition(name: "ring_pip",   proximal: .ringMCP,  center: .ringPIP, distal: .ringDIP),
    JointAngleDefinition(name: "ring_dip",   proximal: .ringPIP,  center: .ringDIP, distal: .ringTip),
    // Little
    JointAngleDefinition(name: "little_mcp", proximal: .wrist,      center: .littleMCP, distal: .littlePIP),
    JointAngleDefinition(name: "little_pip", proximal: .littleMCP,  center: .littlePIP, distal: .littleDIP),
    JointAngleDefinition(name: "little_dip", proximal: .littlePIP,  center: .littleDIP, distal: .littleTip),
]

// MARK: - Angle computation

/// Compute all 15 flexion angles (in degrees, 0°–180°) from a landmarks dict.
/// Returns a dictionary keyed by angle name (e.g. "index_pip" → 142.3).
/// Joints with missing landmarks are omitted.
func computeJointAngles(
    from landmarks: [VNHumanHandPoseObservation.JointName: CGPoint]
) -> [String: Double] {
    var result: [String: Double] = [:]

    for def in allJointAngleDefinitions {
        guard let pA = landmarks[def.proximal],
              let pB = landmarks[def.center],
              let pC = landmarks[def.distal] else { continue }

        // Vectors: BA = proximal - center, BC = distal - center
        let baX = Double(pA.x - pB.x)
        let baY = Double(pA.y - pB.y)
        let bcX = Double(pC.x - pB.x)
        let bcY = Double(pC.y - pB.y)

        let dot = baX * bcX + baY * bcY
        let magBA = (baX * baX + baY * baY).squareRoot()
        let magBC = (bcX * bcX + bcY * bcY).squareRoot()

        guard magBA > 1e-9, magBC > 1e-9 else { continue }

        // Clamp to [-1, 1] to handle floating-point edge cases
        let cosAngle = min(max(dot / (magBA * magBC), -1.0), 1.0)
        let degrees = acos(cosAngle) * 180.0 / .pi

        result[def.name] = degrees
    }

    return result
}
