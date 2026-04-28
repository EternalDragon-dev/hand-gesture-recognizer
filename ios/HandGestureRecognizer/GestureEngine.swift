import Foundation
import Vision
import QuartzCore

// MARK: - Gesture types

enum HandGesture: String {
    case none       = "None"
    case swipeLeft  = "← Swipe Left"
    case swipeRight = "Swipe Right →"
    case swipeUp    = "↑ Swipe Up"
    case swipeDown  = "Swipe Down ↓"
    case pinch      = "Pinch"
    case point      = "Point"
}

// MARK: - Gesture Engine

/// Analyses a rolling history of hand landmarks to detect dynamic gestures.
/// One instance per hand slot.
final class GestureEngine: ObservableObject {

    // MARK: Published state

    @Published var gesture: HandGesture = .none
    @Published var cursorPosition: CGPoint?   // normalised, set when pointing
    @Published var isClicking: Bool = false    // pinch while pointing
    @Published var pinchScale: CGFloat = 1.0  // ratio vs pinch-start distance

    // MARK: Configuration

    /// Minimum normalised displacement for a swipe (0–1 range).
    var swipeThreshold: CGFloat = 0.12
    /// Maximum frames to evaluate for a swipe gesture.
    var swipeWindowSize: Int = 18
    /// Pinch triggers when thumb–index distance < this (normalised).
    var pinchDistanceThreshold: CGFloat = 0.045

    // MARK: Internal state

    private struct Snapshot {
        let indexTip: CGPoint
        let thumbTip: CGPoint
        let fingersExtended: Int
        let timestamp: Double
    }

    private var history: [Snapshot] = []
    private let maxHistory = 25

    /// Distance at the moment the pinch started (for scale calculation).
    private var pinchStartDistance: CGFloat?
    /// Cooldown: ignore swipes for this many frames after one fires.
    private var swipeCooldown: Int = 0

    // MARK: - Public API

    /// Call once per frame with the latest hand data.
    func update(with hand: HandData) {
        guard let indexTip = hand.landmarks[.indexTip],
              let thumbTip = hand.landmarks[.thumbTip] else {
            resetTransient()
            return
        }

        let snap = Snapshot(
            indexTip: indexTip,
            thumbTip: thumbTip,
            fingersExtended: hand.fingersExtended,
            timestamp: CACurrentMediaTime()
        )
        history.append(snap)
        if history.count > maxHistory { history.removeFirst() }

        // Decrement cooldown
        if swipeCooldown > 0 { swipeCooldown -= 1 }

        // --- Detection priority: Pinch > Point > Swipe > None ---

        let thumbIndexDist = distance(thumbTip, indexTip)

        // 1. Pinch
        if thumbIndexDist < pinchDistanceThreshold {
            if pinchStartDistance == nil {
                pinchStartDistance = max(thumbIndexDist, 0.001)
            }
            pinchScale = thumbIndexDist / pinchStartDistance!

            // If also pointing (1 finger), it's a click
            if hand.fingersExtended <= 2 {
                gesture = .pinch
                isClicking = true
                cursorPosition = indexTip
            } else {
                gesture = .pinch
                isClicking = false
                cursorPosition = nil
            }
            return
        }

        // Pinch ended
        pinchStartDistance = nil
        pinchScale = 1.0
        isClicking = false

        // 2. Point (only index finger extended)
        if hand.fingersExtended == 1,
           let indexPIP = hand.landmarks[.indexPIP],
           indexTip.y > indexPIP.y { // index is actually extended (Vision: y up)
            gesture = .point
            cursorPosition = indexTip
            return
        }

        cursorPosition = nil

        // 3. Swipe (need open hand: ≥3 fingers, enough history, no cooldown)
        if swipeCooldown == 0,
           hand.fingersExtended >= 3,
           history.count >= swipeWindowSize {
            if let swipe = detectSwipe() {
                gesture = swipe
                swipeCooldown = swipeWindowSize // prevent rapid re-fire
                // Clear history so the same motion isn't detected again
                history.removeAll()
                return
            }
        }

        // 4. Nothing
        gesture = .none
    }

    /// Call when the hand disappears.
    func reset() {
        history.removeAll()
        resetTransient()
        pinchStartDistance = nil
        swipeCooldown = 0
    }

    // MARK: - Swipe detection

    private func detectSwipe() -> HandGesture? {
        let window = Array(history.suffix(swipeWindowSize))
        guard let first = window.first, let last = window.last else { return nil }

        let dx = last.indexTip.x - first.indexTip.x
        let dy = last.indexTip.y - first.indexTip.y  // Vision: y increases upward

        let absDx = abs(dx)
        let absDy = abs(dy)

        // Horizontal swipe: dx dominant
        if absDx > swipeThreshold && absDx > absDy * 1.4 {
            return dx > 0 ? .swipeRight : .swipeLeft
        }

        // Vertical swipe: dy dominant
        if absDy > swipeThreshold && absDy > absDx * 1.4 {
            return dy > 0 ? .swipeUp : .swipeDown
        }

        return nil
    }

    // MARK: - Helpers

    private func resetTransient() {
        gesture = .none
        cursorPosition = nil
        isClicking = false
        pinchScale = 1.0
    }

    private func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        hypot(a.x - b.x, a.y - b.y)
    }
}
