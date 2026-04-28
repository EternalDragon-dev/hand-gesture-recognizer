import SwiftUI
import Vision

/// Draws hand landmarks, bone connections, fingertip labels, and palm center
/// on a transparent overlay that sits on top of the camera preview.
struct HandOverlayView: View {

    let hands: [HandData]
    let frameSize: CGSize
    var gesture: HandGesture = .none
    var cursorPosition: CGPoint? = nil

    // Colours matching the Python version (BGR → SwiftUI)
    private let landmarkColor  = Color.yellow
    private let fingertipColor = Color.green
    private let centerColor    = Color(red: 1, green: 0, blue: 1) // magenta
    private let labelColor     = Color.white
    private let trailColor     = Color(red: 0.3, green: 0.8, blue: 1.0) // light blue trail

    /// Skeleton colour changes based on gesture state.
    private var connectionColor: Color {
        switch gesture {
        case .pinch:                         return .orange
        case .point:                         return .blue
        case .swipeLeft, .swipeRight,
             .swipeUp, .swipeDown:           return .green
        case .none:                          return .cyan
        }
    }

    // MARK: - Fingertip trail state (persists across redraws via wrapper)

    /// External trail buffer — fed by ContentView.
    var trailPoints: [CGPoint]

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                // Draw trail first (behind hand)
                drawTrail(context: &context, size: size)

                for hand in hands {
                    drawHand(context: &context, hand: hand, size: size)
                }
                // Draw cursor reticle when pointing / clicking
                if let cursor = cursorPosition {
                    drawCursor(context: &context, at: cursor, size: size)
                }
            }
        }
    }

    // MARK: - Drawing

    private func drawHand(context: inout GraphicsContext, hand: HandData, size: CGSize) {
        // 1. Bone connections (cyan lines)
        for (from, to) in HandData.boneConnections {
            guard let p1 = hand.landmarks[from],
                  let p2 = hand.landmarks[to] else { continue }
            let sp1 = toScreen(p1, in: size)
            let sp2 = toScreen(p2, in: size)

            var path = Path()
            path.move(to: sp1)
            path.addLine(to: sp2)
            context.stroke(path, with: .color(connectionColor), lineWidth: 2)
        }

        // 2. All landmarks (small yellow dots)
        for (_, point) in hand.landmarks {
            let sp = toScreen(point, in: size)
            let rect = CGRect(x: sp.x - 3, y: sp.y - 3, width: 6, height: 6)
            context.fill(Path(ellipseIn: rect), with: .color(landmarkColor))
        }

        // 3. Fingertips (larger green circles + labels)
        for (joint, label) in HandData.fingertips {
            guard let point = hand.landmarks[joint] else { continue }
            let sp = toScreen(point, in: size)
            let rect = CGRect(x: sp.x - 8, y: sp.y - 8, width: 16, height: 16)
            context.fill(Path(ellipseIn: rect), with: .color(fingertipColor))

            context.draw(
                Text(label)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(labelColor),
                at: CGPoint(x: sp.x + 16, y: sp.y - 10)
            )
        }

        // 4. Palm center (magenta dot + "CENTER" label)
        if let center = hand.palmCenter {
            let sp = toScreen(center, in: size)
            let rect = CGRect(x: sp.x - 10, y: sp.y - 10, width: 20, height: 20)
            context.fill(Path(ellipseIn: rect), with: .color(centerColor))

            context.draw(
                Text("CENTER")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(centerColor),
                at: CGPoint(x: sp.x + 22, y: sp.y - 12)
            )
        }

        // 5. Handedness + finger count (below palm center)
        if let center = hand.palmCenter {
            let sp = toScreen(center, in: size)
            let info = "\(hand.chirality.rawValue) hand  |  Fingers: \(hand.fingersExtended)"

            // Shadow for readability
            context.draw(
                Text(info)
                    .font(.system(size: 15, weight: .bold))
                    .foregroundColor(.black),
                at: CGPoint(x: sp.x + 1, y: sp.y + 31)
            )
            context.draw(
                Text(info)
                    .font(.system(size: 15, weight: .bold))
                    .foregroundColor(labelColor),
                at: CGPoint(x: sp.x, y: sp.y + 30)
            )
        }
    }

    // MARK: - Fingertip trail

    private func drawTrail(context: inout GraphicsContext, size: CGSize) {
        guard trailPoints.count >= 2 else { return }

        let screenPoints = trailPoints.map { toScreen($0, in: size) }
        let count = screenPoints.count

        for i in 1..<count {
            let progress = CGFloat(i) / CGFloat(count)          // 0→1
            let opacity = 0.1 + progress * 0.9                  // 0.1→1.0
            let lineWidth = 1.0 + progress * 2.5                 // 1.0→3.5

            var segment = Path()
            segment.move(to: screenPoints[i - 1])
            segment.addLine(to: screenPoints[i])
            context.stroke(
                segment,
                with: .color(trailColor.opacity(opacity)),
                lineWidth: lineWidth
            )
        }

        // Glow dot at the newest point
        if let tip = screenPoints.last {
            let glow = CGRect(x: tip.x - 5, y: tip.y - 5, width: 10, height: 10)
            context.fill(Path(ellipseIn: glow), with: .color(trailColor))
        }
    }

    // MARK: - Cursor reticle

    private func drawCursor(context: inout GraphicsContext, at point: CGPoint, size: CGSize) {
        let sp = toScreen(point, in: size)
        let radius: CGFloat = 18
        let color: Color = gesture == .pinch ? .orange : .blue

        // Outer ring
        let ring = Path(ellipseIn: CGRect(x: sp.x - radius, y: sp.y - radius,
                                          width: radius * 2, height: radius * 2))
        context.stroke(ring, with: .color(color), lineWidth: 2.5)

        // Crosshair lines
        let len: CGFloat = 8
        var cross = Path()
        cross.move(to: CGPoint(x: sp.x - radius - len, y: sp.y))
        cross.addLine(to: CGPoint(x: sp.x - radius + 4, y: sp.y))
        cross.move(to: CGPoint(x: sp.x + radius - 4, y: sp.y))
        cross.addLine(to: CGPoint(x: sp.x + radius + len, y: sp.y))
        cross.move(to: CGPoint(x: sp.x, y: sp.y - radius - len))
        cross.addLine(to: CGPoint(x: sp.x, y: sp.y - radius + 4))
        cross.move(to: CGPoint(x: sp.x, y: sp.y + radius - 4))
        cross.addLine(to: CGPoint(x: sp.x, y: sp.y + radius + len))
        context.stroke(cross, with: .color(color), lineWidth: 2)

        // Center dot (filled when clicking)
        if gesture == .pinch {
            let dot = CGRect(x: sp.x - 5, y: sp.y - 5, width: 10, height: 10)
            context.fill(Path(ellipseIn: dot), with: .color(.orange))
        }
    }

    // MARK: - Coordinate conversion

    /// Converts a normalised Vision point (bottom-left origin) to screen
    /// coordinates, accounting for the `.resizeAspectFill` crop applied by
    /// the camera preview layer.
    private func toScreen(_ point: CGPoint, in viewSize: CGSize) -> CGPoint {
        guard frameSize.width > 0, frameSize.height > 0 else {
            // Fallback before frame size is known
            return CGPoint(x: point.x * viewSize.width,
                           y: (1 - point.y) * viewSize.height)
        }

        let frameAspect = frameSize.width / frameSize.height
        let viewAspect  = viewSize.width  / viewSize.height

        if frameAspect > viewAspect {
            // Frame is wider → sides are cropped
            let scale       = viewSize.height / frameSize.height
            let scaledWidth = frameSize.width * scale
            let offsetX     = (scaledWidth - viewSize.width) / 2
            return CGPoint(x: point.x * scaledWidth - offsetX,
                           y: (1 - point.y) * viewSize.height)
        } else {
            // Frame is taller → top/bottom are cropped
            let scale        = viewSize.width / frameSize.width
            let scaledHeight = frameSize.height * scale
            let offsetY      = (scaledHeight - viewSize.height) / 2
            return CGPoint(x: point.x * viewSize.width,
                           y: (1 - point.y) * scaledHeight - offsetY)
        }
    }
}
