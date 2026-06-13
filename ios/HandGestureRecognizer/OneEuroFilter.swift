import Foundation

// MARK: - One Euro Filter
// Adaptive low-pass filter for real-time signal smoothing.
// Reference: Casiez, Roussel, Vogel – CHI 2012
// https://cristal.univ-lille.fr/~casiez/1euro/
//
// Smooths heavily when still (kills jitter), follows quickly when moving (low lag).

/// One Euro Filter for a single scalar signal.
struct OneEuroFilter {

    /// Minimum cutoff frequency (Hz). Lower = more smoothing when still.
    var minCutoff: Double

    /// Speed coefficient. Higher = less lag when moving fast.
    var beta: Double

    /// Cutoff frequency for the derivative filter (Hz).
    var dCutoff: Double

    private var prevFilteredX: Double?
    private var prevFilteredDx: Double = 0
    private var lastTime: Double?

    init(minCutoff: Double = 1.0, beta: Double = 0.007, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    /// Feed a new sample at the given timestamp (seconds) and return the filtered value.
    mutating func filter(_ x: Double, at timestamp: Double) -> Double {
        let dt: Double
        if let last = lastTime {
            dt = max(timestamp - last, 1e-6)
        } else {
            dt = 1.0 / 60.0 // assume ~60 fps on first frame
        }
        lastTime = timestamp

        // 1. Compute raw derivative
        let dx: Double
        if let prev = prevFilteredX {
            dx = (x - prev) / dt
        } else {
            dx = 0
        }

        // 2. Smooth the derivative
        let alphaDx = smoothingFactor(dt: dt, cutoff: dCutoff)
        let filteredDx = alphaDx * dx + (1 - alphaDx) * prevFilteredDx
        prevFilteredDx = filteredDx

        // 3. Adaptive cutoff: fast motion → higher cutoff → less smoothing
        let cutoff = minCutoff + beta * abs(filteredDx)

        // 4. Smooth the signal
        let alphaX = smoothingFactor(dt: dt, cutoff: cutoff)
        let filteredX: Double
        if let prev = prevFilteredX {
            filteredX = alphaX * x + (1 - alphaX) * prev
        } else {
            filteredX = x
        }
        prevFilteredX = filteredX

        return filteredX
    }

    mutating func reset() {
        prevFilteredX = nil
        prevFilteredDx = 0
        lastTime = nil
    }

    private func smoothingFactor(dt: Double, cutoff: Double) -> Double {
        let tau = 1.0 / (2.0 * .pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }
}

// MARK: - Convenience: filter a CGPoint (x, y independently)

/// Pair of OneEuroFilters for a 2D point.
struct OneEuroFilter2D {
    var xFilter: OneEuroFilter
    var yFilter: OneEuroFilter

    init(minCutoff: Double = 1.0, beta: Double = 0.007, dCutoff: Double = 1.0) {
        xFilter = OneEuroFilter(minCutoff: minCutoff, beta: beta, dCutoff: dCutoff)
        yFilter = OneEuroFilter(minCutoff: minCutoff, beta: beta, dCutoff: dCutoff)
    }

    mutating func filter(_ point: CGPoint, at timestamp: Double) -> CGPoint {
        CGPoint(
            x: xFilter.filter(Double(point.x), at: timestamp),
            y: yFilter.filter(Double(point.y), at: timestamp)
        )
    }

    mutating func reset() {
        xFilter.reset()
        yFilter.reset()
    }
}
