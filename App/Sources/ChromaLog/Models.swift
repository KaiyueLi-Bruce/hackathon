import SwiftUI

/// A detected/placed spot, stored in normalized image coordinates (0...1, y
/// increasing downward) so it is resolution independent.
struct Spot: Identifiable, Hashable {
    let id = UUID()
    var point: CGPoint
    var label: SpotLabel
    var note: String = ""
}

/// Rf calibration + spots for a single plate (spec §7).
///
/// Coordinates are normalized to the image (0 = top, 1 = bottom). The baseline
/// (origin) sits lower in the image than the solvent front, so
/// `baselineY > frontY`. Rf is computed per spec §7:
///
///     Rf = (baselineY − spotY) / (baselineY − frontY)
struct Calibration {
    var baselineY: CGFloat = 0.82
    var frontY: CGFloat = 0.15

    /// Valid only when the front is above the baseline with a real gap.
    var isValid: Bool { baselineY - frontY > 0.02 }

    func rf(forNormalizedY y: CGFloat) -> Double {
        guard isValid else { return .nan }
        return Double((baselineY - y) / (baselineY - frontY))
    }
}

extension Double {
    /// Rf values display to two decimals; out-of-range (mis-calibration) shows "—".
    var rfDisplay: String {
        guard isFinite else { return "—" }
        return String(format: "%.2f", self)
    }

    var isRfInRange: Bool { isFinite && self >= -0.02 && self <= 1.02 }
}
