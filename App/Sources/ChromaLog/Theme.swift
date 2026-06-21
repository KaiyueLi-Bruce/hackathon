import SwiftUI

/// Palette per spec §11: cool lab accent (teal), distinguishable marker colors
/// (SM=indigo, Product=teal, By-product=coral), color-blind friendly.
enum Palette {
    static let accent  = Color(red: 0.06, green: 0.55, blue: 0.55)   // teal
    static let indigo  = Color(red: 0.35, green: 0.34, blue: 0.84)   // SM
    static let teal    = Color(red: 0.06, green: 0.60, blue: 0.58)   // Product
    static let coral   = Color(red: 0.95, green: 0.45, blue: 0.39)   // By-product
    static let amber   = Color(red: 0.90, green: 0.62, blue: 0.13)   // Impurity
    static let standard = Color(red: 0.55, green: 0.55, blue: 0.60)  // Standard
}

/// Spot label classes used for marker coloring.
enum SpotLabel: String, CaseIterable {
    case sm = "SM"
    case product = "Product"
    case byproduct = "By-product"
    case impurity = "Impurity"
    case standard = "Standard"
    case cospot = "Co-spot"

    var color: Color {
        switch self {
        case .sm:        return Palette.indigo
        case .product:   return Palette.teal
        case .byproduct: return Palette.coral
        case .impurity:  return Palette.amber
        case .standard:  return Palette.standard
        case .cospot:    return Palette.accent
        }
    }
}

extension Font {
    /// Tabular, monospaced-digit numerals for Rf tables (spec §11).
    static func tabular(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .default).monospacedDigit()
    }
}
