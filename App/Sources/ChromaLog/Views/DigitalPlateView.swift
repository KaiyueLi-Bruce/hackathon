import SwiftUI

/// Standardized "digital plate" redraw (spec §7 step 5): the messy photo
/// becomes a clean schematic — uniform proportions, fixed baseline/front, spots
/// placed by Rf and labeled. Pure vector drawing.
struct DigitalPlateView: View {
    @EnvironmentObject private var store: AppStore

    // Fixed schematic geometry (fraction of the plate rect).
    private let frontFrac: CGFloat = 0.10
    private let baselineFrac: CGFloat = 0.90

    var body: some View {
        GeometryReader { geo in
            let plate = plateRect(in: geo.size)
            ZStack {
                // Plate body.
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color(nsColor: .textBackgroundColor))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .strokeBorder(Color.primary.opacity(0.15), lineWidth: 1)
                    )
                    .frame(width: plate.width, height: plate.height)
                    .position(x: plate.midX, y: plate.midY)

                line(at: frontFrac, in: plate, label: "Solvent front", color: Palette.accent)
                line(at: baselineFrac, in: plate, label: "Baseline", color: Palette.standard)

                ForEach(store.rfResults, id: \.spot.id) { result in
                    spot(result.spot, rf: result.rf, in: plate)
                }
            }
        }
    }

    private func plateRect(in size: CGSize) -> CGRect {
        let w = min(size.width - 80, (size.height - 80) * 0.62)
        let h = min(size.height - 80, w / 0.62)
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: max(h, 1))
    }

    private func line(at frac: CGFloat, in plate: CGRect, label: String, color: Color) -> some View {
        let y = plate.minY + frac * plate.height
        return ZStack {
            Path { p in
                p.move(to: CGPoint(x: plate.minX, y: y))
                p.addLine(to: CGPoint(x: plate.maxX, y: y))
            }
            .stroke(color.opacity(0.7), style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(color)
                .position(x: plate.minX + 42, y: y - 8)
        }
    }

    private func spot(_ spot: Spot, rf: Double, in plate: CGRect) -> some View {
        // y by Rf: Rf 1 → front, Rf 0 → baseline.
        let clampedRf = CGFloat(min(max(rf, 0), 1))
        let y = plate.minY + (baselineFrac - clampedRf * (baselineFrac - frontFrac)) * plate.height
        let x = plate.minX + CGFloat(spot.point.x) * plate.width
        return ZStack {
            Ellipse()
                .fill(spot.displayColor.opacity(0.85))
                .frame(width: 20, height: 12)
                .shadow(color: spot.displayColor.opacity(0.4), radius: 3)
            Text(rf.rfDisplay)
                .font(.tabular(9, weight: .semibold))
                .foregroundStyle(.primary)
                .offset(x: 24)
        }
        .position(x: x, y: y)
    }
}
