import SwiftUI

/// A self-contained view used with `ImageRenderer` to produce the annotated
/// export image. No EnvironmentObject — all data is passed in directly.
struct PlateExportView: View {
    let title: String
    let date: Date
    let solventSystem: String
    let ratio: String
    let rfResults: [(spot: Spot, rf: Double)]

    private let canvasSize = CGSize(width: 600, height: 800)

    // Plate schematic geometry (fractions of the plate rect).
    private let frontFrac: CGFloat = 0.10
    private let baselineFrac: CGFloat = 0.90

    var body: some View {
        ZStack(alignment: .topLeading) {
            // White background.
            Color.white

            // Header annotations.
            headerOverlay

            // Plate schematic.
            GeometryReader { geo in
                let plate = plateRect(in: geo.size)
                ZStack {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.white)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .strokeBorder(Color.black.opacity(0.2), lineWidth: 1)
                        )
                        .frame(width: plate.width, height: plate.height)
                        .position(x: plate.midX, y: plate.midY)

                    referenceLine(at: frontFrac, in: plate, label: "Solvent front", color: Palette.accent)
                    referenceLine(at: baselineFrac, in: plate, label: "Baseline", color: Palette.standard)

                    ForEach(rfResults, id: \.spot.id) { result in
                        spotView(result.spot, rf: result.rf, in: plate)
                    }
                }
            }
        }
        .frame(width: canvasSize.width, height: canvasSize.height)
    }

    // MARK: - Header

    private var headerOverlay: some View {
        ZStack(alignment: .top) {
            // Top-left: date.
            Text(date.formatted(date: .abbreviated, time: .omitted))
                .font(.system(size: 13, weight: .regular, design: .monospaced))
                .foregroundStyle(Color.black.opacity(0.55))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 18)
                .padding(.leading, 24)

            // Top-center: experiment title.
            Text(title.isEmpty ? "Untitled Plate" : title)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Color.black)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.top, 16)

            // Top-right: solvent system / ratio.
            VStack(alignment: .trailing, spacing: 2) {
                if !solventSystem.isEmpty {
                    Text(solventSystem)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Color.black.opacity(0.75))
                }
                if !ratio.isEmpty {
                    Text(ratio)
                        .font(.system(size: 11, weight: .regular))
                        .foregroundStyle(Color.black.opacity(0.55))
                }
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
            .padding(.top, 16)
            .padding(.trailing, 24)
        }
    }

    // MARK: - Plate helpers

    private func plateRect(in size: CGSize) -> CGRect {
        let topPad: CGFloat = 56   // space for the header text
        let available = CGSize(width: size.width, height: size.height - topPad)
        let w = min(available.width - 60, (available.height - 60) * 0.62)
        let h = min(available.height - 60, w / 0.62)
        let x = (size.width - w) / 2
        let y = topPad + (available.height - h) / 2
        return CGRect(x: x, y: y, width: w, height: max(h, 1))
    }

    private func referenceLine(at frac: CGFloat, in plate: CGRect, label: String, color: Color) -> some View {
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

    private func spotView(_ spot: Spot, rf: Double, in plate: CGRect) -> some View {
        let clampedRf = CGFloat(min(max(rf, 0), 1))
        let y = plate.minY + (baselineFrac - clampedRf * (baselineFrac - frontFrac)) * plate.height
        let x = plate.minX + CGFloat(spot.point.x) * plate.width
        return ZStack {
            Ellipse()
                .fill(spot.displayColor.opacity(0.85))
                .frame(width: 20, height: 12)
            Text(rf.rfDisplay)
                .font(.system(size: 9, weight: .semibold).monospacedDigit())
                .foregroundStyle(Color.black)
                .offset(x: 24)
        }
        .position(x: x, y: y)
    }
}

// MARK: - Render helper

extension PlateExportView {
    /// Renders the annotated plate to a 600×800 PNG `NSImage`.
    @MainActor
    static func render(
        title: String,
        date: Date = Date(),
        solventSystem: String,
        ratio: String,
        rfResults: [(spot: Spot, rf: Double)]
    ) -> NSImage? {
        let view = PlateExportView(
            title: title,
            date: date,
            solventSystem: solventSystem,
            ratio: ratio,
            rfResults: rfResults
        )
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2.0   // retina
        guard let cgImage = renderer.cgImage else { return nil }
        return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width / 2,
                                                      height: cgImage.height / 2))
    }
}
