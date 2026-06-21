import SwiftUI

/// Interactive plate with the image, two draggable reference lines (baseline /
/// solvent front) and tappable, draggable spot markers (spec §7). Coordinates
/// are mapped between the on-screen image rect and normalized image space so Rf
/// stays resolution independent.
struct PlateCanvas: View {
    @EnvironmentObject private var store: AppStore
    let image: NSImage

    private let space = "plate"

    var body: some View {
        GeometryReader { geo in
            let rect = Self.imageRect(in: geo.size, image: image)

            ZStack(alignment: .topLeading) {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(.high)
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)
                    .shadow(color: .black.opacity(0.18), radius: 16, y: 6)

                // Tap layer for adding spots (active in Spot mode).
                Color.clear
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)
                    .contentShape(Rectangle())
                    .onTapGesture { location in
                        guard store.isSpotMode else { return }
                        store.addSpot(atNormalized: normalize(location, in: rect))
                    }

                ReferenceLine(
                    title: "Solvent front",
                    color: Palette.accent,
                    rect: rect,
                    space: space,
                    normalizedY: $store.calibration.frontY,
                    clamp: { min(max($0, 0), store.calibration.baselineY - 0.02) },
                    onDrag: { store.calibrationUserModified = true }
                )

                ReferenceLine(
                    title: "Baseline",
                    color: Palette.standard,
                    rect: rect,
                    space: space,
                    normalizedY: $store.calibration.baselineY,
                    clamp: { min(max($0, store.calibration.frontY + 0.02), 1) },
                    onDrag: { store.calibrationUserModified = true }
                )

                ForEach(store.spots) { spot in
                    SpotMarker(
                        spot: spot,
                        rect: rect,
                        space: space,
                        isSelected: store.selectedSpotID == spot.id,
                        rf: store.calibration.rf(forNormalizedY: spot.point.y)
                    )
                }
            }
            .coordinateSpace(name: space)
        }
    }

    private func normalize(_ p: CGPoint, in rect: CGRect) -> CGPoint {
        CGPoint(x: (p.x - rect.minX) / rect.width,
                y: (p.y - rect.minY) / rect.height)
    }

    /// Aspect-fit rect for the image inside `size`.
    static func imageRect(in size: CGSize, image: NSImage) -> CGRect {
        let img = image.size
        guard img.width > 0, img.height > 0 else { return .zero }
        let scale = min(size.width / img.width, size.height / img.height)
        let w = img.width * scale
        let h = img.height * scale
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }
}

// MARK: - Reference line

private struct ReferenceLine: View {
    let title: String
    let color: Color
    let rect: CGRect
    let space: String
    @Binding var normalizedY: CGFloat
    let clamp: (CGFloat) -> CGFloat
    var onDrag: (() -> Void)? = nil

    @GestureState private var isDragging = false

    var body: some View {
        let y = rect.minY + normalizedY * rect.height
        ZStack {
            // Invisible wide hit area so the line is easy to grab.
            Rectangle()
                .fill(Color.clear)
                .frame(width: rect.width, height: 44)
                .contentShape(Rectangle())

            // Visible dashed line.
            Path { p in
                p.move(to: CGPoint(x: rect.minX, y: 0))
                p.addLine(to: CGPoint(x: rect.maxX, y: 0))
            }
            .stroke(color.opacity(isDragging ? 1.0 : 0.85),
                    style: StrokeStyle(lineWidth: isDragging ? 2 : 1.5, dash: [6, 4]))

            // Label pill at the right edge.
            Text(title)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 7)
                .padding(.vertical, 2)
                .background(Capsule().fill(color))
                .position(x: rect.maxX - 46, y: 0)

            // Drag handle nub on the left.
            Image(systemName: "line.3.horizontal")
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(color)
                .padding(4)
                .background(Circle().fill(color.opacity(0.15)))
                .position(x: rect.minX + 14, y: 0)
        }
        .frame(width: rect.width, height: 44)
        .position(x: rect.midX, y: y)
        .highPriorityGesture(
            DragGesture(minimumDistance: 1, coordinateSpace: .named(space))
                .updating($isDragging) { _, state, _ in state = true }
                .onChanged { value in
                    onDrag?()
                    normalizedY = clamp((value.location.y - rect.minY) / rect.height)
                }
        )
        .onHover { hovering in
            if hovering { NSCursor.resizeUpDown.push() } else { NSCursor.pop() }
        }
    }
}

// MARK: - Spot marker

private struct SpotMarker: View {
    @EnvironmentObject private var store: AppStore
    let spot: Spot
    let rect: CGRect
    let space: String
    let isSelected: Bool
    let rf: Double

    var body: some View {
        let x = rect.minX + spot.point.x * rect.width
        let y = rect.minY + spot.point.y * rect.height

        ZStack {
            Circle()
                .stroke(spot.label.color, lineWidth: 2.5)
                .background(Circle().fill(spot.label.color.opacity(0.18)))
                .frame(width: 22, height: 22)

            if isSelected {
                Circle()
                    .stroke(Color.primary.opacity(0.5), lineWidth: 1)
                    .frame(width: 30, height: 30)
            }
        }
        .overlay(alignment: .top) {
            Text(rf.rfDisplay)
                .font(.tabular(9, weight: .semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(Capsule().fill(spot.label.color))
                .offset(y: -16)
                .fixedSize()
        }
        .position(x: x, y: y)
        .gesture(
            DragGesture(coordinateSpace: .named(space))
                .onChanged { value in
                    store.selectedSpotID = spot.id
                    store.moveSpot(spot.id, toNormalized: CGPoint(
                        x: (value.location.x - rect.minX) / rect.width,
                        y: (value.location.y - rect.minY) / rect.height
                    ))
                }
        )
        .onTapGesture { store.selectedSpotID = spot.id }
    }
}
