import SwiftUI

/// Interactive plate with the image, two draggable reference lines (baseline /
/// solvent front) and tappable, draggable spot markers (spec §7). Coordinates
/// are mapped between the on-screen image rect and normalized image space so Rf
/// stays resolution independent.
///
/// Interaction:
///   • Single click on the image → add a new spot.
///   • Double click on a spot     → delete it.
///   • Reference lines span the full image width; the drag hit zone is the large
///     empty strip toward each line's end edge (front→top, baseline→bottom), so
///     dragging a line never collides with adding/selecting spots in the body.
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

                // Single-click anywhere on the image adds a spot.
                Color.clear
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)
                    .contentShape(Rectangle())
                    .gesture(
                        SpatialTapGesture(coordinateSpace: .named(space))
                            .onEnded { value in
                                store.addSpot(atNormalized: normalize(value.location, in: rect))
                            }
                    )

                ReferenceLine(
                    title: "Solvent front",
                    color: Palette.accent,
                    rect: rect,
                    canvasSize: geo.size,
                    space: space,
                    normalizedY: $store.calibration.frontY,
                    clamp: { min(max($0, 0), store.calibration.baselineY - 0.02) },
                    onDrag: { store.calibrationUserModified = true }
                )

                ReferenceLine(
                    title: "Baseline",
                    color: Palette.standard,
                    rect: rect,
                    canvasSize: geo.size,
                    space: space,
                    normalizedY: $store.calibration.baselineY,
                    clamp: { min(max($0, store.calibration.frontY + 0.02), 1) },
                    onDrag: { store.calibrationUserModified = true }
                )

                ForEach(store.spots) { spot in
                    SpotMarker(
                        spot: spot,
                        image: image,
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
    let canvasSize: CGSize
    let space: String
    @Binding var normalizedY: CGFloat
    let clamp: (CGFloat) -> CGFloat
    var onDrag: (() -> Void)? = nil

    @GestureState private var isDragging = false
    private let bandH: CGFloat = 28     // grab-tab height
    private let tabW: CGFloat = 40      // grab-tab width (lives outside the image)

    var body: some View {
        let y = rect.minY + normalizedY * rect.height
        // Line extends from the canvas left edge to the canvas right edge — i.e.
        // it overhangs the image on both sides. Handles sit in those overhangs.
        let leftX = max(0, rect.minX - tabW)
        let rightX = min(canvasSize.width, rect.maxX + tabW)

        ZStack {
            // 1) Visible line spanning beyond the image on both ends.
            Rectangle()
                .fill(color.opacity(isDragging ? 1.0 : 0.9))
                .frame(width: rightX - leftX, height: isDragging ? 2.5 : 1.8)
                .position(x: (leftX + rightX) / 2, y: y)
                .allowsHitTesting(false)

            // 2) Two grab handles in the overhang regions (OUTSIDE the image),
            //    so dragging up/down never touches spots inside the image.
            handle(centerX: rect.minX - tabW / 2, y: y)
            handle(centerX: rect.maxX + tabW / 2, y: y)

            // 3) Label pill on the right overhang.
            Text(title)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(Capsule().fill(color))
                .position(x: rect.maxX + tabW / 2, y: y - 14)
                .allowsHitTesting(false)
        }
    }

    @ViewBuilder
    private func handle(centerX: CGFloat, y: CGFloat) -> some View {
        ZStack {
            Capsule().fill(color.opacity(isDragging ? 0.9 : 0.6))
                .frame(width: tabW - 8, height: 6)
            Image(systemName: "line.3.horizontal")
                .font(.system(size: 8, weight: .bold))
                .foregroundStyle(.white)
        }
        .frame(width: tabW, height: bandH)
        .contentShape(Rectangle())
        .position(x: centerX, y: y)
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
    let image: NSImage
    let rect: CGRect
    let space: String
    let isSelected: Bool
    let rf: Double

    @State private var showPicker = false
    @GestureState private var isDragging = false

    var body: some View {
        let x = rect.minX + spot.point.x * rect.width
        let y = rect.minY + spot.point.y * rect.height

        ZStack {
            Circle()
                .stroke(spot.displayColor, lineWidth: 2.5)
                .background(Circle().fill(spot.displayColor.opacity(0.18)))
                .frame(width: 22, height: 22)

            if isSelected {
                Circle()
                    .stroke(Color.primary.opacity(0.5), lineWidth: 1)
                    .frame(width: 30, height: 30)
            }
        }
        .overlay(alignment: .top) {
            Text("\(spot.displayName)  \(rf.rfDisplay)")
                .font(.tabular(9, weight: .semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 4).padding(.vertical, 1)
                .background(Capsule().fill(spot.displayColor))
                .offset(y: -16)
                .fixedSize()
        }
        .overlay {
            if isDragging {
                SpotMagnifier(image: image, normalizedPoint: spot.point)
                    .offset(y: -90)
                    .allowsHitTesting(false)
                    .transition(.opacity.combined(with: .scale(scale: 0.85, anchor: .bottom)))
            }
        }
        .animation(.spring(response: 0.18, dampingFraction: 0.75), value: isDragging)
        .contentShape(Circle())
        .position(x: x, y: y)
        .highPriorityGesture(
            TapGesture(count: 2).onEnded { store.deleteSpot(spot.id) }
        )
        .gesture(
            DragGesture(coordinateSpace: .named(space))
                .updating($isDragging) { _, state, _ in state = true }
                .onChanged { value in
                    store.selectedSpotID = spot.id
                    store.moveSpot(spot.id, toNormalized: CGPoint(
                        x: (value.location.x - rect.minX) / rect.width,
                        y: (value.location.y - rect.minY) / rect.height
                    ))
                }
        )
        .onTapGesture {
            store.selectedSpotID = spot.id
            showPicker = true
        }
        .popover(isPresented: $showPicker, arrowEdge: .trailing) {
            SpotLabelPopover(spotID: spot.id) { showPicker = false }
                .environmentObject(store)
        }
    }
}

// MARK: - Magnifier loupe shown while dragging a spot

private struct SpotMagnifier: View {
    let image: NSImage
    let normalizedPoint: CGPoint

    private let size: CGFloat = 88
    private let zoom: CGFloat = 4.0

    var body: some View {
        let imgW = image.size.width
        let imgH = image.size.height
        // Offset scaled image so the spot center lands in the middle of the loupe.
        let ox = size / 2 - normalizedPoint.x * imgW * zoom
        let oy = size / 2 - normalizedPoint.y * imgH * zoom

        ZStack {
            Circle()
                .fill(.regularMaterial)

            Image(nsImage: image)
                .resizable()
                .interpolation(.high)
                .frame(width: imgW * zoom, height: imgH * zoom)
                .offset(x: ox, y: oy)
                .frame(width: size, height: size)
                .clipShape(Circle())

            // Crosshair
            Path { p in
                p.move(to: CGPoint(x: size / 2 - 10, y: size / 2))
                p.addLine(to: CGPoint(x: size / 2 + 10, y: size / 2))
                p.move(to: CGPoint(x: size / 2, y: size / 2 - 10))
                p.addLine(to: CGPoint(x: size / 2, y: size / 2 + 10))
            }
            .stroke(Color.white.opacity(0.85), lineWidth: 1)

            Circle()
                .stroke(Color.primary.opacity(0.25), lineWidth: 1.5)
        }
        .frame(width: size, height: size)
        .shadow(color: .black.opacity(0.3), radius: 8, y: 3)
    }
}

/// Quick label picker shown beside a spot: preset labels + custom text + delete.
private struct SpotLabelPopover: View {
    @EnvironmentObject private var store: AppStore
    let spotID: Spot.ID
    let onDone: () -> Void

    @State private var customText = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Label").font(.system(size: 10, weight: .semibold)).foregroundStyle(.secondary)
            ForEach(SpotLabel.allCases, id: \.self) { lab in
                Button {
                    store.setLabel(lab, for: spotID); onDone()
                } label: {
                    HStack(spacing: 7) {
                        Circle().fill(lab.color).frame(width: 9, height: 9)
                        Text(lab.rawValue).font(.system(size: 12))
                        Spacer()
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            Divider()
            HStack(spacing: 6) {
                TextField("Custom…", text: $customText)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12))
                    .onSubmit(applyCustom)
                Button("Set", action: applyCustom)
                    .disabled(customText.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            Divider()
            Button(role: .destructive) {
                store.deleteSpot(spotID); onDone()
            } label: {
                Label("Delete spot", systemImage: "trash").font(.system(size: 12))
            }
            .buttonStyle(.plain)
        }
        .padding(10)
        .frame(width: 184)
    }

    private func applyCustom() {
        let t = customText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        store.setCustomLabel(t, for: spotID)
        onDone()
    }
}
