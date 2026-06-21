import SwiftUI
import UniformTypeIdentifiers

/// Reaction time-course strip, merged into the floating toolbar (spec §6).
/// • Plus-box adds a new plate; a new plus appears after each add.
/// • Click an existing plate → import a new image overwriting it.
/// • Click a non-active plate → select it (show on canvas).
/// • Drag plates to reorder.
struct PlateStrip: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        HStack(spacing: 6) {
            ForEach(Array(store.plates.enumerated()), id: \.element.id) { idx, _ in
                Thumb(index: idx,
                      image: store.thumbnailImage(idx),
                      isCurrent: idx == store.activeIndex)
            }
            AddBox()
        }
    }
}

private struct Thumb: View {
    @EnvironmentObject private var store: AppStore
    let index: Int
    let image: NSImage?
    let isCurrent: Bool

    var body: some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .fill(Color.primary.opacity(0.06))
            .frame(width: 28, height: 34)
            .overlay {
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 28, height: 34)
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(isCurrent ? Palette.accent : Color.primary.opacity(0.14),
                                  lineWidth: isCurrent ? 2 : 1)
            )
            .overlay(alignment: .bottomTrailing) {
                Text("\(index + 1)")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.white)
                    .padding(2)
                    .background(Circle().fill(Palette.accent.opacity(isCurrent ? 1 : 0.5)))
                    .offset(x: 3, y: 3)
            }
            .help(isCurrent ? "Click to replace this plate" : "Click to view · drag to reorder")
            .onTapGesture {
                if index == store.activeIndex { store.overwritePlate(index) }
                else { store.selectPlate(index) }
            }
            .onDrag { NSItemProvider(object: "\(index)" as NSString) }
            .onDrop(of: [UTType.text], isTargeted: nil) { providers in
                guard let p = providers.first else { return false }
                _ = p.loadObject(ofClass: NSString.self) { obj, _ in
                    if let s = obj as? String, let from = Int(s) {
                        Task { @MainActor in store.movePlate(from: from, to: index) }
                    }
                }
                return true
            }
    }
}

private struct AddBox: View {
    @EnvironmentObject private var store: AppStore
    @State private var hovering = false

    var body: some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
            .foregroundStyle(Palette.accent.opacity(hovering ? 1 : 0.5))
            .frame(width: 28, height: 34)
            .overlay {
                Image(systemName: "plus")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Palette.accent.opacity(hovering ? 1 : 0.7))
            }
            .onHover { hovering = $0 }
            .onTapGesture { store.addPlateFromPanel() }
            .help("Add a plate to the time course")
    }
}
