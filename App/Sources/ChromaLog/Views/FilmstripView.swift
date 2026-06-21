import SwiftUI

/// Reaction time-course filmstrip (spec §6/§11): plates laid out by time, the
/// current plate highlighted, trailing "+" to add a plate. M0 is a styled
/// placeholder reflecting the current plate.
struct FilmstripView: View {
    @EnvironmentObject private var store: AppStore

    private let slots = 3

    var body: some View {
        HStack(spacing: 10) {
            Text("Reaction time course")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                ForEach(0..<slots, id: \.self) { index in
                    Thumbnail(
                        image: index == store.plateIndex - 1 ? store.plateImage : nil,
                        isCurrent: index == store.plateIndex - 1
                    )
                }
                AddPlateButton()
            }

            Text("0 → 2 → 4h")
                .font(.tabular(11))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            Capsule(style: .continuous).fill(.thinMaterial)
        )
        .overlay(
            Capsule(style: .continuous).strokeBorder(Color.primary.opacity(0.06), lineWidth: 1)
        )
    }
}

private struct Thumbnail: View {
    let image: NSImage?
    let isCurrent: Bool

    var body: some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .fill(Color.primary.opacity(0.06))
            .frame(width: 30, height: 38)
            .overlay {
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 30, height: 38)
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(isCurrent ? Palette.accent : Color.primary.opacity(0.12),
                                  lineWidth: isCurrent ? 2 : 1)
            )
    }
}

private struct AddPlateButton: View {
    @State private var hovering = false

    var body: some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
            .foregroundStyle(Color.secondary.opacity(hovering ? 0.8 : 0.4))
            .frame(width: 30, height: 38)
            .overlay { Image(systemName: "plus").font(.system(size: 12, weight: .medium)).foregroundStyle(.secondary) }
            .onHover { hovering = $0 }
    }
}
