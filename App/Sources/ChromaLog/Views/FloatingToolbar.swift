import SwiftUI

/// Floating glass toolbar centered at the bottom of the canvas (spec §6/§11).
/// Auto-detect is the highlighted primary action. M0 wires only Import.
struct FloatingToolbar: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        HStack(spacing: 6) {
            ToolButton(title: "Import", systemImage: "tray.and.arrow.down") {
                store.presentImportPanel()
            }
            divider
            ToolButton(title: "Rotate L", systemImage: "rotate.left", disabled: !store.hasImage) {
                store.rotatePlate(clockwise: false)
            }
            ToolButton(title: "Rotate R", systemImage: "rotate.right", disabled: !store.hasImage) {
                store.rotatePlate(clockwise: true)
            }
            divider
            ToolButton(title: "Spot", systemImage: "smallcircle.filled.circle",
                       active: store.isSpotMode, disabled: !store.hasImage) {
                store.isSpotMode.toggle()
            }
            ToolButton(title: "Auto-detect", systemImage: "wand.and.stars", prominent: true,
                       disabled: !store.hasImage || store.isAutoDetecting,
                       loading: store.isAutoDetecting) {
                store.runAutoDetect()
            }
            ToolButton(title: "Redraw", systemImage: "square.on.square.dashed",
                       active: store.showDigitalPlate, disabled: !store.hasImage) {
                store.showDigitalPlate.toggle()
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            Capsule(style: .continuous)
                .fill(.regularMaterial)
        )
        .overlay(
            Capsule(style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.18), radius: 16, y: 6)
    }

    private var divider: some View {
        Rectangle()
            .fill(Color.primary.opacity(0.12))
            .frame(width: 1, height: 22)
            .padding(.horizontal, 2)
    }
}

private struct ToolButton: View {
    let title: String
    let systemImage: String
    var prominent: Bool = false
    var active: Bool = false
    var disabled: Bool = false
    var loading: Bool = false
    let action: () -> Void

    @State private var hovering = false

    private var background: Color {
        if prominent { return Palette.accent }
        if active { return Palette.accent.opacity(0.22) }
        if hovering { return Color.primary.opacity(0.08) }
        return .clear
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                if loading {
                    ProgressView()
                        .scaleEffect(0.65)
                        .frame(width: 14, height: 14)
                } else {
                    Image(systemName: systemImage)
                        .font(.system(size: 13, weight: .medium))
                }
                Text(title)
                    .font(.system(size: 12.5, weight: prominent ? .semibold : .regular))
            }
            .foregroundStyle(prominent ? Color.white : (active ? Palette.accent : Color.primary))
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(background)
            )
        }
        .buttonStyle(.plain)
        .opacity(disabled ? 0.4 : 1)
        .disabled(disabled)
        .onHover { hovering = $0 }
    }
}
