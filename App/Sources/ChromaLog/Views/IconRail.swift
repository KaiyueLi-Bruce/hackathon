import SwiftUI

/// Narrow icon navigation (spec §6). Uses vibrancy material; selected item is
/// filled with the accent color; bottom holds Settings.
struct IconRail: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        VStack(spacing: 6) {
            // Experiments (home) + a single Library button. The old Plates/Compounds/
            // Search buttons all just opened the archive, so they're collapsed into one.
            RailButton(systemImage: NavSection.experiments.systemImage,
                       help: NavSection.experiments.title,
                       isSelected: store.selectedSection == .experiments) {
                store.selectedSection = .experiments
            }
            RailButton(systemImage: NavSection.plates.systemImage,
                       help: "Library",
                       isSelected: store.selectedSection == .plates) {
                store.selectedSection = .plates
                store.openArchive()
            }

            Spacer()

            RailButton(systemImage: "gearshape", help: "Settings", isSelected: false) {
                store.showSettings = true
            }
        }
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.ultraThinMaterial)
    }
}

private struct RailButton: View {
    let systemImage: String
    let help: String
    let isSelected: Bool
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(isSelected ? Color.white : Color.secondary)
                .frame(width: 34, height: 34)
                .background(
                    RoundedRectangle(cornerRadius: 9, style: .continuous)
                        .fill(isSelected ? Palette.accent : (hovering ? Color.secondary.opacity(0.14) : Color.clear))
                )
        }
        .buttonStyle(.plain)
        .help(help)
        .onHover { hovering = $0 }
    }
}
