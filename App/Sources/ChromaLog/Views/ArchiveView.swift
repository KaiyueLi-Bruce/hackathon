import SwiftUI

/// Searchable archive grid (spec §11 #3): thumbnail cards with key metadata;
/// a top search bar filters by compound/project/solvent/visualization. Clicking
/// a card loads it back into the workspace.
struct ArchiveView: View {
    @EnvironmentObject private var store: AppStore
    @Environment(\.dismiss) private var dismiss

    private let columns = [GridItem(.adaptive(minimum: 160, maximum: 220), spacing: 16)]

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()

            if store.experiments.isEmpty {
                emptyState
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 16) {
                        ForEach(store.experiments) { record in
                            ArchiveCard(record: record)
                        }
                    }
                    .padding(20)
                }
            }
        }
        .frame(minWidth: 620, minHeight: 460)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
            TextField("Search compound · project · solvent · visualization",
                      text: $store.archiveQuery)
                .textFieldStyle(.plain)
                .font(.system(size: 14))
                .onChange(of: store.archiveQuery) { _, _ in store.refreshExperiments() }

            Spacer()
            Button("Done") { dismiss() }
                .keyboardShortcut(.defaultAction)
        }
        .padding(16)
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "tray")
                .font(.system(size: 36, weight: .light))
                .foregroundStyle(.secondary)
            Text(store.archiveQuery.isEmpty ? "No saved plates yet" : "No matches")
                .font(.system(size: 15, weight: .medium))
            if store.archiveQuery.isEmpty {
                Text("Calibrate a plate and press Save to build your archive.")
                    .font(.system(size: 12)).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct ArchiveCard: View {
    @EnvironmentObject private var store: AppStore
    let record: ExperimentRecord
    @State private var hovering = false

    private var thumbnail: NSImage? { AppDatabase.shared.loadImage(record.imageFileName) }
    private var seriesCount: Int { AppDatabase.shared.seriesCount(record.id) }

    /// A narrower "sheet" peeking above the main card to suggest a stack.
    private func sheet(inset: CGFloat, lift: CGFloat, shade: Double) -> some View {
        RoundedRectangle(cornerRadius: 10, style: .continuous)
            .fill(Color(nsColor: .controlBackgroundColor))
            .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Color.primary.opacity(shade)))
            .frame(height: 140)
            .padding(.horizontal, inset)
            .offset(y: -lift)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack(alignment: .top) {
                // Stacked "sheets" peeking above the card when it's a multi-plate series.
                if seriesCount > 1 {
                    sheet(inset: 24, lift: 12, shade: 0.10)
                    sheet(inset: 13, lift: 6, shade: 0.16)
                }
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color.primary.opacity(0.05))
                    if let thumbnail {
                        Image(nsImage: thumbnail)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                    }
                }
                .frame(height: 140)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.08)))
            }
            .padding(.top, seriesCount > 1 ? 12 : 0)
            .overlay(alignment: .bottomTrailing) {
                if seriesCount > 1 {
                    Text("\(seriesCount) plates")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 7).padding(.vertical, 3)
                        .background(Capsule().fill(Palette.accent))
                        .padding(8)
                }
            }
            .overlay(alignment: .topTrailing) {
                if hovering {
                    Button { store.deleteExperiment(record) } label: {
                        Image(systemName: "trash.circle.fill")
                            .font(.system(size: 18))
                            .foregroundStyle(.white, Color(Palette.coral))
                    }
                    .buttonStyle(.plain)
                    .padding(6)
                }
            }

            Text(record.title)
                .font(.system(size: 13, weight: .semibold))
                .lineLimit(1)

            HStack(spacing: 6) {
                Text(record.channel)
                if !record.solventSystem.isEmpty {
                    Text("·"); Text(record.solventSystem).lineLimit(1)
                }
            }
            .font(.system(size: 11))
            .foregroundStyle(.secondary)
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor).opacity(hovering ? 0.9 : 0.5))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(hovering ? Palette.accent.opacity(0.5) : Color.clear, lineWidth: 1)
        )
        .onHover { hovering = $0 }
        .onTapGesture { store.loadExperiment(record) }
    }
}
