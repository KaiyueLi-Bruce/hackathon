import SwiftUI

/// Canvas-first three-region shell (spec §6): icon rail (~54px) · dominant
/// canvas · context inspector (~208px). Left/right collapse into Focus mode.
struct ContentView: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        HStack(spacing: 0) {
            if store.showRail {
                IconRail()
                    .frame(width: 54)
                    .transition(.move(edge: .leading).combined(with: .opacity))
                Divider()
            }

            CanvasView()
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            if store.showInspector {
                Divider()
                InspectorView()
                    .frame(width: 280)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .animation(.spring(response: 0.32, dampingFraction: 0.86), value: store.showRail)
        .animation(.spring(response: 0.32, dampingFraction: 0.86), value: store.showInspector)
        .background(Color(nsColor: .windowBackgroundColor))
        .toolbar { toolbarContent }
        .sheet(isPresented: $store.showArchive) {
            ArchiveView()
        }
        .sheet(isPresented: $store.showSettings) {
            SettingsView()
        }
        .overlay(alignment: .bottom) {
            if let status = store.saveStatus {
                Text(status)
                    .font(.system(size: 12, weight: .medium))
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(Capsule().fill(.regularMaterial))
                    .overlay(Capsule().strokeBorder(Color.primary.opacity(0.08)))
                    .padding(.bottom, 200)
                    .transition(.opacity)
                    .task(id: status) {
                        try? await Task.sleep(nanoseconds: 2_000_000_000)
                        store.saveStatus = nil
                    }
            }
        }
        .animation(.easeInOut, value: store.saveStatus)
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Button {
                withAnimation { store.showRail.toggle() }
            } label: {
                Image(systemName: "sidebar.left")
            }
            .help("Toggle navigation")
        }

        ToolbarItem(placement: .principal) {
            // Click to rename the plate/experiment (auto-named by date otherwise).
            TextField("Plate name", text: $store.plateTitle)
                .textFieldStyle(.plain)
                .multilineTextAlignment(.center)
                .font(.system(size: 13, weight: .semibold))
                .frame(minWidth: 180, maxWidth: 320)
                .help("Click to rename")
        }

        ToolbarItemGroup(placement: .primaryAction) {
            Button { store.showSettings = true } label: { Image(systemName: "gearshape") }
                .help("Settings (AI / OpenRouter)")
            Button { store.saveCurrentPlate() } label: { Image(systemName: "square.and.arrow.down") }
                .help("Save plate")
                .keyboardShortcut("s", modifiers: [.command])
                .disabled(!store.hasImage)
            Button {
                withAnimation { store.showInspector.toggle() }
            } label: {
                Image(systemName: "sidebar.right")
            }
            .help("Toggle inspector")
        }
    }
}
