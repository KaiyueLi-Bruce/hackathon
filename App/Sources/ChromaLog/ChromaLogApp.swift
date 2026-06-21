import SwiftUI
import AppKit

@main
struct AutoChemApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var store = AppStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
                .frame(minWidth: 1000, minHeight: 660)
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified(showsTitle: false))
        .commands {
            CommandGroup(after: .toolbar) {
                Button("Toggle Sidebar") { store.showRail.toggle() }
                    .keyboardShortcut("[", modifiers: [.command])
                Button("Toggle Inspector") { store.showInspector.toggle() }
                    .keyboardShortcut("]", modifiers: [.command])
                Button("Focus Mode") { store.enterFocusMode() }
                    .keyboardShortcut("f", modifiers: [.command, .shift])
            }
        }
    }
}

/// Forces a regular (foreground) activation policy so the window appears and
/// takes focus when launched as a Swift Package executable.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.global(qos: .background).async {
            SidecarManager.shared.start()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        SidecarManager.shared.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
