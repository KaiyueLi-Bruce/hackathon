import SwiftUI

/// AI / OpenRouter settings (spec 附录 D, three-tier detection).
/// Key stored in macOS Keychain; model + toggle in UserDefaults.
struct SettingsView: View {
    @EnvironmentObject private var store: AppStore
    @Environment(\.dismiss) private var dismiss

    @State private var apiKey: String = ""
    @State private var keySaved: Bool = KeychainHelper.hasAPIKey

    /// Curated vision-capable OpenRouter models (user can also type a custom id).
    private let presetModels = [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.0-flash-exp",
        "qwen/qwen2.5-vl-72b-instruct",
        "Custom…",
    ]
    @State private var customModel: String = ""
    @State private var picked: String = "openai/gpt-4o"

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("AI Detection (OpenRouter)")
                .font(.system(size: 16, weight: .semibold))

            Text("Tier 1 = AI coarse regions + OpenCV refine. Tier 2 = OpenCV only "
                 + "(offline / no key). The AI only flags where real spots are; OpenCV "
                 + "computes exact positions and Rf.")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Toggle("Use AI detection when available", isOn: $store.useAI)

            VStack(alignment: .leading, spacing: 6) {
                Text("OpenRouter API key").font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
                HStack {
                    SecureField(keySaved ? "•••••••• (saved in Keychain)" : "sk-or-...", text: $apiKey)
                        .textFieldStyle(.roundedBorder)
                    Button("Save") {
                        guard !apiKey.isEmpty else { return }
                        KeychainHelper.saveAPIKey(apiKey)
                        apiKey = ""; keySaved = true
                    }
                    Button("Clear") {
                        KeychainHelper.deleteAPIKey(); keySaved = false
                    }.disabled(!keySaved)
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Vision model").font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
                Picker("", selection: $picked) {
                    ForEach(presetModels, id: \.self) { Text($0).tag($0) }
                }
                .labelsHidden()
                if picked == "Custom…" {
                    TextField("provider/model-id", text: $customModel)
                        .textFieldStyle(.roundedBorder)
                }
            }

            HStack {
                Spacer()
                Button("Done") {
                    store.openRouterModel = (picked == "Custom…") ? customModel : picked
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 440)
        .onAppear {
            // Initialize picker from stored model.
            if presetModels.contains(store.openRouterModel) {
                picked = store.openRouterModel
            } else {
                picked = "Custom…"; customModel = store.openRouterModel
            }
        }
    }
}
