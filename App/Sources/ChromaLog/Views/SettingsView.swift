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

    private let reportPresetModels = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.0-flash-exp",
        "Custom…",
    ]
    @State private var customReportModel: String = ""
    @State private var pickedReport: String = "openai/gpt-4o"

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
                Text("Vision model (image recognition)").font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
                Picker("", selection: $picked) {
                    ForEach(presetModels, id: \.self) { Text($0).tag($0) }
                }
                .labelsHidden()
                if picked == "Custom…" {
                    TextField("provider/model-id", text: $customModel)
                        .textFieldStyle(.roundedBorder)
                }
                Text("Tip: use a fast vision model (e.g. gemini-2.0-flash); avoid 'reasoning' models.")
                    .font(.system(size: 9)).foregroundStyle(.tertiary)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Report model (can differ from vision)").font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
                Picker("", selection: $pickedReport) {
                    ForEach(reportPresetModels, id: \.self) { Text($0).tag($0) }
                }
                .labelsHidden()
                if pickedReport == "Custom…" {
                    TextField("provider/model-id", text: $customReportModel)
                        .textFieldStyle(.roundedBorder)
                }
            }

            Divider()

            GroupBox("YOLO Spot Detector") {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(yoloDotColor)
                            .frame(width: 8, height: 8)
                        Text(yoloStatusLabel)
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                        if let dateStr = yoloTrainedAtFormatted {
                            Text("(trained \(dateStr))")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                    }

                    Button {
                        store.startYoloTraining()
                    } label: {
                        if store.yoloStatus == "training" {
                            HStack(spacing: 6) {
                                ProgressView().controlSize(.small)
                                Text("Training… (~5 min)")
                            }
                        } else {
                            Text("Re-train YOLO (YOLOv8n)")
                        }
                    }
                    .disabled(store.yoloStatus == "training")

                    Text("Trains on synthetic TLC data locally. Requires: pip install ultralytics")
                        .font(.system(size: 9)).foregroundStyle(.tertiary)
                }
            }

            HStack {
                Spacer()
                Button("Done") {
                    store.openRouterModel = (picked == "Custom…") ? customModel : picked
                    store.reportModel = (pickedReport == "Custom…") ? customReportModel : pickedReport
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
            if reportPresetModels.contains(store.reportModel) {
                pickedReport = store.reportModel
            } else {
                pickedReport = "Custom…"; customReportModel = store.reportModel
            }
            store.refreshYoloStatus()
        }
    }

    /// Parses `store.yoloTrainedAt` (ISO 8601) and returns a `yyyy-MM-dd` string, or nil.
    private var yoloTrainedAtFormatted: String? {
        guard let raw = store.yoloTrainedAt else { return nil }
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        var date = isoFormatter.date(from: raw)
        if date == nil {
            // Fallback without fractional seconds.
            isoFormatter.formatOptions = [.withInternetDateTime]
            date = isoFormatter.date(from: raw)
        }
        guard let d = date else { return nil }
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd"
        return df.string(from: d)
    }

    private var yoloDotColor: Color {
        switch store.yoloStatus {
        case "ready":    return .green
        case "training": return .yellow
        default:         return .gray
        }
    }

    private var yoloStatusLabel: String {
        switch store.yoloStatus {
        case "ready":    return "Ready"
        case "training": return "Training…"
        default:         return "Not trained"
        }
    }
}
