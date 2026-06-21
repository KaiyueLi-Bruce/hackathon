import SwiftUI

/// Context inspector (spec §6): a segmented control switches Results / Conditions
/// / AI over a card stack. M0 provides the segmented control and placeholder
/// cards; real data lands in M1–M3.
struct InspectorView: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $store.inspectorTab) {
                ForEach(InspectorTab.allCases) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(12)

            Divider()

            ScrollView {
                VStack(spacing: 12) {
                    switch store.inspectorTab {
                    case .results:    ResultsSection()
                    case .conditions: ConditionsSection()
                    case .ai:         AISection()
                    }
                }
                .padding(12)
            }
        }
        .background(.ultraThinMaterial)
        .onAppear { store.refreshModelInfo() }
    }
}

// MARK: - Section placeholders (M0)

private struct ResultsSection: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        if store.hasImage {
            DetectionTuningCard()
        }

        Card(title: "Rf values") {
            if store.spots.isEmpty {
                EmptyHint(text: store.isSpotMode
                          ? "Tap the plate to add spots"
                          : "Enable Spot, then tap the plate")
            } else {
                VStack(spacing: 8) {
                    ForEach(store.rfResults, id: \.spot.id) { result in
                        SpotRow(spot: result.spot, rf: result.rf)
                    }
                }
            }
        }

        Card(title: "Co-spot check") {
            CoSpotRow(delta: store.coSpotDelta)
        }

        if !store.spots.isEmpty {
            Button(role: .destructive) { store.clearSpots() } label: {
                Label("Clear spots", systemImage: "trash").frame(maxWidth: .infinity)
            }
            .controlSize(.small)
        }
    }
}

/// Two live-tuning sliders that re-run auto-detect on release (spec Appendix D / debug).
private struct DetectionTuningCard: View {
    @EnvironmentObject private var store: AppStore

    private var engineLabel: String {
        switch store.lastEngineUsed {
        case "ai+opencv": return "Engine: AI + OpenCV"
        case "yolo":      return "Engine: YOLO"
        case "opencv":    return "Engine: OpenCV"
        default:          return store.useAI ? "AI on (run Auto-detect)" : "OpenCV (offline)"
        }
    }

    var body: some View {
        Card(title: "Auto-detect tuning") {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 6) {
                    Image(systemName: store.useAI ? "sparkles" : "cpu")
                        .font(.system(size: 11))
                        .foregroundStyle(store.useAI ? Palette.accent : .secondary)
                    Text(engineLabel)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Settings") { store.showSettings = true }
                        .controlSize(.mini)
                }
                if store.modelTrainedCount > 0 {
                    Text("Learned from \(store.modelTrainedCount) corrections")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                TuneSlider(title: "Sensitivity",
                           caption: "Threshold k · higher = more conservative, fewer spots",
                           value: $store.hatThreshK, range: 2.0...7.0, step: 0.5,
                           disabled: store.isAutoDetecting) {
                    store.runAutoDetect()
                }
                TuneSlider(title: "Spot count",
                           caption: "Area knee · higher = more spots detected",
                           value: $store.kneeDeviation, range: 1.0...10.0, step: 0.5,
                           disabled: store.isAutoDetecting) {
                    store.runAutoDetect()
                }
            }
        }
    }
}

private struct TuneSlider: View {
    let title: String
    let caption: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    var disabled: Bool = false
    let onCommit: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(title).font(.system(size: 12, weight: .medium))
                Spacer()
                Text(String(format: "%.1f", value))
                    .font(.tabular(12, weight: .medium))
                    .foregroundStyle(.secondary)
            }
            Slider(value: $value, in: range, step: step) { editing in
                if !editing { onCommit() }   // re-run detection on release
            }
            .tint(Palette.accent)
            .disabled(disabled)
            Text(caption)
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
        }
    }
}

private struct SpotRow: View {
    @EnvironmentObject private var store: AppStore
    let spot: Spot
    let rf: Double

    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(spot.label.color).frame(width: 9, height: 9)

            Menu {
                ForEach(SpotLabel.allCases, id: \.self) { label in
                    Button(label.rawValue) { store.setLabel(label, for: spot.id) }
                }
            } label: {
                Text(spot.label.rawValue).font(.system(size: 12))
            }
            .menuStyle(.borderlessButton)
            .fixedSize()

            Spacer()

            Text(rf.rfDisplay)
                .font(.tabular(12, weight: .medium))
                .foregroundStyle(rf.isRfInRange ? .primary : Color(Palette.coral))

            Button { store.deleteSpot(spot.id) } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)
        }
        .padding(6)
        .background(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(store.selectedSpotID == spot.id ? Palette.accent.opacity(0.12) : .clear)
        )
        .contentShape(Rectangle())
        .onTapGesture { store.selectedSpotID = spot.id }
    }
}

private struct CoSpotRow: View {
    let delta: Double?

    var body: some View {
        HStack {
            if let delta {
                let aligned = delta < 0.05
                Text(aligned ? "Aligned (ΔRf \(delta.rfDisplay))" : "Distinct (ΔRf \(delta.rfDisplay))")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                Spacer()
                Image(systemName: aligned ? "checkmark.circle.fill" : "xmark.circle")
                    .foregroundStyle(aligned ? Color(Palette.teal) : .secondary)
            } else {
                Text("Add ≥2 labeled spots")
                    .font(.system(size: 12)).foregroundStyle(.secondary)
                Spacer()
                Image(systemName: "minus.circle").foregroundStyle(.secondary)
            }
        }
    }
}

private struct EmptyHint: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.system(size: 12))
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct ConditionsSection: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        Card(title: "Conditions") {
            VStack(alignment: .leading, spacing: 10) {
                EditableField(label: "Solvent system", placeholder: "EtOAc / hexanes", text: $store.solventSystem)
                EditableField(label: "Ratio", placeholder: "1:3", text: $store.ratio)
                EditableField(label: "Stationary phase", placeholder: "Silica gel", text: $store.stationaryPhase)
                EditableField(label: "Visualization", placeholder: "UV254 / KMnO₄", text: $store.visualization)
                EditableField(label: "Plate type", placeholder: "Glass-backed", text: $store.plateType)
            }
        }
    }
}

private struct EditableField: View {
    let label: String
    let placeholder: String
    @Binding var text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
            TextField(placeholder, text: $text)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12))
        }
    }
}

private struct AISection: View {
    var body: some View {
        Card(title: "AI report") {
            VStack(alignment: .leading, spacing: 10) {
                Text("Generate a structured report from the detected Rf values and conditions.")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                Button {
                } label: {
                    HStack {
                        Image(systemName: "sparkles")
                        Text("Generate AI report")
                    }
                    .frame(maxWidth: .infinity)
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                .tint(Palette.accent)
                .disabled(true)
            }
        }
    }
}

// MARK: - Reusable card primitives

private struct Card<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor).opacity(0.6))
        )
    }
}

