import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Left icon-rail navigation sections.
enum NavSection: String, CaseIterable, Identifiable {
    case experiments
    case plates
    case compounds
    case search

    var id: String { rawValue }

    var title: String {
        switch self {
        case .experiments: return "Experiments"
        case .plates:      return "All Plates"
        case .compounds:   return "Compounds"
        case .search:      return "Search"
        }
    }

    var systemImage: String {
        switch self {
        case .experiments: return "flask"
        case .plates:      return "rectangle.stack"
        case .compounds:   return "fingerprint"
        case .search:      return "magnifyingglass"
        }
    }
}

/// Right inspector context tabs.
enum InspectorTab: String, CaseIterable, Identifiable {
    case results = "Results"
    case conditions = "Conditions"
    case ai = "AI"

    var id: String { rawValue }
}

/// Central observable state for the app shell (M0 scope).
@MainActor
final class AppStore: ObservableObject {
    // Navigation & layout
    @Published var selectedSection: NavSection = .experiments
    @Published var inspectorTab: InspectorTab = .results
    @Published var showRail: Bool = true
    @Published var showInspector: Bool = true

    // Current plate (M0: just the imported image + a display title)
    @Published var plateImage: NSImage?
    @Published var plateTitle: String = "New Plate"
    @Published var captureChannel: String = "UV254"
    @Published var reactionName: String = "Reaction A"
    @Published var plateIndex: Int = 1

    // Calibration & spots (M1)
    @Published var calibration = Calibration()
    @Published var spots: [Spot] = []
    @Published var selectedSpotID: Spot.ID?
    @Published var isSpotMode: Bool = false
    @Published var nextLabel: SpotLabel = .sm

    // Conditions (M2)
    @Published var solventSystem: String = ""
    @Published var ratio: String = ""
    @Published var stationaryPhase: String = ""
    @Published var visualization: String = ""
    @Published var plateType: String = ""

    // Persistence / views (M2)
    @Published var currentExperimentID: String?
    @Published var showDigitalPlate: Bool = false
    @Published var showArchive: Bool = false
    @Published var experiments: [ExperimentRecord] = []
    @Published var archiveQuery: String = ""
    @Published var saveStatus: String?
    @Published var isAutoDetecting: Bool = false
    @Published var detectWarnings: [String] = []
    /// Snapshot of auto-detected spot centroids (before user edits) — training negatives source.
    @Published var autoCandidates: [CGPoint] = []
    /// Number of correction samples the engine has learned from (for the inspector status line).
    @Published var modelTrainedCount: Int = 0

    /// Set to true when the user manually drags a reference line after import.
    /// Prevents subsequent auto-detect runs from overwriting the user's calibration.
    var calibrationUserModified: Bool = false
    /// Set to true when the user manually adds, moves, or deletes a spot after import.
    /// Prevents subsequent auto-detect runs from overwriting the user's spots.
    var spotsUserModified: Bool = false

    // Auto-detect tuning (passed to CV sidecar; defaults match Config)
    @Published var hatThreshK: Double = 4.0      // 越大越保守, 标越少
    @Published var kneeDeviation: Double = 5.0   // 越大标越多

    // AI detection (OpenRouter) — three-tier: AI -> OpenCV fallback
    @Published var showSettings: Bool = false
    @Published var lastEngineUsed: String?       // opencv | ai+opencv | yolo
    @Published var useAI: Bool = UserDefaults.standard.bool(forKey: "useAI") {
        didSet { UserDefaults.standard.set(useAI, forKey: "useAI") }
    }
    @Published var openRouterModel: String =
        UserDefaults.standard.string(forKey: "orModel") ?? "openai/gpt-4o" {
        didSet { UserDefaults.standard.set(openRouterModel, forKey: "orModel") }
    }
    /// Has a key been stored in Keychain?
    var hasOpenRouterKey: Bool { KeychainHelper.hasAPIKey }

    /// Original imported image, kept as the detection source so re-running with
    /// new slider values never compounds re-rectification. Display uses `plateImage`
    /// (becomes the rectified image after a detect).
    private var sourceImage: NSImage?

    var titleBarText: String {
        "\(reactionName) · Plate \(plateIndex) · \(captureChannel)"
    }

    var hasImage: Bool { plateImage != nil }

    // MARK: - Rf

    /// Spots paired with their computed Rf, ordered top-of-plate first
    /// (highest Rf first), matching how a chemist reads a plate.
    var rfResults: [(spot: Spot, rf: Double)] {
        spots
            .map { ($0, calibration.rf(forNormalizedY: $0.point.y)) }
            .sorted { $0.1 > $1.1 }
    }

    /// Smallest Rf gap between two differently-labeled spots — a lightweight
    /// co-spot alignment hint (spec §6 "Co-spot check").
    var coSpotDelta: Double? {
        let results = rfResults
        var best: Double?
        for i in results.indices {
            for j in (i + 1)..<results.count where results[i].spot.label != results[j].spot.label {
                let d = abs(results[i].rf - results[j].rf)
                if best == nil || d < best! { best = d }
            }
        }
        return best
    }

    // MARK: - Layout helpers

    func enterFocusMode() {
        showRail = false
        showInspector = false
    }

    func exitFocusMode() {
        showRail = true
        showInspector = true
    }

    var isFocusMode: Bool { !showRail && !showInspector }

    // MARK: - Image import

    func importImage(_ image: NSImage, title: String? = nil) {
        plateImage = image
        sourceImage = image          // 检测源 = 原图
        if let title { plateTitle = title }
        // Reset annotation for the freshly imported plate.
        spots.removeAll()
        selectedSpotID = nil
        calibration = Calibration()
        isSpotMode = false
        calibrationUserModified = false
        spotsUserModified = false
        showDigitalPlate = false
        currentExperimentID = nil
        solventSystem = ""; ratio = ""; stationaryPhase = ""
        visualization = ""; plateType = ""
        saveStatus = nil
        rectifyOnImport()   // 导入即正畸: 立即把画布换成正畸后的图
    }

    /// Calls the sidecar to rectify the freshly imported image and swaps the
    /// displayed plate to the rectified result (spec 附录 D: 导入即显示正畸图).
    private func rectifyOnImport() {
        guard let image = sourceImage, let data = image.jpegData() else { return }
        let client = CVClient.shared
        Task.detached { [weak self] in
            guard let self else { return }
            guard await client.waitForReady(timeout: 8) else { return }
            guard let res = try? await client.rectify(imageData: data),
                  let b64 = res.image_b64,
                  let pngData = Data(base64Encoded: b64),
                  let rectified = NSImage(data: pngData) else { return }
            await MainActor.run {
                // 仅当这张原图仍是当前图时才替换 (防快速连续导入串图)
                if self.sourceImage === image { self.plateImage = rectified }
            }
        }
    }

    // MARK: - Rotation

    /// Rotate the plate 90° (manual orientation fix, e.g. landscape plates).
    /// Rotates both the displayed image and the detection source; clears
    /// calibration/spots since rotation changes the development direction.
    func rotatePlate(clockwise: Bool) {
        guard let img = plateImage else { return }
        plateImage = img.rotated90(clockwise: clockwise)
        if let s = sourceImage { sourceImage = s.rotated90(clockwise: clockwise) }
        spots.removeAll()
        selectedSpotID = nil
        calibration = Calibration()
        showDigitalPlate = false
        saveStatus = clockwise ? "Rotated 90° right" : "Rotated 90° left"
    }

    // MARK: - Spot management (M1)

    func addSpot(atNormalized point: CGPoint) {
        let clamped = CGPoint(x: min(max(point.x, 0), 1), y: min(max(point.y, 0), 1))
        let spot = Spot(point: clamped, label: nextLabel)
        spots.append(spot)
        selectedSpotID = spot.id
        spotsUserModified = true
    }

    func moveSpot(_ id: Spot.ID, toNormalized point: CGPoint) {
        guard let index = spots.firstIndex(where: { $0.id == id }) else { return }
        spots[index].point = CGPoint(x: min(max(point.x, 0), 1), y: min(max(point.y, 0), 1))
        spotsUserModified = true
    }

    func setLabel(_ label: SpotLabel, for id: Spot.ID) {
        guard let index = spots.firstIndex(where: { $0.id == id }) else { return }
        spots[index].label = label
        spotsUserModified = true
    }

    func deleteSpot(_ id: Spot.ID) {
        spots.removeAll { $0.id == id }
        if selectedSpotID == id { selectedSpotID = nil }
        spotsUserModified = true
    }

    func clearSpots() {
        spots.removeAll()
        selectedSpotID = nil
        spotsUserModified = false   // clear is a reset, not a user override
    }

    func resetCalibration() {
        calibration = Calibration()
    }

    func importImage(from url: URL) {
        guard let image = NSImage(contentsOf: url) else { return }
        importImage(image, title: url.deletingPathExtension().lastPathComponent)
    }

    /// Presents an open panel for picking an image file.
    func presentImportPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .tiff, .heic, .image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            importImage(from: url)
        }
    }

    func refreshModelInfo() {
        Task { @MainActor in
            if let info = await CVClient.shared.modelInfo() {
                self.modelTrainedCount = info.n_samples
            }
        }
    }

    // MARK: - Auto-detect (M5)

    func runAutoDetect() {
        guard let image = sourceImage ?? plateImage, let imageData = image.jpegData() else { return }
        guard !isAutoDetecting else { return }
        isAutoDetecting = true
        detectWarnings = []
        saveStatus = "Connecting to CV engine..."

        let client = CVClient.shared
        let k = hatThreshK
        let dev = kneeDeviation
        let ai = useAI
        let model = openRouterModel
        let key = useAI ? KeychainHelper.loadAPIKey() : nil
        Task.detached { [weak self] in
            guard let self else { return }
            do {
                guard await client.waitForReady(timeout: 15) else {
                    await self.finishDetection(status: "CV engine did not start. Run: python -m chromalog_cv.server")
                    return
                }
                let result = try await client.detect(imageData: imageData, hatThreshK: k, kneeDeviation: dev,
                                                     useAI: ai, orModel: model, orKey: key)
                await self.applyDetection(result)
            } catch {
                await self.finishDetection(status: error.localizedDescription)
            }
        }
    }

    @MainActor
    private func finishDetection(status: String) {
        saveStatus = status
        isAutoDetecting = false
    }

    @MainActor
    private func applyDetection(_ result: CVDetectResponse) {
        // 显示正畸后的图 (坐标基准), 不再显示原图; 斑点/基线坐标均对齐此图
        if let b64 = result.image_b64,
           let data = Data(base64Encoded: b64),
           let rectified = NSImage(data: data) {
            plateImage = rectified
        }

        // Only update calibration if the user hasn't manually adjusted the lines.
        if !calibrationUserModified {
            if let baselineY = result.baseline_y_norm {
                calibration.baselineY = CGFloat(baselineY)
            }
            if let frontY = result.front_y_norm {
                calibration.frontY = CGFloat(frontY)
            }
        }

        // Only update spots if the user hasn't manually edited them.
        if !spotsUserModified {
            spots.removeAll()
            for cvSpot in result.spots {
                spots.append(Spot(
                    point: CGPoint(x: cvSpot.x, y: cvSpot.y),
                    label: .product
                ))
            }
        }

        lastEngineUsed = result.engine_used
        autoCandidates = result.spots.map { CGPoint(x: $0.x, y: $0.y) }
        detectWarnings = result.warnings
        if !result.warnings.isEmpty {
            saveStatus = "Detection completed with \(result.warnings.count) warning(s)"
        } else {
            saveStatus = "Auto-detected \(result.spots.count) spot(s) in \(result.n_lanes) lane(s)"
        }
        isAutoDetecting = false
    }

    /// Handles a SwiftUI `onDrop` of image data or a file URL.
    func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }

        if provider.canLoadObject(ofClass: NSImage.self) {
            provider.loadObject(ofClass: NSImage.self) { [weak self] object, _ in
                guard let image = object as? NSImage else { return }
                Task { @MainActor in self?.importImage(image) }
            }
            return true
        }

        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier) { [weak self] item, _ in
                guard
                    let data = item as? Data,
                    let url = URL(dataRepresentation: data, relativeTo: nil)
                else { return }
                Task { @MainActor in self?.importImage(from: url) }
            }
            return true
        }

        return false
    }

    // MARK: - Persistence (M2)

    func saveCurrentPlate() {
        guard plateImage != nil else { return }
        let db = AppDatabase.shared
        let id = currentExperimentID ?? UUID().uuidString

        // Render the annotated digital-plate image for the archive thumbnail.
        let titleStr = plateTitle.isEmpty ? "Untitled Plate" : plateTitle
        let annotated = PlateExportView.render(
            title: titleStr,
            date: Date(),
            solventSystem: solventSystem,
            ratio: ratio,
            rfResults: rfResults
        )
        let thumbnailImage = annotated ?? plateImage!

        do {
            // Save annotated redraw as the primary image (archive thumbnail).
            let fileName = try db.saveImage(thumbnailImage, named: id)

            // Save original photo separately so it can be restored on open.
            var originalFileName: String? = nil
            if let src = sourceImage {
                originalFileName = try db.saveImage(src, named: "\(id)_src")
            }

            let record = ExperimentRecord(
                id: id,
                title: titleStr,
                reactionName: reactionName,
                plateIndex: plateIndex,
                channel: captureChannel,
                imageFileName: fileName,
                originalImageFileName: originalFileName,
                baselineY: Double(calibration.baselineY),
                frontY: Double(calibration.frontY),
                solventSystem: solventSystem,
                ratio: ratio,
                stationaryPhase: stationaryPhase,
                visualization: visualization,
                plateType: plateType,
                createdAt: Date()
            )
            let spotRecords = spots.map {
                SpotRecord(id: $0.id.uuidString, experimentId: id,
                           x: Double($0.point.x), y: Double($0.point.y),
                           label: $0.label.rawValue, note: $0.note)
            }
            try db.save(record, spots: spotRecords)
            currentExperimentID = id
            saveStatus = "Saved \(record.title)"
            // Online learning from this correction (best-effort; never blocks saving).
            if let rectified = plateImage, let rectData = rectified.jpegData() {
                let finals = spots.map { $0.point }
                let cands = autoCandidates
                Task.detached {
                    await CVClient.shared.learn(rectified: rectData,
                                                finalSpots: finals, autoCandidates: cands)
                    if let info = await CVClient.shared.modelInfo() {
                        await MainActor.run { self.modelTrainedCount = info.n_samples }
                    }
                }
            }
            refreshExperiments()
        } catch {
            saveStatus = "Save failed: \(error.localizedDescription)"
        }
    }

    func refreshExperiments() {
        experiments = (try? AppDatabase.shared.search(archiveQuery)) ?? []
    }

    func loadExperiment(_ record: ExperimentRecord) {
        let db = AppDatabase.shared
        // Prefer original photo; fall back to the stored thumbnail if absent.
        let originalFileName = record.originalImageFileName ?? record.imageFileName
        guard let image = db.loadImage(originalFileName) else { return }
        plateImage = image
        sourceImage = image
        plateTitle = record.title
        reactionName = record.reactionName
        plateIndex = record.plateIndex
        captureChannel = record.channel
        calibration = Calibration(baselineY: CGFloat(record.baselineY),
                                  frontY: CGFloat(record.frontY))
        solventSystem = record.solventSystem
        ratio = record.ratio
        stationaryPhase = record.stationaryPhase
        visualization = record.visualization
        plateType = record.plateType
        let loaded = (try? db.spots(for: record.id)) ?? []
        spots = loaded.map {
            Spot(point: CGPoint(x: $0.x, y: $0.y),
                 label: SpotLabel(rawValue: $0.label) ?? .product,
                 note: $0.note)
        }
        currentExperimentID = record.id
        selectedSpotID = nil
        isSpotMode = false
        calibrationUserModified = false
        spotsUserModified = false
        showDigitalPlate = false
        showArchive = false
        saveStatus = nil
    }

    func deleteExperiment(_ record: ExperimentRecord) {
        try? AppDatabase.shared.deleteExperiment(record.id)
        if currentExperimentID == record.id { currentExperimentID = nil }
        refreshExperiments()
    }

    func openArchive() {
        refreshExperiments()
        showArchive = true
    }
}

extension NSImage {
    /// Returns a copy rotated 90° (clockwise = right, else left).
    func rotated90(clockwise: Bool) -> NSImage {
        let newSize = NSSize(width: size.height, height: size.width)
        let out = NSImage(size: newSize)
        out.lockFocus()
        if let ctx = NSGraphicsContext.current {
            ctx.imageInterpolation = .high
            let t = NSAffineTransform()
            t.translateX(by: newSize.width / 2, yBy: newSize.height / 2)
            t.rotate(byDegrees: clockwise ? -90 : 90)
            t.translateX(by: -size.width / 2, yBy: -size.height / 2)
            t.concat()
            draw(at: .zero, from: NSRect(origin: .zero, size: size),
                 operation: .copy, fraction: 1.0)
        }
        out.unlockFocus()
        return out
    }
}
