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
        case .compounds:   return "atom"
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

/// One plate in the reaction time course (its image + per-plate annotations).
struct PlateSnapshot: Identifiable {
    let id = UUID()
    var image: NSImage?
    var sourceImage: NSImage?
    var spots: [Spot] = []
    var calibration = Calibration()
    var autoCandidates: [CGPoint] = []
    var calibrationUserModified = false
    var spotsUserModified = false
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

    // Reaction time course: a sequence of plates. The active one is mirrored into
    // the single-plate @Published fields above (plateImage/spots/calibration/…),
    // so all existing views keep working; inactive plates live as snapshots here.
    @Published var plates: [PlateSnapshot] = []
    @Published var activeIndex: Int = 0
    /// Minutes between consecutive plates in the time course (shown when >1 plate).
    @Published var intervalMinutes: Double = 10

    var plateCount: Int { plates.count }

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
    /// Report model — can differ from the vision (image-recognition) model.
    @Published var reportModel: String =
        UserDefaults.standard.string(forKey: "reportModel") ?? "openai/gpt-4o" {
        didSet { UserDefaults.standard.set(reportModel, forKey: "reportModel") }
    }

    // YOLO spot detector
    @Published var yoloStatus: String = "not_trained"   // not_trained | training | ready
    @Published var yoloTrainedAt: String? = nil          // ISO 8601 string from server
    var useYolo: Bool { yoloStatus == "ready" }

    private var yoloPollingTask: Task<Void, Never>?

    // AI report (M3, spec §10)
    @Published var notebookText: String = ""        // optional lab notebook
    @Published var notebookName: String?
    @Published var reportQuestions: [String] = []   // asked when no notebook
    @Published var reportAnswers: [String] = []     // parallel to reportQuestions
    @Published var reportMarkdown: String?
    @Published var isGeneratingReport: Bool = false
    @Published var reportStatus: String?
    /// Has a key been stored in Keychain?
    var hasOpenRouterKey: Bool { KeychainHelper.hasAPIKey }

    /// Original imported image, kept as the detection source so re-running with
    /// new slider values never compounds re-rectification. Display uses `plateImage`
    /// (becomes the rectified image after a detect).
    private var sourceImage: NSImage?

    var titleBarText: String {
        "\(reactionName) · Plate \(plateIndex) · \(captureChannel)"
    }

    /// Mobile phase = solvent system; shown top-right. Defaults shown when unset.
    var mobilePhaseDisplay: String {
        let s = solventSystem.trimmingCharacters(in: .whitespacesAndNewlines)
        return s.isEmpty ? "default" : s
    }

    /// Auto plate name when the user left it blank: "实验N · YYYY-MM-DD".
    private func autoPlateName() -> String {
        let df = DateFormatter(); df.dateFormat = "yyyy-MM-dd"
        let today = (try? AppDatabase.shared.allExperiments())?
            .filter { Calendar.current.isDateInToday($0.createdAt) }.count ?? 0
        return "实验\(today + 1) · \(df.string(from: Date()))"
    }

    /// Fill in sensible default conditions for any field the user left blank,
    /// so saved experiments are always searchable.
    private func applyDefaultConditions() {
        func d(_ v: String, _ def: String) -> String {
            v.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? def : v
        }
        solventSystem  = d(solventSystem, "EtOAc/Hexanes")
        ratio          = d(ratio, "1:1")
        stationaryPhase = d(stationaryPhase, "Silica gel")
        visualization  = d(visualization, captureChannel)
        plateType      = d(plateType, "Glass-backed")
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
        // Ensure there is an active plate slot (first import creates the sequence).
        let firstImport = plates.isEmpty
        if plates.isEmpty { plates = [PlateSnapshot()]; activeIndex = 0 }
        plateImage = image
        sourceImage = image          // 检测源 = 原图
        // Auto-name on a brand-new experiment (date + 实验N); user can rename in the title bar.
        if firstImport { plateTitle = defaultPlateName() }
        else if let title { plateTitle = title }
        // Reset annotation for the freshly imported plate.
        spots.removeAll()
        selectedSpotID = nil
        calibration = Calibration()
        isSpotMode = false
        calibrationUserModified = false
        spotsUserModified = false
        showDigitalPlate = false
        // Conditions are shared across the time course; only clear on a brand-new run.
        if plates.count <= 1 {
            currentExperimentID = nil
            solventSystem = ""; ratio = ""; stationaryPhase = ""
            visualization = ""; plateType = ""
        }
        saveStatus = nil
        saveActiveDoc()
        rectifyOnImport()   // 导入即正畸: 立即把画布换成正畸后的图
    }

    // MARK: - Reaction time course (multi-plate)

    /// Persist the live (mirrored) active-plate fields back into `plates[activeIndex]`.
    private func saveActiveDoc() {
        guard plates.indices.contains(activeIndex) else { return }
        plates[activeIndex].image = plateImage
        plates[activeIndex].sourceImage = sourceImage
        plates[activeIndex].spots = spots
        plates[activeIndex].calibration = calibration
        plates[activeIndex].autoCandidates = autoCandidates
        plates[activeIndex].calibrationUserModified = calibrationUserModified
        plates[activeIndex].spotsUserModified = spotsUserModified
    }

    /// Load `plates[activeIndex]` into the live mirrored fields.
    private func loadActiveDoc() {
        guard plates.indices.contains(activeIndex) else { return }
        let d = plates[activeIndex]
        plateImage = d.image
        sourceImage = d.sourceImage
        spots = d.spots
        calibration = d.calibration
        autoCandidates = d.autoCandidates
        calibrationUserModified = d.calibrationUserModified
        spotsUserModified = d.spotsUserModified
        selectedSpotID = nil
        showDigitalPlate = false
        plateIndex = activeIndex + 1
    }

    /// Switch which plate is shown on the canvas.
    func selectPlate(_ index: Int) {
        guard plates.indices.contains(index), index != activeIndex else { return }
        saveActiveDoc()
        activeIndex = index
        loadActiveDoc()
    }

    /// Plus-box: import a brand-new plate appended to the end of the sequence.
    func addPlateFromPanel() {
        presentImagePanel { [weak self] image, title in
            guard let self else { return }
            self.saveActiveDoc()
            self.plates.append(PlateSnapshot())
            self.activeIndex = self.plates.count - 1
            self.importImage(image, title: title)
        }
    }

    /// Click an existing thumbnail: import a new image overwriting that slot.
    func overwritePlate(_ index: Int) {
        presentImagePanel { [weak self] image, title in
            guard let self, self.plates.indices.contains(index) else { return }
            self.saveActiveDoc()
            self.activeIndex = index
            self.importImage(image, title: title)
        }
    }

    /// Drag-reorder plates; keeps the same plate active.
    func movePlate(from: Int, to: Int) {
        guard plates.indices.contains(from), from != to else { return }
        saveActiveDoc()
        let activeID = plates[activeIndex].id
        let item = plates.remove(at: from)
        let dest = to > from ? to - 1 : to
        plates.insert(item, at: max(0, min(dest, plates.count)))
        activeIndex = plates.firstIndex(where: { $0.id == activeID }) ?? 0
        plateIndex = activeIndex + 1
    }

    /// Image to show for thumbnail `i` (live mirror for the active one).
    func thumbnailImage(_ i: Int) -> NSImage? {
        i == activeIndex ? plateImage : (plates.indices.contains(i) ? plates[i].image : nil)
    }

    private func presentImagePanel(_ completion: @escaping (NSImage, String?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .tiff, .heic, .image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url, let image = NSImage(contentsOf: url) {
            completion(image, url.deletingPathExtension().lastPathComponent)
        }
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
        spots[index].customLabel = ""        // preset overrides any custom text
        spotsUserModified = true
    }

    /// Set a free-text label on a spot (e.g. a compound name not in the presets).
    func setCustomLabel(_ text: String, for id: Spot.ID) {
        guard let index = spots.firstIndex(where: { $0.id == id }) else { return }
        spots[index].customLabel = text.trimmingCharacters(in: .whitespacesAndNewlines)
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
        let yolo = useYolo
        Task.detached { [weak self] in
            guard let self else { return }
            do {
                guard await client.waitForReady(timeout: 15) else {
                    await self.finishDetection(status: "CV engine did not start. Run: python -m chromalog_cv.server")
                    return
                }
                let result = try await client.detect(imageData: imageData, hatThreshK: k, kneeDeviation: dev,
                                                     useAI: ai, orModel: model, orKey: key,
                                                     useYolo: yolo)
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
            autoCandidates = result.spots.map { CGPoint(x: $0.x, y: $0.y) }
        }

        lastEngineUsed = result.engine_used
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

    /// Auto name: "<yyyy-MM-dd> 实验N" (N = today's saved count + 1).
    private func defaultPlateName() -> String {
        let df = DateFormatter(); df.dateFormat = "yyyy-MM-dd"
        let today = (try? AppDatabase.shared.allExperiments())?
            .filter { Calendar.current.isDateInToday($0.createdAt) }.count ?? 0
        return "\(df.string(from: Date())) 实验\(today + 1)"
    }

    private func rfList(for p: PlateSnapshot) -> [(spot: Spot, rf: Double)] {
        p.spots.map { ($0, p.calibration.rf(forNormalizedY: $0.point.y)) }
            .sorted { $0.1 > $1.1 }
    }

    func saveCurrentPlate() {
        guard plateImage != nil else { return }
        saveActiveDoc()                       // flush live edits into plates[]
        let db = AppDatabase.shared
        let id = currentExperimentID ?? UUID().uuidString
        if plateTitle.trimmingCharacters(in: .whitespaces).isEmpty { plateTitle = defaultPlateName() }
        let titleStr = plateTitle

        do {
            // Persist every plate of the time course + a JSON sidecar describing the series.
            var seriesPlates: [SeriesPlate] = []
            for (i, p) in plates.enumerated() {
                guard let img = p.image else { continue }
                let rectName = try db.saveImage(img, named: "\(id)_p\(i)_rect")
                var srcName: String? = nil
                if let s = p.sourceImage, s !== p.image {
                    srcName = try db.saveImage(s, named: "\(id)_p\(i)_src")
                }
                let sp = p.spots.map { s in
                    SeriesSpot(x: Double(s.point.x), y: Double(s.point.y),
                               label: s.label.rawValue, custom: s.customLabel)
                }
                seriesPlates.append(SeriesPlate(
                    rectFile: rectName, srcFile: srcName,
                    baselineY: Double(p.calibration.baselineY),
                    frontY: Double(p.calibration.frontY), spots: sp))
            }
            let doc = SeriesDoc(count: seriesPlates.count, intervalMinutes: intervalMinutes,
                                plates: seriesPlates)
            if let data = try? JSONEncoder().encode(doc) {
                try db.writeFile(data, name: "\(id)_series.json")
            }

            // Archive thumbnail = the FIRST plate's annotated redraw.
            let first = plates.first
            let firstRf = first.map { rfList(for: $0) } ?? rfResults
            let annotated = PlateExportView.render(
                title: titleStr, date: Date(),
                solventSystem: solventSystem, ratio: ratio, rfResults: firstRf)
            let thumb = annotated ?? (first?.image ?? plateImage!)
            let fileName = try db.saveImage(thumb, named: id)

            let firstCal = first?.calibration ?? calibration
            let record = ExperimentRecord(
                id: id,
                title: titleStr,
                reactionName: reactionName,
                plateIndex: max(1, plates.count),
                channel: captureChannel,
                imageFileName: fileName,
                originalImageFileName: nil,
                baselineY: Double(firstCal.baselineY),
                frontY: Double(firstCal.frontY),
                solventSystem: solventSystem,
                ratio: ratio,
                stationaryPhase: stationaryPhase,
                visualization: visualization,
                plateType: plateType,
                createdAt: Date()
            )
            let spotRecords = (first?.spots ?? spots).map {
                SpotRecord(id: $0.id.uuidString, experimentId: id,
                           x: Double($0.point.x), y: Double($0.point.y),
                           label: $0.label.rawValue, note: $0.note)
            }
            try db.save(record, spots: spotRecords)
            currentExperimentID = id
            saveStatus = plates.count > 1 ? "Saved series (\(plates.count) plates)" : "Saved \(titleStr)"
            // Online learning from this correction (best-effort; never blocks saving).
            if let rectified = plateImage, let rectData = rectified.jpegData() {
                let finals = spots.map { $0.point }
                let cands = autoCandidates
                // 用户改过基线/前沿才把线位置纳入学习
                let bY = calibrationUserModified ? Double(calibration.baselineY) : nil
                let fY = calibrationUserModified ? Double(calibration.frontY) : nil
                Task.detached {
                    await CVClient.shared.learn(rectified: rectData,
                                                finalSpots: finals, autoCandidates: cands,
                                                baselineY: bY, frontY: fY)
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
        // Restore the whole time course if a series sidecar exists.
        if let data = db.readFile("\(record.id)_series.json"),
           let doc = try? JSONDecoder().decode(SeriesDoc.self, from: data),
           !doc.plates.isEmpty {
            loadSeries(doc, record: record)
            return
        }
        // Working image = the saved rectified/rotated plate ("<id>_rect"), the coordinate
        // basis for spots so points line up. Fall back to the redraw thumbnail for old records.
        let displayed = db.loadImage("\(record.id)_rect.png") ?? db.loadImage(record.imageFileName)
        guard let displayed else { return }
        plateImage = displayed
        // Original (for re-detect); falls back to the displayed image if not stored.
        sourceImage = (record.originalImageFileName.flatMap { db.loadImage($0) }) ?? displayed
        autoCandidates = []
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
        // Reset the time-course to this single loaded plate (mirror stays consistent).
        plates = [PlateSnapshot()]
        activeIndex = 0
        saveActiveDoc()
    }

    /// Restore an entire reaction time course from its JSON sidecar.
    private func loadSeries(_ doc: SeriesDoc, record: ExperimentRecord) {
        let db = AppDatabase.shared
        var snaps: [PlateSnapshot] = []
        for sp in doc.plates {
            guard let img = db.loadImage(sp.rectFile) else { continue }
            var snap = PlateSnapshot()
            snap.image = img
            snap.sourceImage = (sp.srcFile.flatMap { db.loadImage($0) }) ?? img
            snap.calibration = Calibration(baselineY: CGFloat(sp.baselineY),
                                           frontY: CGFloat(sp.frontY))
            snap.spots = sp.spots.map {
                Spot(point: CGPoint(x: $0.x, y: $0.y),
                     label: SpotLabel(rawValue: $0.label) ?? .product,
                     customLabel: $0.custom)
            }
            snaps.append(snap)
        }
        guard !snaps.isEmpty else { return }
        plates = snaps
        activeIndex = 0
        intervalMinutes = doc.intervalMinutes
        plateTitle = record.title
        reactionName = record.reactionName
        captureChannel = record.channel
        solventSystem = record.solventSystem
        ratio = record.ratio
        stationaryPhase = record.stationaryPhase
        visualization = record.visualization
        plateType = record.plateType
        currentExperimentID = record.id
        selectedSpotID = nil
        isSpotMode = false
        calibrationUserModified = false
        spotsUserModified = false
        autoCandidates = []
        showDigitalPlate = false
        showArchive = false
        saveStatus = nil
        loadActiveDoc()        // mirror plates[0] into the live fields
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

    // MARK: - AI report (M3, spec §10)

    private func reportData() -> [String: Any] {
        let rf = rfResults.map { item -> [String: Any] in
            var entry: [String: Any] = ["label": item.spot.displayName]
            if item.rf.isFinite { entry["rf"] = (item.rf * 1000).rounded() / 1000 }
            return entry
        }
        return [
            "reaction": reactionName,
            "channel": captureChannel,
            "plate_count": plateCount,
            "interval_minutes": Int(intervalMinutes),
            "conditions": [
                "solvent_system": solventSystem, "ratio": ratio,
                "stationary_phase": stationaryPhase, "visualization": visualization,
                "plate_type": plateType,
            ],
            "rf_table": rf,
        ]
    }

    private func reportPayloadJSON(answers: String) -> String? {
        let body: [String: Any] = ["data": reportData(), "notebook": notebookText, "answers": answers]
        guard let d = try? JSONSerialization.data(withJSONObject: body) else { return nil }
        return String(data: d, encoding: .utf8)
    }

    /// Load an optional lab notebook (.txt / .md). Supersedes the Q&A path.
    func loadNotebook() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.plainText, .text]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url,
           let s = try? String(contentsOf: url, encoding: .utf8) {
            notebookText = s
            notebookName = url.lastPathComponent
            reportQuestions = []; reportAnswers = []
        }
    }

    func clearNotebook() { notebookText = ""; notebookName = nil }

    /// Entry point for the "Generate report" button.
    func startReport() {
        guard hasOpenRouterKey else {
            reportStatus = "Set an OpenRouter key in Settings first"; showSettings = true; return
        }
        if !notebookText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            runGenerateReport(answers: "")                 // have notebook → generate
        } else if reportQuestions.isEmpty {
            requestReportQuestions()                        // no notebook → ask first
        } else {
            let combined = zip(reportQuestions, reportAnswers)
                .map { "Q: \($0)\nA: \($1)" }.joined(separator: "\n")
            runGenerateReport(answers: combined)            // generate from answers
        }
    }

    private func requestReportQuestions() {
        guard let key = KeychainHelper.loadAPIKey(), let payload = reportPayloadJSON(answers: "") else { return }
        isGeneratingReport = true; reportStatus = "Asking clarifying questions…"
        let model = reportModel
        Task.detached { [weak self] in
            guard let self else { return }
            guard await CVClient.shared.waitForReady(timeout: 10) else {
                await self.finishReport(status: "CV engine not running"); return
            }
            do {
                let r = try await CVClient.shared.report(mode: "questions", payloadJSON: payload, model: model, key: key)
                await MainActor.run {
                    if r.ok, let qs = r.questions, !qs.isEmpty {
                        self.reportQuestions = qs
                        self.reportAnswers = Array(repeating: "", count: qs.count)
                        self.reportStatus = "Answer the questions, then Generate"
                    } else {
                        self.reportStatus = r.error ?? "No questions returned"
                    }
                    self.isGeneratingReport = false
                }
            } catch { await self.finishReport(status: error.localizedDescription) }
        }
    }

    private func runGenerateReport(answers: String) {
        guard let key = KeychainHelper.loadAPIKey(), let payload = reportPayloadJSON(answers: answers) else { return }
        isGeneratingReport = true; reportStatus = "Generating report…"
        let model = reportModel
        Task.detached { [weak self] in
            guard let self else { return }
            guard await CVClient.shared.waitForReady(timeout: 10) else {
                await self.finishReport(status: "CV engine not running"); return
            }
            do {
                let r = try await CVClient.shared.report(mode: "report", payloadJSON: payload, model: model, key: key)
                await MainActor.run {
                    if r.ok, let md = r.markdown {
                        self.reportMarkdown = md; self.reportStatus = nil
                    } else {
                        self.reportStatus = r.error ?? "Report failed"
                    }
                    self.isGeneratingReport = false
                }
            } catch { await self.finishReport(status: error.localizedDescription) }
        }
    }

    @MainActor private func finishReport(status: String) {
        reportStatus = status; isGeneratingReport = false
    }

    // MARK: - YOLO model management

    func startYoloTraining() {
        // Cancel any prior polling task before starting a new one.
        yoloPollingTask?.cancel()
        yoloPollingTask = Task { @MainActor in
            do {
                guard await CVClient.shared.waitForReady(timeout: 5) else {
                    yoloStatus = "not_trained"; return
                }
                try await CVClient.shared.trainYolo()
                yoloStatus = "training"
                await pollYoloStatus()
            } catch {
                yoloStatus = "not_trained"
            }
        }
    }

    /// Poll until the server reports non-training, or until 72 iterations (~6 min) elapsed.
    private func pollYoloStatus() async {
        let maxPolls = 72
        var polls = 0
        while yoloStatus == "training" && polls < maxPolls {
            do {
                try await Task.sleep(nanoseconds: 5_000_000_000)
                polls += 1
                let info = try await CVClient.shared.yoloModelStatus()
                await MainActor.run {
                    self.yoloStatus = info.status
                    self.yoloTrainedAt = info.trained_at
                }
            } catch {
                break
            }
        }
        // If we timed out still training, mark as error.
        if yoloStatus == "training" {
            yoloStatus = "not_trained"
        }
    }

    func refreshYoloStatus() {
        Task {
            guard let info = try? await CVClient.shared.yoloModelStatus() else { return }
            await MainActor.run {
                self.yoloStatus = info.status
                self.yoloTrainedAt = info.trained_at
            }
        }
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
