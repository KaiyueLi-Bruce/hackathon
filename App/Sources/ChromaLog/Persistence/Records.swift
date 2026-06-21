import Foundation
import GRDB

/// One saved plate/experiment (spec §9, Plate + Conditions flattened for the
/// hackathon single-plate scope; multi-plate per experiment is a later step).
struct ExperimentRecord: Codable, Identifiable, FetchableRecord, PersistableRecord {
    static let databaseTableName = "experiment"

    var id: String
    var title: String
    var reactionName: String
    var plateIndex: Int
    var channel: String
    var imageFileName: String          // annotated redraw (archive thumbnail)
    var originalImageFileName: String? // original photo (restored on open)
    var baselineY: Double
    var frontY: Double
    var solventSystem: String
    var ratio: String
    var stationaryPhase: String
    var visualization: String
    var plateType: String
    var createdAt: Date
}

// MARK: - Reaction time-course series (stored as a JSON sidecar "<id>_series.json")

/// A whole time-course (multiple plates) persisted alongside an experiment.
struct SeriesDoc: Codable {
    var count: Int
    var intervalMinutes: Double
    var plates: [SeriesPlate]
}

struct SeriesPlate: Codable {
    var rectFile: String         // displayed/rectified image (coordinate basis)
    var srcFile: String?         // original photo (for re-detect)
    var baselineY: Double
    var frontY: Double
    var spots: [SeriesSpot]
}

struct SeriesSpot: Codable {
    var x: Double
    var y: Double
    var label: String
    var custom: String           // free-text custom label (preserved across save/load)
}

/// A persisted spot in normalized image coordinates (spec §9).
struct SpotRecord: Codable, Identifiable, FetchableRecord, PersistableRecord {
    static let databaseTableName = "spot"

    var id: String
    var experimentId: String
    var x: Double
    var y: Double
    var label: String
    var note: String
}
