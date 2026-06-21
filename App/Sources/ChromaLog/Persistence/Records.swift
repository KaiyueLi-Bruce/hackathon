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
