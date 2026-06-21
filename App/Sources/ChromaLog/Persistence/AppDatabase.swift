import Foundation
import AppKit
import GRDB

/// Local-first SQLite store (spec §5/§9) backed by GRDB. The database and plate
/// images live in Application Support; nothing leaves the machine.
final class AppDatabase {
    static let shared = try! AppDatabase()

    let dbQueue: DatabaseQueue
    private let imagesDir: URL

    init() throws {
        let fm = FileManager.default
        let appSupport = try fm.url(for: .applicationSupportDirectory,
                                    in: .userDomainMask,
                                    appropriateFor: nil,
                                    create: true)
        let root = appSupport.appendingPathComponent("ChromaLog", isDirectory: true)
        imagesDir = root.appendingPathComponent("images", isDirectory: true)
        try fm.createDirectory(at: imagesDir, withIntermediateDirectories: true)

        let dbURL = root.appendingPathComponent("chromalog.sqlite")
        dbQueue = try DatabaseQueue(path: dbURL.path)
        try migrator.migrate(dbQueue)
    }

    private var migrator: DatabaseMigrator {
        var migrator = DatabaseMigrator()
        migrator.registerMigration("v1") { db in
            try db.create(table: "experiment") { t in
                t.column("id", .text).primaryKey()
                t.column("title", .text).notNull()
                t.column("reactionName", .text).notNull()
                t.column("plateIndex", .integer).notNull()
                t.column("channel", .text).notNull()
                t.column("imageFileName", .text).notNull()
                t.column("baselineY", .double).notNull()
                t.column("frontY", .double).notNull()
                t.column("solventSystem", .text).notNull().defaults(to: "")
                t.column("ratio", .text).notNull().defaults(to: "")
                t.column("stationaryPhase", .text).notNull().defaults(to: "")
                t.column("visualization", .text).notNull().defaults(to: "")
                t.column("plateType", .text).notNull().defaults(to: "")
                t.column("createdAt", .datetime).notNull()
            }
            try db.create(table: "spot") { t in
                t.column("id", .text).primaryKey()
                t.column("experimentId", .text).notNull()
                    .indexed()
                    .references("experiment", onDelete: .cascade)
                t.column("x", .double).notNull()
                t.column("y", .double).notNull()
                t.column("label", .text).notNull()
                t.column("note", .text).notNull().defaults(to: "")
            }
        }
        migrator.registerMigration("v2") { db in
            try db.alter(table: "experiment") { t in
                t.add(column: "originalImageFileName", .text)
            }
        }
        return migrator
    }

    // MARK: - Image files

    func saveImage(_ image: NSImage, named name: String) throws -> String {
        guard
            let tiff = image.tiffRepresentation,
            let rep = NSBitmapImageRep(data: tiff),
            let png = rep.representation(using: .png, properties: [:])
        else {
            throw NSError(domain: "AutoChem", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Could not encode image"])
        }
        let fileName = "\(name).png"
        try png.write(to: imagesDir.appendingPathComponent(fileName))
        return fileName
    }

    func loadImage(_ fileName: String) -> NSImage? {
        NSImage(contentsOf: imagesDir.appendingPathComponent(fileName))
    }

    // MARK: - Arbitrary sidecar files (e.g. the series JSON)

    func writeFile(_ data: Data, name: String) throws {
        try data.write(to: imagesDir.appendingPathComponent(name))
    }

    func readFile(_ name: String) -> Data? {
        try? Data(contentsOf: imagesDir.appendingPathComponent(name))
    }

    /// Number of plates in a saved time course (1 if it's a single plate).
    func seriesCount(_ id: String) -> Int {
        guard let d = readFile("\(id)_series.json"),
              let doc = try? JSONDecoder().decode(SeriesDoc.self, from: d) else { return 1 }
        return max(1, doc.count)
    }

    // MARK: - Queries

    func save(_ experiment: ExperimentRecord, spots: [SpotRecord]) throws {
        try dbQueue.write { db in
            try experiment.save(db)
            try SpotRecord
                .filter(Column("experimentId") == experiment.id)
                .deleteAll(db)
            for spot in spots { try spot.insert(db) }
        }
    }

    func allExperiments() throws -> [ExperimentRecord] {
        try dbQueue.read { db in
            try ExperimentRecord.order(Column("createdAt").desc).fetchAll(db)
        }
    }

    func search(_ query: String) throws -> [ExperimentRecord] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return try allExperiments() }
        let like = "%\(trimmed)%"
        return try dbQueue.read { db in
            try ExperimentRecord
                .filter(
                    Column("title").like(like) ||
                    Column("solventSystem").like(like) ||
                    Column("visualization").like(like) ||
                    Column("reactionName").like(like)
                )
                .order(Column("createdAt").desc)
                .fetchAll(db)
        }
    }

    func spots(for experimentID: String) throws -> [SpotRecord] {
        try dbQueue.read { db in
            try SpotRecord
                .filter(Column("experimentId") == experimentID)
                .fetchAll(db)
        }
    }

    func deleteExperiment(_ id: String) throws {
        _ = try dbQueue.write { db in
            try ExperimentRecord.deleteOne(db, key: id)
        }
    }
}
