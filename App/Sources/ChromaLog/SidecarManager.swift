import Foundation

final class SidecarManager {
    static let shared = SidecarManager()

    private var process: Process?
    private(set) var isRunning = false

    private init() {}

    private func log(_ msg: String) {
        let line = "[cv sidecar] \(msg)"
        fputs("\(line)\n", stderr)
        if let data = (line + "\n").data(using: .utf8) {
            LogFile.shared.append(data)
        }
    }

    private func findCVDir() -> String? {
        let venvSuffix = "/.venv/bin/python3"
        let knownFallback = "/Users/bruceli/Documents/hackathon/cv"

        var roots: [String] = [knownFallback]

        if let exe = Bundle.main.executableURL {
            roots.append(exe.path)
        }
        roots.append(Bundle.main.bundleURL.path)
        roots.append(FileManager.default.currentDirectoryPath)

        for root in roots {
            var url = URL(fileURLWithPath: root)
            for depth in 0..<8 {
                let resolved = (url.path as NSString).standardizingPath
                let venv = "\(resolved)\(venvSuffix)"
                if FileManager.default.isExecutableFile(atPath: venv) {
                    log("found at \(resolved) (depth=\(depth), root=\(root))")
                    return resolved
                }
                let parent = url.deletingLastPathComponent()
                if parent.path == url.path { break }
                url = parent
            }
        }

        log("not found. exe=\(Bundle.main.executableURL?.path ?? "nil"), "
            + "bundle=\(Bundle.main.bundleURL.path), cwd=\(FileManager.default.currentDirectoryPath)")
        return nil
    }

    func start() {
        guard !isRunning else {
            log("already running, skip")
            return
        }
        guard let dir = findCVDir() else {
            log("skipped — cv/ with .venv not found")
            return
        }
        let python = "\(dir)/.venv/bin/python3"

        // Kill any stale sidecar holding port 8765 so the freshly-built code is used
        // (otherwise an old process keeps answering /health but lacks new endpoints).
        killStale()

        let task = Process()
        task.executableURL = URL(fileURLWithPath: python)
        task.arguments = ["-m", "chromalog_cv.server"]
        task.currentDirectoryURL = URL(fileURLWithPath: dir)
        task.environment = [
            "PYTHONUNBUFFERED": "1",
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:\(dir)/.venv/bin",
        ]

        let errPipe = Pipe()
        task.standardError = errPipe
        task.standardOutput = FileHandle.nullDevice

        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.count > 0, let msg = String(data: data, encoding: .utf8) {
                let trimmed = msg.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    self?.log(trimmed)
                }
            }
        }

        task.terminationHandler = { [weak self] _ in
            self?.isRunning = false
            self?.process = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
        }

        do {
            try task.run()
            process = task
            isRunning = true
            log("started from \(dir)")
        } catch {
            log("failed to start: \(error)")
        }
    }

    /// Terminate any previously-running sidecar (e.g. left over from a prior run)
    /// so the new one can bind port 8765 and serve the current code.
    private func killStale() {
        let kill = Process()
        kill.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        kill.arguments = ["-f", "chromalog_cv.server"]
        try? kill.run()
        kill.waitUntilExit()
        Thread.sleep(forTimeInterval: 0.4)   // let the port free up
    }

    func stop() {
        guard let task = process, task.isRunning else { return }
        task.terminationHandler = nil
        task.terminate()
        task.waitUntilExit()
        process = nil
        isRunning = false
        log("stopped")
    }
}

private final class LogFile {
    static let shared = LogFile()

    private let handle: FileHandle?

    private init() {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("chromalog_sidecar.log").path
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        if let h = FileHandle(forWritingAtPath: path) {
            h.seekToEndOfFile()
            handle = h
        } else {
            handle = nil
        }
    }

    func append(_ data: Data) {
        handle?.write(data)
    }
}