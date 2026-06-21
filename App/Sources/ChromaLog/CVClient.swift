import Foundation
import AppKit

struct CVDetectResponse: Decodable {
    let width: Int
    let height: Int
    let rectified: Bool
    let rectify_confidence: Double
    let polarity: String
    let polarity_uncertain: Bool
    let minority_frac: Double
    let baseline_y_norm: Double?
    let front_y_norm: Double?
    let baseline_from: String
    let front_from: String
    let n_lanes: Int
    let spots: [CVSpot]
    let warnings: [String]
    let engine_used: String?    // opencv | ai+opencv | yolo
    let image_b64: String?      // 正畸后的图 (坐标基准): app 显示它而非原图
    let debug_png_b64: String?
}

struct CVSpot: Decodable {
    let x: Double
    let y: Double
    let bbox_norm: [Double]
    let area_px: Int
    let lane: Int
    let shape: String
    let rf: Double?
}

struct CVRectifyResponse: Decodable {
    let width: Int
    let height: Int
    let rectified: Bool
    let rectify_confidence: Double
    let note: String
    let image_b64: String?
}

struct CVModelInfo: Decodable {
    let trained: Bool
    let n_samples: Int
    let updated_at: String?
}

struct YoloModelStatus: Decodable {
    let status: String      // "not_trained" | "training" | "ready"
    let trained_at: String?
}

enum CVClientError: LocalizedError {
    case notRunning
    case invalidResponse
    case serverError(String)

    var errorDescription: String? {
        switch self {
        case .notRunning:
            return "OpenCV sidecar is not running. Start it with:\npython -m chromalog_cv.server"
        case .invalidResponse:
            return "Invalid response from CV sidecar"
        case .serverError(let msg):
            return "CV sidecar error: \(msg)"
        }
    }
}

final class CVClient {
    static let shared = CVClient()

    private let baseURL = "http://127.0.0.1:8765"
    private let session: URLSession

    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        session = URLSession(configuration: config)
    }

    func checkHealth() async -> Bool {
        guard let url = URL(string: "\(baseURL)/health") else { return false }
        do {
            let (_, response) = try await session.data(from: url)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    func waitForReady(timeout: TimeInterval = 10) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if await checkHealth() { return true }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        return false
    }

    func detect(imageData: Data, debug: Bool = false,
                hatThreshK: Double? = nil, kneeDeviation: Double? = nil,
                useAI: Bool = false, orModel: String? = nil, orKey: String? = nil,
                useYolo: Bool = false) async throws -> CVDetectResponse {
        var urlComponents = URLComponents(string: "\(baseURL)/detect")!
        var items: [URLQueryItem] = []
        if debug { items.append(URLQueryItem(name: "debug", value: "true")) }
        if let k = hatThreshK { items.append(URLQueryItem(name: "hat_thresh_k", value: String(k))) }
        if let d = kneeDeviation { items.append(URLQueryItem(name: "knee_deviation", value: String(d))) }
        if useAI, let m = orModel, orKey != nil {
            items.append(URLQueryItem(name: "use_ai", value: "true"))
            items.append(URLQueryItem(name: "or_model", value: m))
        }
        if useYolo {
            items.append(URLQueryItem(name: "use_yolo", value: "true"))
        }
        if !items.isEmpty { urlComponents.queryItems = items }
        guard let url = urlComponents.url else {
            throw CVClientError.invalidResponse
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        // AI path makes (slow) OpenRouter calls server-side; give it a long timeout.
        // OpenCV-only stays snappy.
        request.timeoutInterval = useAI ? 150 : 30

        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        if useAI, let key = orKey, orModel != nil {
            request.setValue(key, forHTTPHeaderField: "X-OpenRouter-Key")
        }

        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"plate.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw CVClientError.notRunning
        }

        if httpResponse.statusCode == 400, let errorPayload = try? JSONDecoder().decode([String: String].self, from: data),
           let msg = errorPayload["error"] {
            throw CVClientError.serverError(msg)
        }

        guard httpResponse.statusCode == 200 else {
            throw CVClientError.notRunning
        }

        let decoder = JSONDecoder()
        return try decoder.decode(CVDetectResponse.self, from: data)
    }

    /// 只做正畸 (导入时调用), 返回正畸后的图。
    func rectify(imageData: Data) async throws -> CVRectifyResponse {
        guard let url = URL(string: "\(baseURL)/rectify") else { throw CVClientError.invalidResponse }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"plate.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw CVClientError.notRunning }
        guard http.statusCode == 200 else { throw CVClientError.notRunning }
        return try JSONDecoder().decode(CVRectifyResponse.self, from: data)
    }

    func modelInfo() async -> CVModelInfo? {
        guard let url = URL(string: "\(baseURL)/model") else { return nil }
        do {
            let (data, resp) = try await session.data(from: url)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else { return nil }
            return try JSONDecoder().decode(CVModelInfo.self, from: data)
        } catch { return nil }
    }

    /// Best-effort online learning from one manual correction. Errors are ignored.
    func learn(rectified: Data, finalSpots: [CGPoint], autoCandidates: [CGPoint],
               baselineY: Double? = nil, frontY: Double? = nil) async {
        guard let url = URL(string: "\(baseURL)/learn") else { return }
        var payload: [String: Any] = [
            "final_spots": finalSpots.map { [$0.x, $0.y] },
            "auto_candidates": autoCandidates.map { [$0.x, $0.y] },
        ]
        if let b = baselineY { payload["baseline_y"] = b }
        if let f = frontY { payload["front_y"] = f }
        guard let payloadData = try? JSONSerialization.data(withJSONObject: payload) else { return }
        let payloadStr = String(data: payloadData, encoding: .utf8) ?? "{}"

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        // image part
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"rect.png\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
        body.append(rectified)
        body.append("\r\n".data(using: .utf8)!)
        // payload field
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"payload\"\r\n\r\n".data(using: .utf8)!)
        body.append(payloadStr.data(using: .utf8)!)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body
        _ = try? await session.data(for: request)
    }
}

extension CVClient {
    /// Trigger background YOLO training on the sidecar.
    func trainYolo() async throws {
        guard let url = URL(string: "\(baseURL)/train-yolo") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 10
        let (_, resp) = try await session.data(for: req)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
            throw CVClientError.notRunning
        }
    }

    /// Poll the sidecar for YOLO model status.
    func yoloModelStatus() async throws -> YoloModelStatus {
        guard let url = URL(string: "\(baseURL)/yolo-model") else {
            throw CVClientError.invalidResponse
        }
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode(YoloModelStatus.self, from: data)
    }
}

struct CVReportResponse: Decodable {
    let ok: Bool
    let questions: [String]?
    let markdown: String?
    let error: String?
}

extension CVClient {
    /// AI report (spec §10). mode = "questions" | "report".
    func report(mode: String, payloadJSON: String, model: String, key: String) async throws -> CVReportResponse {
        var comps = URLComponents(string: "\(baseURL)/report")!
        comps.queryItems = [URLQueryItem(name: "mode", value: mode),
                            URLQueryItem(name: "model", value: model)]
        guard let url = comps.url else { throw CVClientError.invalidResponse }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue(key, forHTTPHeaderField: "X-OpenRouter-Key")
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"payload\"\r\n\r\n".data(using: .utf8)!)
        body.append(payloadJSON.data(using: .utf8)!)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body
        let (data, resp) = try await session.data(for: request)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if code == 404 {
            throw CVClientError.serverError("Sidecar is outdated (no /report). Quit and relaunch the app.")
        }
        guard code == 200 else { throw CVClientError.notRunning }
        return try JSONDecoder().decode(CVReportResponse.self, from: data)
    }
}

extension NSImage {
    func jpegData(compressionQuality: CGFloat = 0.92) -> Data? {
        guard let tiff = tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff) else { return nil }
        return bitmap.representation(using: .jpeg, properties: [.compressionFactor: compressionQuality])
    }
}