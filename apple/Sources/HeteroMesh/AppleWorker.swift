import Foundation

public actor AppleWorker {
    public let client: PinnedHTTPClient
    private var token: String?
    private var capabilities: JSONValue?
    private let budgetBytes: Int64
    public init(client: PinnedHTTPClient, budgetBytes: Int64 = 64 * 1024 * 1024) throws {
        guard budgetBytes >= 8*1024*1024, budgetBytes <= 512*1024*1024 else { throw ModelError.invalidInput("budget") }
        self.client = client; self.budgetBytes = budgetBytes
    }
    public func register(pairingToken: String) async throws {
        #if os(iOS)
        let platform = "ios"
        #else
        let platform = "macos"
        #endif
        let cap: JSONValue = .object([
            JSONEntry("protocol_version",.integer(1)),JSONEntry("platform",.string(platform)),
            JSONEntry("runtime",.string("swift6")),JSONEntry("backend",.string("swift_cpu")),
            JSONEntry("supported_ops",.array([.string("tiny_transformer_block_v1")])),
            JSONEntry("wire_dtypes",.array(TensorDType.allCases.map { .string($0.rawValue) })),
            JSONEntry("compute_dtypes",.array([.string("float32")])),
            JSONEntry("memory",.object([JSONEntry("unified",.bool(true)),JSONEntry("host_budget_bytes",.integer(budgetBytes)),JSONEntry("accelerator_budget_bytes",.null)]))
        ])
        let reply = try await client.json("POST",path:"/v1/nodes/register",token:pairingToken,value:.object([JSONEntry("capabilities",cap)]))
        guard let token = reply["token"]?.string, !token.isEmpty else { throw TransportError.invalidResponse }
        self.token = token; self.capabilities = cap
    }
    public func heartbeat(state: String = "foreground") async throws {
        guard let token else { throw TransportError.invalidResponse }
        _ = try await client.json("POST",path:"/v1/heartbeat",token:token,value:.object([JSONEntry("state",.string(state)),JSONEntry("capabilities",capabilities ?? .object([]))]))
    }
    public func runOne() async throws -> Bool {
        guard let token else { throw TransportError.invalidResponse }
        let reply = try await client.json("POST",path:"/v1/work/lease",token:token)
        guard let task = reply["task"] else { throw TransportError.invalidResponse }
        if task == .null { return false }
        do { try await execute(task:task,token:token) }
        catch {
            let identity = try? identityEntries(task)
            if let identity {
                _ = try? await client.json("POST",path:"/v1/work/error",token:token,value:.object(identity + [JSONEntry("code",.string(error is CancellationError ? "CANCELLED" : "EXECUTION_FAILED")),JSONEntry("transient",.bool(false))]))
            }
            throw error
        }
        return true
    }
    public func run() async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                while true {
                    try Task.checkCancellation()
                    try await self.heartbeat()
                    try await Task.sleep(for:.seconds(5))
                }
            }
            group.addTask {
                while true {
                    try Task.checkCancellation()
                    _ = try await self.runOne()
                    try await Task.sleep(for:.milliseconds(500))
                }
            }
            defer { group.cancelAll() }
            try await group.next()
        }
    }
    private func identityEntries(_ task: JSONValue) throws -> [JSONEntry] {
        let names = ["job_id","recovery_epoch","task_id","step_index","fragment_id","attempt_id","manifest_digest","input_digest"]
        return try names.map { name in
            guard let value = task[name] else { throw TransportError.invalidResponse }
            if name == "recovery_epoch" || name == "step_index" { guard let i = value.integer, i >= 0 else { throw TransportError.invalidResponse } }
            else { guard let s = value.string, !s.isEmpty else { throw TransportError.invalidResponse } }
            return JSONEntry(name,value)
        }
    }
    private func execute(task: JSONValue, token: String) async throws {
        let identity = try identityEntries(task)
        guard task["operation"]?.string == "tiny_transformer_block_v1",
              case .object(let refs) = task["inputs"], Set(refs.map(\.key)) == Set(TinyTransformer.inputNames),
              case .object(let parameters) = task["parameters"], parameters.allSatisfy({ $0.key == "epsilon" }),
              (task["parameters"]?["epsilon"] == nil || task["parameters"]?["epsilon"]?.string != nil),
              let epsilon = Float(task["parameters"]?["epsilon"]?.string ?? "0.00001"), epsilon >= 1e-8, epsilon <= 0.01, epsilon.isFinite else { throw ModelError.unsupportedOperation }
        let headers = ["X-Job-Id":task["job_id"]!.string!,"X-Recovery-Epoch":String(task["recovery_epoch"]!.integer!),"X-Attempt-Id":task["attempt_id"]!.string!]
        var inputs: [String:TensorFrame] = [:]; var total: Int64 = 0
        // G1 uses small measured profiles; bound each artifact and total resident inputs.
        let maxArtifactBytes = min(8*1024*1024,Int(budgetBytes/8))
        for entry in refs {
            try Task.checkCancellation()
            guard let digest = entry.value.string, digest.utf8.count == 64,
                  digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { throw TransportError.invalidResponse }
            let data = try await client.request("GET",path:"/v1/artifacts/"+digest,token:token,headers:headers,maxResponseBytes:maxArtifactBytes)
            guard CanonicalJSON.sha256(data) == digest else { throw ProtocolError.hashMismatch }
            total += Int64(data.count)
            guard total <= budgetBytes/4 else { throw ProtocolError.payloadTooLarge }
            inputs[entry.key] = try TensorFrame.decode(data,maxPayloadBytes:maxArtifactBytes)
        }
        let output = try await Task.detached(priority:.userInitiated) { try TinyTransformer.execute(inputs:inputs,epsilon:epsilon) }.value
        try Task.checkCancellation()
        guard case .object(let outputs) = task["outputs"], outputs.count == 1, outputs[0].key == "hidden" else { throw ModelError.invalidInput("output contract") }
        try WorkerContract.validateOutput(output,spec:outputs[0].value)
        let data = try output.encoded(), digest = CanonicalJSON.sha256(data)
        var uploadHeaders = headers; uploadHeaders["Content-Type"] = "application/octet-stream"; uploadHeaders["X-Output-Name"] = "hidden"
        _ = try await client.request("PUT",path:"/v1/artifacts/"+digest,token:token,body:data,headers:uploadHeaders)
        try Task.checkCancellation()
        _ = try await client.json("POST",path:"/v1/work/result",token:token,value:.object(identity + [JSONEntry("outputs",.object([JSONEntry("hidden",.string(digest))]))]))
    }
}
