import Foundation
import HeteroMesh

private func canonicalOrderingAndEscapes() throws {
    let input = Data("{\"中\":true,\"a\":\"line\\n\\u0001/\",\"z\":-9223372036854775808}".utf8)
    try expect(try CanonicalJSON.encode(CanonicalJSON.parse(input)) == Data("{\"a\":\"line\\n\\u0001/\",\"z\":-9223372036854775808,\"中\":true}".utf8))
}

private func rejectsAmbiguousJSON() throws {
    for invalid in ["{\"a\":1,\"a\":2}", "{\"a\":1,\"\\u0061\":2}", "1.0", "1e0", "9223372036854775808", "[NaN]", "\"\\ud800\"", "[1,]", "01", "true false"] {
        try rejects { _ = try CanonicalJSON.parse(Data(invalid.utf8)) }
    }
}

private func roundTripTensorAndIntegrity() throws {
    let payload = Data([0,0,128,63,0,0,0,192])
    let frame = try TensorFrame(name: "hidden", dtype: .float32, shape: [2], payload: payload)
    let encoded = try frame.encoded()
    let decoded = try TensorFrame.decode(encoded)
    try expect(decoded.payload == payload)
    try expect(decoded.shape == [2])
    var damaged = encoded; damaged[damaged.count - 1] ^= 1
    try rejects { _ = try TensorFrame.decode(damaged) }
    try rejects { _ = try TensorFrame.decode(encoded + Data([0])) }
    try rejects { _ = try TensorFrame.decode(encoded, maxPayloadBytes: 4) }
}

private func tensorRejectsInvalidDimensionsBeforeAllocation() throws {
    try rejects { _ = try TensorFrame(name: "x", dtype: .uint8, shape: [0, 2_147_483_648], payload: Data()) }
    try rejects { _ = try TensorFrame(name: "x", dtype: .uint8, shape: [2_147_483_647, 2_147_483_647, 8], payload: Data()) }
    try rejects { _ = try TensorFrame(name: "../x", dtype: .uint8, shape: [0], payload: Data()) }
    try rejects { _ = try TensorFrame(name: "x", dtype: .float32, shape: [], payload: Data()) }
}

private func bf16PreservesBitsAndFinitePolicy() throws {
    let bits: [UInt8] = [0,0,0,128,128,63,0,192,1,0,127,127,128,127,193,127]
    let f = try TensorFrame(name: "bf16", dtype: .bfloat16, shape: [8], payload: Data(bits))
    try expect(try TensorFrame.decode(f.encoded()).payload == Data(bits))
    try rejects { _ = try f.finiteFloat32Values() }
    let finite = try TensorFrame(name: "bf16", dtype: .bfloat16, shape: [1], payload: Data([128,63]))
    try expect(try finite.finiteFloat32Values() == [1])
}

private struct Failure: Error { let message: String }
private func expect(_ value: @autoclosure () throws -> Bool) throws { if try !value() { throw Failure(message: "expectation failed") } }
private func rejects(_ operation: () throws -> Void) throws { do { try operation() } catch { return }; throw Failure(message: "invalid input accepted") }
public func runProtocolChecks(fixtures: String? = nil) throws {
    try canonicalOrderingAndEscapes()
    try rejectsAmbiguousJSON()
    try roundTripTensorAndIntegrity()
    try rejectsUnknownTensorHeaderField()
    try tensorRejectsInvalidDimensionsBeforeAllocation()
    try bf16PreservesBitsAndFinitePolicy()
    if let fixtures { try sharedFixtures(URL(fileURLWithPath: fixtures)) }
}
private func sharedFixtures(_ root: URL) throws {
    let entries = try CanonicalJSON.parse(Data(contentsOf: root.appendingPathComponent("canonical.json"))).array!
    for entry in entries {
        let data = try CanonicalJSON.encode(entry["input"]!)
        try expect(data.map { String(format: "%02x", $0) }.joined() == entry["expected_hex"]?.string)
        try expect(CanonicalJSON.sha256(data) == entry["sha256"]?.string)
    }
    let index = try CanonicalJSON.parse(Data(contentsOf: root.appendingPathComponent("tensors/index.json"))).array!
    for entry in index {
        let file = try Data(contentsOf: root.appendingPathComponent("tensors/" + entry["file"]!.string!))
        let tensor = try TensorFrame.decode(file)
        try expect(tensor.payload.map { String(format: "%02x", $0) }.joined() == entry["payload_hex"]?.string)
        try expect(CanonicalJSON.sha256(file) == entry["artifact_sha256"]?.string)
        try expect(TensorFrame.decode(tensor.encoded()).payload == tensor.payload)
    }
}

private func rejectsUnknownTensorHeaderField() throws {
    let frame = try TensorFrame(name: "x", dtype: .uint8, shape: [0], payload: Data())
    let encoded = try frame.encoded()
    let header = try CanonicalJSON.parse(Data(encoded.dropFirst(4)))
    guard case .object(let entries) = header else { throw Failure(message: "header") }
    let modified = try CanonicalJSON.encode(.object(entries + [JSONEntry("execute", .string("arbitrary"))]))
    let n = UInt32(modified.count)
    let wire = Data([UInt8((n >> 24) & 255), UInt8((n >> 16) & 255), UInt8((n >> 8) & 255), UInt8(n & 255)]) + modified
    try rejects { _ = try TensorFrame.decode(wire) }
}
