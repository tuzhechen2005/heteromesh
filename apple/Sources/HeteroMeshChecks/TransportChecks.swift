import Foundation
import HeteroMesh

public func runTransportChecks() throws {
    let der = Data([1,2,3,4]), fingerprint = CanonicalJSON.sha256(Data([1,2,3,4]))
    let pin = try CertificatePin(fingerprint)
    guard pin.matches(der: der), !pin.matches(der: Data([1,2,3,5])) else { throw ModelError.invalidInput("certificate pin") }
    for invalid in ["", "abc", String(repeating:"g",count:64)] {
        do { _ = try CertificatePin(invalid) } catch { continue }
        throw ModelError.invalidInput("invalid pin accepted")
    }
    do { _ = try PinnedHTTPClient(baseURL:URL(string:"http://localhost:1234")!,pin:pin) } catch { return }
    throw ModelError.invalidInput("plaintext accepted")
}

public func runWorkerContractChecks() throws {
    let output = try TensorFrame.float32(name:"hidden",shape:[1,2],values:[0,1])
    let valid: JSONValue = .object([JSONEntry("dtype",.string("float32")),JSONEntry("shape",.array([.integer(1),.integer(2)]))])
    try WorkerContract.validateOutput(output,spec:valid)
    let invalid: JSONValue = .object([JSONEntry("dtype",.string("float32")),JSONEntry("shape",.array([.integer(1),.string("bad"),.integer(2)]))])
    do { try WorkerContract.validateOutput(output,spec:invalid) } catch {
        guard !ParticipationPolicy.shouldStop(.active), !ParticipationPolicy.shouldStop(.inactive), ParticipationPolicy.shouldStop(.background) else { throw ModelError.invalidInput("foreground lifecycle policy") }
        return
    }
    throw ModelError.invalidInput("malformed output shape accepted")
}
