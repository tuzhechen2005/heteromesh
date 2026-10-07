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
