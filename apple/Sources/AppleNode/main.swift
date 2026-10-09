import Foundation
import HeteroMesh

@main struct AppleNode {
    static func main() async {
        do {
            let args = CommandLine.arguments
            guard args.count >= 4, let url = URL(string:args[2]) else {
                print("Usage: apple-node probe|once|run https://coordinator:port certificate-sha256; pairing token is read from HETEROMESH_PAIRING_TOKEN")
                exit(2)
            }
            let client = try PinnedHTTPClient(baseURL:url,pin:CertificatePin(args[3]))
            if args[1] == "probe" {
                _ = try await client.json("GET",path:"/v1/health",token:ProcessInfo.processInfo.environment["HETEROMESH_TEST_TOKEN"])
                print("TLS pin and protocol verified"); return
            }
            guard ["once","run"].contains(args[1]), let pairing = ProcessInfo.processInfo.environment["HETEROMESH_PAIRING_TOKEN"] else { throw TransportError.invalidResponse }
            let worker = try AppleWorker(client:client)
            try await worker.register(pairingToken:pairing)
            if args[1] == "once" { print(try await worker.runOne() ? "Computed and submitted fragment" : "No task available") }
            else { try await worker.run() }
        } catch {
            if let model = error as? ModelError { FileHandle.standardError.write(Data("Model: \(model)\n".utf8)) }
            if let transport = error as? TransportError { FileHandle.standardError.write(Data("Transport: \(transport)\n".utf8)) }
            if let proto = error as? ProtocolError { FileHandle.standardError.write(Data("Protocol: \(proto)\n".utf8)) }
            FileHandle.standardError.write(Data("Node failed: \(type(of:error)) code=\((error as NSError).code)\n".utf8)); exit(1)
        }
    }
}
