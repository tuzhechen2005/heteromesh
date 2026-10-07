import Foundation
import HeteroMeshChecks

do {
    try runProtocolChecks(fixtures: CommandLine.arguments.dropFirst().first)
    try runTransformerChecks()
    if CommandLine.arguments.count > 2 { try runTransformerGoldens(CommandLine.arguments[2]) }
    try runTransportChecks()
    try runWorkerContractChecks()
    print("PASS: Swift protocol conformance")
} catch {
    FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8))
    exit(1)
}
