import Foundation
import HeteroMeshChecks

do {
    try runProtocolChecks(fixtures: CommandLine.arguments.dropFirst().first)
    print("PASS: Swift protocol conformance")
} catch {
    FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8))
    exit(1)
}
