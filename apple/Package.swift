// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HeteroMesh",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [.library(name: "HeteroMesh", targets: ["HeteroMesh"]), .executable(name: "protocol-checks", targets: ["ProtocolChecks"])],
    targets: [
        .target(name: "HeteroMesh"),
        .target(name: "HeteroMeshChecks", dependencies: ["HeteroMesh"]),
        .executableTarget(name: "ProtocolChecks", dependencies: ["HeteroMeshChecks"]),
        .testTarget(name: "HeteroMeshTests", dependencies: ["HeteroMeshChecks"])
    ]
)
