// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HeteroMesh",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [.library(name: "HeteroMesh", targets: ["HeteroMesh"]), .executable(name: "protocol-checks", targets: ["ProtocolChecks"]), .executable(name: "apple-node", targets: ["AppleNode"])],
    targets: [
        .target(name: "HeteroMesh"),
        .executableTarget(name: "AppleNode", dependencies: ["HeteroMesh"]),
        .target(name: "HeteroMeshChecks", dependencies: ["HeteroMesh"]),
        .executableTarget(name: "ProtocolChecks", dependencies: ["HeteroMeshChecks"]),
        .testTarget(name: "HeteroMeshTests", dependencies: ["HeteroMeshChecks"])
    ]
)
