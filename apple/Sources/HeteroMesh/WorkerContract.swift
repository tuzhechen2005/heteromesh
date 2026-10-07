import Foundation

package enum WorkerContract {
    package static func validateOutput(_ output: TensorFrame, spec: JSONValue) throws {
        guard spec["dtype"]?.string == "float32", let shape = spec["shape"]?.array, shape.allSatisfy({ $0.integer != nil }), shape.compactMap(\.integer) == output.shape else { throw ModelError.invalidInput("output contract") }
    }
}

public enum NodePhase: Sendable { case active, inactive, background }
public enum ParticipationPolicy {
    public static func shouldStop(_ phase: NodePhase) -> Bool { phase == .background }
}
