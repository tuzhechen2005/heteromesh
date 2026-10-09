import Foundation
import HeteroMesh

public func runTransformerChecks() throws {
    func tensor(_ name: String, _ shape: [Int64], _ values: [Float]) throws -> TensorFrame {
        try TensorFrame.float32(name: name, shape: shape, values: values)
    }
    var inputs: [String: TensorFrame] = [:]
    inputs["hidden"] = try tensor("hidden", [1,2], [1,3])
    inputs["condition"] = try tensor("condition", [1,2], [0,0])
    for name in ["wq","wk","wv","wo","w1","w2"] { inputs[name] = try tensor(name, [2,2], [1,0,0,1]) }
    for name in ["b1","b2","ln1_bias","ln2_bias"] { inputs[name] = try tensor(name, [2], [0,0]) }
    for name in ["ln1_weight","ln2_weight"] { inputs[name] = try tensor(name, [2], [1,1]) }
    let output = try TinyTransformer.execute(inputs: inputs, epsilon: 0.00001)
    let values = try output.finiteFloat32Values()
    guard abs(values[0]) < 0.0001, abs(values[1] - 5) < 0.0001 else { throw ModelError.invalidInput("reference output") }
    inputs["condition"] = try tensor("condition", [1,2], [4,0])
    let changed = try TinyTransformer.execute(inputs: inputs, epsilon: 0.00001).finiteFloat32Values()
    guard abs(changed[0] - values[0]) > 1 else { throw ModelError.invalidInput("condition must affect output") }
    inputs.removeValue(forKey: "wk")
    do { _ = try TinyTransformer.execute(inputs: inputs, epsilon: 0.00001) } catch { return }
    throw ModelError.invalidInput("missing weight accepted")
}

public func runTransformerGoldens(_ directory: String) throws {
    let root = URL(fileURLWithPath:directory)
    let index = try CanonicalJSON.parse(Data(contentsOf:root.appendingPathComponent("index.json")))
    guard case .object(let inputRefs) = index["inputs"] else { throw ModelError.invalidInput("fixture inputs") }
    var inputs: [String:TensorFrame] = [:]
    for entry in inputRefs {
        let data = try Data(contentsOf:root.appendingPathComponent(entry.value["file"]!.string!))
        guard CanonicalJSON.sha256(data) == entry.value["sha256"]?.string else { throw ProtocolError.hashMismatch }
        inputs[entry.key] = try TensorFrame.decode(data)
    }
    for step in 0..<2 {
        let output = try TinyTransformer.execute(inputs:inputs)
        let reference = index["outputs"]![String(step)]!
        let data = try Data(contentsOf:root.appendingPathComponent(reference["file"]!.string!))
        guard CanonicalJSON.sha256(data) == reference["sha256"]?.string else { throw ProtocolError.hashMismatch }
        let expected = try TensorFrame.decode(data).finiteFloat32Values(), actual = try output.finiteFloat32Values()
        guard actual.count == expected.count else { throw ModelError.invalidInput("golden shape") }
        for (a,b) in zip(actual,expected) {
            guard abs(a-b) <= 0.00001 + 0.0001*abs(b) else { throw ModelError.invalidInput("golden numeric mismatch") }
        }
        inputs["hidden"] = output
    }
}
