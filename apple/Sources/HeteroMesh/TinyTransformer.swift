import Foundation

public enum ModelError: Error, Sendable { case invalidInput(String), unsupportedOperation, computationFailed }

extension TensorFrame {
    public static func float32(name: String, shape: [Int64], values: [Float]) throws -> TensorFrame {
        guard values.allSatisfy(\.isFinite) else { throw ProtocolError.nonFinite }
        var data = Data(); data.reserveCapacity(values.count * 4)
        for v in values {
            let b = v.bitPattern
            data.append(contentsOf: [UInt8(b & 255), UInt8((b >> 8) & 255), UInt8((b >> 16) & 255), UInt8((b >> 24) & 255)])
        }
        return try TensorFrame(name: name, dtype: .float32, shape: shape, payload: data)
    }
}

/// Real reference computation for the shared tiny_transformer_block_v1 operation.
public enum TinyTransformer {
    public static let inputNames = ["hidden","condition","wq","wk","wv","wo","w1","b1","w2","b2","ln1_weight","ln1_bias","ln2_weight","ln2_bias"]
    public static func execute(inputs: [String: TensorFrame], epsilon: Float = 0.00001) throws -> TensorFrame {
        guard Set(inputs.keys) == Set(inputNames), epsilon.isFinite, epsilon >= 1e-8, epsilon <= 0.01,
              let hidden = inputs["hidden"], hidden.shape.count == 2,
              let w1 = inputs["w1"], w1.shape.count == 2 else { throw ModelError.invalidInput("inputs") }
        let n = Int(hidden.shape[0]), d = Int(hidden.shape[1]), f = Int(w1.shape[1])
        // The reference CPU profile is intentionally bounded. Larger profiles require measured budgets.
        guard n > 0, d > 0, f > 0, n <= 512, d <= 256, f <= 1024 else { throw ModelError.invalidInput("unsupported dimensions") }
        let expected: [String: [Int64]] = [
            "hidden":[Int64(n),Int64(d)], "condition":[Int64(n),Int64(d)],
            "wq":[Int64(d),Int64(d)], "wk":[Int64(d),Int64(d)], "wv":[Int64(d),Int64(d)], "wo":[Int64(d),Int64(d)],
            "w1":[Int64(d),Int64(f)], "b1":[Int64(f)], "w2":[Int64(f),Int64(d)], "b2":[Int64(d)],
            "ln1_weight":[Int64(d)], "ln1_bias":[Int64(d)], "ln2_weight":[Int64(d)], "ln2_bias":[Int64(d)]
        ]
        var x: [String:[Float]] = [:]
        for name in inputNames {
            guard let t = inputs[name], t.name == name, t.dtype == .float32, t.shape == expected[name] else { throw ModelError.invalidInput(name) }
            x[name] = try t.finiteFloat32Values()
        }
        func mm(_ a: [Float], _ b: [Float], _ rows: Int, _ inner: Int, _ cols: Int) -> [Float] {
            var out = [Float](repeating: 0, count: rows*cols)
            for r in 0..<rows { for k in 0..<inner { for c in 0..<cols { out[r*cols+c] += a[r*inner+k]*b[k*cols+c] } } }
            return out
        }
        func ln(_ a: [Float], _ weight: [Float], _ bias: [Float]) -> [Float] {
            var out = a
            for r in 0..<n {
                var mean: Float = 0; for c in 0..<d { mean += a[r*d+c] }; mean /= Float(d)
                var variance: Float = 0; for c in 0..<d { let delta = a[r*d+c]-mean; variance += delta*delta }; variance /= Float(d)
                let inv = 1 / sqrt(variance + epsilon)
                for c in 0..<d { out[r*d+c] = (a[r*d+c]-mean)*inv*weight[c]+bias[c] }
            }
            return out
        }
        let h0 = x["hidden"]!, condition = x["condition"]!
        let z = ln(zip(h0, condition).map(+), x["ln1_weight"]!, x["ln1_bias"]!)
        let q = mm(z,x["wq"]!,n,d,d), k = mm(z,x["wk"]!,n,d,d), v = mm(z,x["wv"]!,n,d,d)
        var scores = [Float](repeating: 0, count:n*n)
        for r in 0..<n { for c in 0..<n { for j in 0..<d { scores[r*n+c] += q[r*d+j]*k[c*d+j] }; scores[r*n+c] /= sqrt(Float(d)) } }
        for r in 0..<n {
            let maxValue = scores[r*n..<r*n+n].max()!
            var total: Float = 0
            for c in 0..<n { scores[r*n+c] = exp(scores[r*n+c]-maxValue); total += scores[r*n+c] }
            for c in 0..<n { scores[r*n+c] /= total }
        }
        let att = mm(mm(scores,v,n,n,d),x["wo"]!,n,d,d)
        let h = zip(h0,att).map(+)
        let z2 = ln(h,x["ln2_weight"]!,x["ln2_bias"]!)
        var ff = mm(z2,x["w1"]!,n,d,f)
        for r in 0..<n { for c in 0..<f { ff[r*f+c] = max(0,ff[r*f+c]+x["b1"]![c]) } }
        let projected = mm(ff,x["w2"]!,n,f,d)
        var output = h
        for r in 0..<n { for c in 0..<d { output[r*d+c] += projected[r*d+c]+x["b2"]![c] } }
        return try TensorFrame.float32(name:"hidden",shape:hidden.shape,values:output)
    }
}
