import Foundation

public enum TensorDType: String, Sendable, CaseIterable {
    case float32, float16, bfloat16, int32, int64, uint8
    public var byteWidth: Int { switch self { case .float32, .int32: 4; case .float16, .bfloat16: 2; case .int64: 8; case .uint8: 1 } }
}

public struct TensorFrame: Sendable {
    public let name: String
    public let dtype: TensorDType
    public let shape: [Int64]
    public let payload: Data
    public static let defaultMaxPayloadBytes = 256 * 1024 * 1024

    public init(name: String, dtype: TensorDType, shape: [Int64], payload: Data, maxPayloadBytes: Int = defaultMaxPayloadBytes) throws {
        try Self.validate(name: name, dtype: dtype, shape: shape, byteCount: payload.count, limit: maxPayloadBytes)
        self.name = name; self.dtype = dtype; self.shape = shape; self.payload = payload
    }

    private static func validate(name: String, dtype: TensorDType, shape: [Int64], byteCount: Int, limit: Int) throws {
        guard !name.isEmpty, name.utf8.count <= 128,
            name.utf8.allSatisfy({ (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0) || [45,46,95].contains($0) }),
            shape.count <= 8, shape.allSatisfy({ $0 >= 0 && $0 <= 2_147_483_647 }), byteCount >= 0, limit >= 0 else { throw ProtocolError.invalidTensor }
        guard byteCount <= limit else { throw ProtocolError.payloadTooLarge }
        var elements: Int64 = 1
        if shape.contains(0) { elements = 0 } else {
            for size in shape {
                let (next, overflow) = elements.multipliedReportingOverflow(by: size)
                guard !overflow, next <= Int64(limit / dtype.byteWidth) else { throw ProtocolError.payloadTooLarge }
                elements = next
            }
        }
        guard elements <= Int64(limit / dtype.byteWidth), elements * Int64(dtype.byteWidth) == Int64(byteCount) else { throw ProtocolError.invalidTensor }
    }

    public func encoded() throws -> Data {
        let h: JSONValue = .object([
            JSONEntry("version", .integer(1)), JSONEntry("name", .string(name)), JSONEntry("dtype", .string(dtype.rawValue)),
            JSONEntry("shape", .array(shape.map(JSONValue.integer))), JSONEntry("byte_order", .string("little")),
            JSONEntry("layout", .string("contiguous")), JSONEntry("payload_bytes", .integer(Int64(payload.count))),
            JSONEntry("sha256", .string(CanonicalJSON.sha256(payload)))
        ])
        let header = try CanonicalJSON.encode(h)
        let count = UInt32(header.count)
        return Data([UInt8((count >> 24) & 255), UInt8((count >> 16) & 255), UInt8((count >> 8) & 255), UInt8(count & 255)]) + header + payload
    }

    public static func decode(_ data: Data, maxPayloadBytes: Int = defaultMaxPayloadBytes) throws -> TensorFrame {
        guard data.count >= 4 else { throw ProtocolError.invalidTensor }
        let bytes = data.prefix(4)
        let headerLength = bytes.reduce(0) { ($0 << 8) | Int($1) }
        guard headerLength > 0, headerLength <= 65_536, data.count >= 4 + headerLength else { throw ProtocolError.invalidTensor }
        let h = try CanonicalJSON.parse(data.subdata(in: 4..<4+headerLength))
        guard case .object(let entries) = h,
              Set(entries.map(\.key)) == Set(["version", "name", "dtype", "shape", "byte_order", "layout", "payload_bytes", "sha256"]) else { throw ProtocolError.invalidTensor }
        guard h["version"]?.integer == 1, let name = h["name"]?.string,
              let dtypeText = h["dtype"]?.string, let dtype = TensorDType(rawValue: dtypeText),
              h["byte_order"]?.string == "little", h["layout"]?.string == "contiguous",
              let dims = h["shape"]?.array, dims.allSatisfy({ $0.integer != nil }),
              let count = h["payload_bytes"]?.integer, count >= 0, count <= Int64(Int.max),
              let hash = h["sha256"]?.string, hash.utf8.count == 64, hash.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { throw ProtocolError.invalidTensor }
        let shape = dims.compactMap(\.integer)
        try validate(name: name, dtype: dtype, shape: shape, byteCount: Int(count), limit: maxPayloadBytes)
        guard data.count - 4 - headerLength == Int(count) else { throw ProtocolError.invalidTensor }
        let payload = data.subdata(in: 4+headerLength..<data.count)
        guard CanonicalJSON.sha256(payload) == hash else { throw ProtocolError.hashMismatch }
        return try TensorFrame(name: name, dtype: dtype, shape: shape, payload: payload, maxPayloadBytes: maxPayloadBytes)
    }

    /// Explicit model-input conversion. The codec itself preserves arbitrary bit patterns.
    public func finiteFloat32Values() throws -> [Float] {
        let b = Array(payload); var values: [Float] = []; values.reserveCapacity(b.count / dtype.byteWidth)
        for i in stride(from: 0, to: b.count, by: dtype.byteWidth) {
            let value: Float
            switch dtype {
            case .float32:
                let bits = UInt32(b[i]) | (UInt32(b[i+1]) << 8) | (UInt32(b[i+2]) << 16) | (UInt32(b[i+3]) << 24)
                value = Float(bitPattern: bits)
            case .float16: value = Float(Float16(bitPattern: UInt16(b[i]) | (UInt16(b[i+1]) << 8)))
            case .bfloat16: value = Float(bitPattern: (UInt32(b[i]) | (UInt32(b[i+1]) << 8)) << 16)
            default: throw ProtocolError.unsupportedDType
            }
            guard value.isFinite else { throw ProtocolError.nonFinite }; values.append(value)
        }
        return values
    }
}
