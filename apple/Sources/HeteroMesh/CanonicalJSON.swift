import Foundation
import CryptoKit

public enum ProtocolError: Error, Equatable, Sendable {
    case invalidJSON, duplicateKey, integerOverflow, depthLimit
    case invalidTensor, payloadTooLarge, hashMismatch, nonFinite, unsupportedDType
}

public struct JSONEntry: Sendable, Equatable {
    public let key: String
    public let value: JSONValue
    public init(_ key: String, _ value: JSONValue) { self.key = key; self.value = value }
}

public indirect enum JSONValue: Sendable, Equatable {
    case object([JSONEntry]), array([JSONValue]), string(String), integer(Int64), bool(Bool), null
    public subscript(_ key: String) -> JSONValue? {
        guard case .object(let entries) = self else { return nil }
        return entries.first { Array($0.key.utf8) == Array(key.utf8) }?.value
    }
    public var string: String? { if case .string(let s) = self { return s }; return nil }
    public var integer: Int64? { if case .integer(let n) = self { return n }; return nil }
    public var array: [JSONValue]? { if case .array(let a) = self { return a }; return nil }
}

/// D-07 restricted canonical JSON; intentionally not RFC 8785.
public enum CanonicalJSON {
    public static func parse(_ data: Data, maxBytes: Int = 65_536) throws -> JSONValue {
        guard data.count <= maxBytes, String(data: data, encoding: .utf8) != nil else { throw ProtocolError.invalidJSON }
        var parser = Parser(bytes: Array(data))
        let value = try parser.value(depth: 0)
        parser.whitespace()
        guard parser.index == parser.bytes.count else { throw ProtocolError.invalidJSON }
        return value
    }

    public static func encode(_ value: JSONValue) throws -> Data { Data(try render(value, depth: 0).utf8) }
    public static func digest(_ value: JSONValue) throws -> String { sha256(try encode(value)) }
    public static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    private static func render(_ value: JSONValue, depth: Int) throws -> String {
        guard depth <= 64 else { throw ProtocolError.depthLimit }
        switch value {
        case .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .integer(let n): return String(n)
        case .string(let s): return quoted(s)
        case .array(let values): return "[" + (try values.map { try render($0, depth: depth + 1) }).joined(separator: ",") + "]"
        case .object(let entries):
            var seen = Set<Data>()
            for entry in entries { guard seen.insert(Data(entry.key.utf8)).inserted else { throw ProtocolError.duplicateKey } }
            let sorted = entries.sorted { $0.key.unicodeScalars.lexicographicallyPrecedes($1.key.unicodeScalars, by: { $0.value < $1.value }) }
            return "{" + (try sorted.map { quoted($0.key) + ":" + (try render($0.value, depth: depth + 1)) }).joined(separator: ",") + "}"
        }
    }

    private static func quoted(_ string: String) -> String {
        var out = "\""
        for scalar in string.unicodeScalars {
            switch scalar.value {
            case 34: out += "\\\""
            case 92: out += "\\\\"
            case 8: out += "\\b"
            case 9: out += "\\t"
            case 10: out += "\\n"
            case 12: out += "\\f"
            case 13: out += "\\r"
            case 0...31: out += String(format: "\\u%04x", scalar.value)
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out + "\""
    }

    private struct Parser {
        let bytes: [UInt8]
        var index = 0
        mutating func whitespace() { while index < bytes.count && [9,10,13,32].contains(bytes[index]) { index += 1 } }
        mutating func take(_ byte: UInt8) -> Bool {
            if index < bytes.count && bytes[index] == byte { index += 1; return true }; return false
        }
        mutating func value(depth: Int) throws -> JSONValue {
            guard depth <= 64 else { throw ProtocolError.depthLimit }
            whitespace()
            guard index < bytes.count else { throw ProtocolError.invalidJSON }
            switch bytes[index] {
            case 34: return .string(try string())
            case 123:
                index += 1; whitespace(); var entries: [JSONEntry] = []; var seen = Set<Data>()
                if take(125) { return .object(entries) }
                while true {
                    whitespace(); let key = try string()
                    guard seen.insert(Data(key.utf8)).inserted else { throw ProtocolError.duplicateKey }
                    whitespace(); guard take(58) else { throw ProtocolError.invalidJSON }
                    entries.append(JSONEntry(key, try value(depth: depth + 1))); whitespace()
                    if take(125) { break }; guard take(44) else { throw ProtocolError.invalidJSON }
                }
                return .object(entries)
            case 91:
                index += 1; whitespace(); var values: [JSONValue] = []
                if take(93) { return .array(values) }
                while true {
                    values.append(try value(depth: depth + 1)); whitespace()
                    if take(93) { break }; guard take(44) else { throw ProtocolError.invalidJSON }
                }
                return .array(values)
            case 116: try literal("true"); return .bool(true)
            case 102: try literal("false"); return .bool(false)
            case 110: try literal("null"); return .null
            case 45, 48...57:
                let start = index; _ = take(45)
                guard index < bytes.count else { throw ProtocolError.invalidJSON }
                if take(48) {} else {
                    guard (49...57).contains(bytes[index]) else { throw ProtocolError.invalidJSON }
                    while index < bytes.count && (48...57).contains(bytes[index]) { index += 1 }
                }
                guard let integer = Int64(String(decoding: bytes[start..<index], as: UTF8.self)) else { throw ProtocolError.integerOverflow }
                return .integer(integer)
            default: throw ProtocolError.invalidJSON
            }
        }
        mutating func literal(_ text: String) throws {
            let expected = Array(text.utf8)
            guard index + expected.count <= bytes.count, Array(bytes[index..<index+expected.count]) == expected else { throw ProtocolError.invalidJSON }
            index += expected.count
        }
        mutating func string() throws -> String {
            guard take(34) else { throw ProtocolError.invalidJSON }
            let start = index - 1
            while index < bytes.count {
                let b = bytes[index]; index += 1
                if b == 34 {
                    do { return try JSONDecoder().decode(String.self, from: Data(bytes[start..<index])) }
                    catch { throw ProtocolError.invalidJSON }
                }
                if b < 32 { throw ProtocolError.invalidJSON }
                if b == 92 {
                    guard index < bytes.count else { throw ProtocolError.invalidJSON }
                    index += 1 // Decoder validates escapes and paired Unicode surrogates.
                }
            }
            throw ProtocolError.invalidJSON
        }
    }
}
