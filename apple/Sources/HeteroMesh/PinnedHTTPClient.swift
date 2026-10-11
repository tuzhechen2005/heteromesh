import Foundation
import Security

public enum TransportError: Error, Sendable {
    case invalidEndpoint, invalidPin, invalidPath, invalidResponse, responseTooLarge, httpStatus(Int)
}
public struct CertificatePin: Sendable {
    public let fingerprint: String
    public init(_ fingerprint: String) throws {
        guard fingerprint.utf8.count == 64, fingerprint.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { throw TransportError.invalidPin }
        self.fingerprint = fingerprint
    }
    public func matches(der: Data) -> Bool { CanonicalJSON.sha256(der) == fingerprint }
}

private final class PinDelegate: NSObject, URLSessionTaskDelegate {
    let pin: CertificatePin
    init(pin: CertificatePin) { self.pin = pin }
    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping @Sendable (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              let trust = challenge.protectionSpace.serverTrust,
              let chain = SecTrustCopyCertificateChain(trust) as? [SecCertificate], let certificate = chain.first,
              pin.matches(der: SecCertificateCopyData(certificate) as Data) else {
            completionHandler(.cancelAuthenticationChallenge,nil); return
        }
        // Explicit user-confirmed DER pin authenticates this private cluster certificate.
        completionHandler(.useCredential, URLCredential(trust:trust))
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping @Sendable (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        urlSession(session,didReceive:challenge,completionHandler:completionHandler)
    }
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        completionHandler(nil) // Credentials must never follow a redirect.
    }
}

public final class PinnedHTTPClient: Sendable {
    private let baseURL: URL
    private let session: URLSession
    private let pinDelegate: PinDelegate
    public init(baseURL: URL, pin: CertificatePin) throws {
        guard baseURL.scheme == "https", baseURL.host != nil, baseURL.user == nil, baseURL.password == nil,
              baseURL.query == nil, baseURL.fragment == nil, ["","/"].contains(baseURL.path) else { throw TransportError.invalidEndpoint }
        self.baseURL = baseURL
        let config = URLSessionConfiguration.ephemeral
        config.httpCookieStorage = nil; config.urlCache = nil; config.urlCredentialStorage = nil
        config.timeoutIntervalForRequest = 60; config.timeoutIntervalForResource = 1800
        self.pinDelegate = PinDelegate(pin:pin)
        self.session = URLSession(configuration:config,delegate:pinDelegate,delegateQueue:nil)
    }
    public func request(_ method: String, path: String, token: String? = nil, body: Data? = nil,
                        headers: [String:String] = [:], maxResponseBytes: Int = 65_536) async throws -> Data {
        guard path.hasPrefix("/v1/"), !path.contains(".."), !path.contains("?"), !path.contains("#"),
              maxResponseBytes >= 0, ["GET","POST","PUT","DELETE"].contains(method),
              let url = URL(string:path,relativeTo:baseURL)?.absoluteURL,
              url.host == baseURL.host, url.port == baseURL.port, url.scheme == "https" else { throw TransportError.invalidPath }
        var request = URLRequest(url:url); request.httpMethod = method; request.httpBody = body
        request.setValue("application/json",forHTTPHeaderField:"Content-Type")
        if let token { request.setValue("Bearer " + token,forHTTPHeaderField:"Authorization") }
        if let body { request.setValue(String(body.count),forHTTPHeaderField:"Content-Length") }
        for (key,value) in headers { request.setValue(value,forHTTPHeaderField:key) }
        let (bytes,response) = try await session.bytes(for:request,delegate:pinDelegate)
        guard let http = response as? HTTPURLResponse else { throw TransportError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else { throw TransportError.httpStatus(http.statusCode) }
        guard response.expectedContentLength <= Int64(maxResponseBytes) else { throw TransportError.responseTooLarge }
        var data = Data()
        for try await byte in bytes {
            guard data.count < maxResponseBytes else { throw TransportError.responseTooLarge }
            data.append(byte)
        }
        return data
    }
    public func json(_ method: String, path: String, token: String? = nil, value: JSONValue = .object([])) async throws -> JSONValue {
        let data = try await request(method,path:path,token:token,body:method == "GET" ? nil : CanonicalJSON.encode(value))
        let parsed = try CanonicalJSON.parse(data)
        guard parsed["protocol_version"]?.integer == 1 else { throw TransportError.invalidResponse }
        return parsed
    }
}
