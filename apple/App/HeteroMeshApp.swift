import SwiftUI
import HeteroMesh

@MainActor final class NodeModel: ObservableObject {
    @Published var address = ""
    @Published var fingerprint = ""
    @Published var pairingToken = ""
    @Published var status = "未连接。此开发版只运行 tiny Transformer，尚不支持 H3。"
    @Published var running = false
    private var task: Task<Void,Never>?
    private var worker: AppleWorker?
    private var generation = 0
    func start() {
        guard !running else { return }
        generation += 1; let runID = generation
        running = true; status = "正在验证证书并配对…"
        task = Task {
            do {
                guard let url = URL(string:address) else { throw TransportError.invalidEndpoint }
                let client = try PinnedHTTPClient(baseURL:url,pin:CertificatePin(fingerprint))
                let node = try AppleWorker(client:client)
                worker = node
                try await node.register(pairingToken:pairingToken)
                pairingToken = ""
                status = "已加入，前台等待模型片段。计算后端：Swift CPU。"
                try await node.run()
            } catch is CancellationError { guard runID == generation else { return }; status = "已停止参与；协调器可等待设备重新加入。" }
            catch { guard runID == generation else { return }; status = "连接或计算失败：\(type(of:error))。请检查配对、证书和协调器日志。" }
            if runID == generation { running = false }
        }
    }
    func stop() {
        generation += 1; task?.cancel(); task = nil; running = false
        status = "已停止参与。重新加入需新配对码。"
        if let worker { Task { try? await worker.heartbeat(state:"suspended") } }
    }
}

@main struct HeteroMeshApp: App {
    @StateObject private var model = NodeModel()
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            NavigationStack {
                Form {
                    Section("加入自己的局域网集群") {
                        TextField("https://协调器地址:端口",text:$model.address).textInputAutocapitalization(.never).autocorrectionDisabled()
                        TextField("通过可信渠道确认的证书 SHA-256",text:$model.fingerprint).textInputAutocapitalization(.never).autocorrectionDisabled()
                        SecureField("一次性配对码",text:$model.pairingToken)
                        Button(model.running ? "停止参与" : "加入并开始计算") { model.running ? model.stop() : model.start() }
                    }
                    Section("节点状态") { Text(model.status) }
                    Section("运行方式") {
                        Text("保持应用在前台。切到后台或锁屏会停止领取任务；重新参与需要新的配对码。当前预算为 64 MiB，只接受有界的 float32 小模型片段。")
                        Text("参与节点会收到模型权重和中间数据，请只加入信任的个人设备集群。")
                    }
                }.navigationTitle("HeteroMesh 节点")
            }
        }.onChange(of:scenePhase) { phase in if phase != .active { model.stop() } }
    }
}
