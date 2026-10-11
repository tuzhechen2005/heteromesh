# Apple 节点开发与验证

当前提交只交付 Swift 6 协议库及跨语言验证入口；worker、iOS App、真实模型计算会在后续 PR 完成。它不代表 iPhone/H3 已可执行。

## 本地验证

```sh
swift run --package-path apple protocol-checks fixtures
```

此命令运行同一组有失败退出码的真实断言，读取仓库公共 `fixtures/canonical.json` 和 `fixtures/tensors/`，比较规范化字节、内容校验、BF16位模式、非法JSON和超限张量。`protocol-checks` 是测试工具，不是网络worker。

完整 Xcode 提供 Testing 模块时也运行：

```sh
swift test --package-path apple
```

当前开发机只有 Command Line Tools，Swift 6.2 的 `import Testing` 与 `import XCTest` 都返回模块不存在。因此本机执行的是独立 conformance runner；不把导入失败称为业务测试的RED，不声称本机已跑swift test或iOS编译。CI使用Xcode/macOS runner运行swift test、共享fixtures，并用iOS Simulator SDK实际编译协议库。库编译不是App安装或iPhone真机计算。

## TDD与范围证据

- 先写协议断言，首次swift test被缺Testing模块阻止；改为共享断言runner后基本编解码和共享fixture通过。
- 新增严格schema的unknown-header-field拒绝断言；`swift run --package-path apple protocol-checks /tmp/heteromesh-protocol/fixtures`得到 `FAIL: Failure(message: "invalid input accepted")`，退出1。
- 增加与共享schema一致的精确header字段集校验后，同命令退出0且输出PASS。
- 基础本机已覆盖canonical键排序/转义/int64/重复键、tensor长度/哈希/形状/溢出/预算/尾字节、BF16保留NaN/Inf位模式及计算有限值检查。
- 公共fixtures来自独立research agent，Swift读取同一字节；不另生成一套Swift自己的参考答案。

协议库提供内存内codec，调用者仍须在网络读取时先验证header并按4MiB分块约束缓冲。本提交没有网络数据面，不能把内存内Data解码宣称为已完成流控。

## 待完成与真实硬件状态

| 项目 | 当前证据 |
|---|---|
| Swift协议逻辑与共享fixture | 本机PASS，CI结果见PR |
| iOS SDK库编译 | CI负责，非本机验证 |
| iOS App/前台生命周期/签名安装 | NOT RUN，后续实现 |
| Windows/Mac/iPhone共同模型计算 | NOT RUN |
| H3离线视频与容量扩展 | NOT RUN |

## 前台节点与实际计算（后续实现）

`apple-node` 和 iOS App 使用同一 Swift worker：配对后主动领取任务，心跳独立于计算，下载授权张量，执行 `tiny_transformer_block_v1` 的真实 float32 attention、LayerNorm、FFN、残差，再上传和提交结果。当前后端明确是 Swift CPU；没有把它表述为 GPU/ANE 执行。H3操作明确不支持。N/D/F与epsilon范围遵守tiny规范，默认总输入预算64MiB，每制品至多8MiB。

```sh
swift build --package-path apple --product apple-node
# 配对码通过环境变量传递，勿写入命令行或公共日志。
# 设置 HETEROMESH_PAIRING_TOKEN 后：
apple/.build/debug/apple-node run https://YOUR_COORDINATOR:PORT CONFIRMED_CERTIFICATE_SHA256
```

证书DER指纹通过可信渠道手动确认。URLSession握手回调在发送Authorization之前验证精确指纹，明确拒绝重定向；测试用真实本机HTTPS服务器证明错误指纹没有发送HTTP或Authorization，正确指纹能请求服务。

```sh
python3 apple/IntegrationTests/tls_integration.py
python3 apple/IntegrationTests/coordinator_integration.py
swift run --package-path apple protocol-checks fixtures fixtures/tiny-transformer
```

第二条需安装仓库Python依赖；它启动真实Python协调器和Swift进程，所有权重经TLS传输，服务端接收并验证实际算出的结果。这个测试是同一Mac上的跨语言/跨进程证据，不是两台物理设备。

独立NumPy goldens为N=3、D=4、F=8、有非零condition的两个依赖block，Swift输出按atol=1e-5/rtol=1e-4验证。TDD先加入Transformer与证书接口测试，因缺少实现失败，再完成实现；真实TLS集成曾暴露task认证回调缺失，跨服务集成暴露X-Output-Name授权字段缺失，修复后均通过。

## iOS App构建与限制

安装完整Xcode和XcodeGen后，在apple目录运行 `xcodegen generate`，再打开生成的HeteroMeshApp.xcodeproj。CI编译无签名Simulator App；真机需用户选择开发团队并签名，不在公共仓库保存签名凭证。

App前台加入、状态展示、停止按钮和scenePhase挂起处理已经实现；退出前取消领取并尽力报告suspended。iOS可能立即挂起，报告送达不能保证，协调器仍需失联策略。重新加入使用新的配对码；当前凭据仅在进程内存，不提供后台常驻或自动恢复承诺。局域网权限通过Info.plist声明。

实际iOS SDK编译结果以CI为证，本机缺SDK；iPhone安装、锁屏、热状态和内存压力仍NOT RUN。开发App只能证明应用与受测小模型链路，H3和两台iPhone最终目标保持未完成。
