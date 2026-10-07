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
