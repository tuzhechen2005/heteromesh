# 第二轮产品交叉复核

日期：2026-10-06  
评审者：`/root/product`  
非作者审查范围：workflow、protocol-v1、system-spec、test-plan；不审查批准本人编写的 PRD。  
当前结论：**批准设计冻结并进入按规格实施**。第一轮主要产品问题已修复，下述 P1 在本轮复读后关闭；bfloat16 重复拼写是非阻塞编辑项。设计冻结不代表 iOS/H3 真机验收通过。

## 已在实际文件确认的修订

- workflow 明确 H3 是用户指定目标，替代模型不改变最终验收。
- REQ-018..021 和 LIFE/POLICY/DELETE/PAUSE/INSTALL/BACKEND 测试补足生命周期、用户限制、清理、暂停和安装需求；PRD 到 REQ 映射已存在。
- 检查点恢复引入 epoch，区分物理重算与可见提交；已提交同结果重发先返回既有收据，再拒绝过期新结果。
- 制品授权绑定 job/epoch/lease，不从相同 digest 自动继承其他作业权限。
- 流控统一为每会话最多两个 4MiB 在途块，总缓冲 8MiB，多流共享预算。
- 带请求体才要求 Content-Length；无 body GET/DELETE 可省略。
- 审查记录只追加，绑定完整 SHA，非作者审查、权限验证、撤销/编辑/删除事件与 fail-closed 规则已写入；明确共享账号不能提供独立 GitHub 身份。
- 开发制品允许明确 NOT_RUN，最终能力发布要求对应 commit/profile 的全部硬件证据 PASS；两台 iPhone 的最终测试 E2E-03 保留。
- 第一协议实现 PR 要先提交 JSON schema 与共享跨语言 fixture；首阶段 CPU/tiny 结果不能冒充目标模型。

## 尚待统一的精确契约

1. **P1 摘要算法：** system-spec §4 写 RFC8785，protocol D-07 写自定义 Unicode 码点排序及禁浮点 JSON。两者不等价；应使用同一明确算法和同一跨语言字节 fixture。
2. **P1 epoch 字段：** system-spec/test-plan 使用 execution_epoch，protocol 使用 recovery_epoch；必须统一 wire 和持久化字段名称。
3. **P2 暂停边界：** system-spec 要求完整步，protocol 允许片段（H3 完整步）。统一为明确 profile 安全边界及对应检查点完整性，或一律完整步，避免实现方分别选择。
4. **P2 dtype 列表：** protocol D-03 的 bfloat16 重复，移除重复值。

修订后直接复读相关行完成关闭记录；不需要重复扩展审查范围。研究未知的实际峰值、iPhone 运算能力、H3 权重适用许可与完整视频耗时仍按研究门槛处理，不因设计投票而变成已验证事实。

## 关闭复核

已再次读取 system-spec §4、§6 及 test-plan 故障段：摘要明确引用 D-07，epoch 统一 recovery_epoch，REQ-020 和状态机按声明的安全边界暂停，H3 采用完整步。P1 摘要和字段冲突关闭；暂停语义关闭。protocol dtype 重复不改变允许集合，可由作者直接修正编辑项。

本批准针对被评的设计文件，不替代后续具体 PR 完整 head SHA 的非作者审查，也不批准本人 PRD。第一协议 PR 的 schema/共享 fixtures、CI gate 与实际 RED/GREEN 仍需分别验证。
