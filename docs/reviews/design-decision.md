# 设计会议与第二轮决议

日期：2026-10-06。协调者 `/root`；产品 `/root/product`；技术研究 `/root/research`；规格与QA `/root/spec_qa`。

两轮讨论分别记录于 `round-1-*` 与 `round-2-*`。每人审查他人文档；PRD由research/QA审查，spec/test由product/research审查，protocol/workflow由三位非作者审查。讨论后重新阅读实际修订，未用表决替代硬件测试。

## 已解决事项

| 发现 | 决定与实际修改 |
|---|---|
| H3候选措辞可能缩小目标 | H3为首目标，更换最终模型需用户明确变更；基础模型不替代验收 |
| 需求映射遗漏 | REQ-018..021补生命周期、资源政策、暂停安装与后端证据；测试矩阵映射全部21项 |
| 两处HTTP/codec规范冲突 | `protocol-v1.md`为唯一线上契约，system-spec只引用 |
| BF16与手机float32不同 | wire包含bfloat16原始字节，compute单独协商；显式无损扩展，不隐式转fp16 |
| canonical JSON不一致 | 使用D07限定格式，删除RFC8785表述；跨语言fixtures覆盖Unicode/控制字符 |
| 恢复已提交片段歧义 | recovery_epoch隔离重算；持久化结果先查幂等，再检查未提交租约过期 |
| pause粒度冲突 | 第一版统一完整推理步边界暂停，暂停和取消状态分开 |
| 缓冲与授权含糊 | 每会话2×4MiB在途块；artifact内容复用不等于跨作业权限 |
| 共用GitHub账号不能native approve | 真实非作者agent审查+完整SHA记录+可信默认分支CI gate，明确流程身份限制 |
| edited/deleted批准可能继续生效 | append-only review；编辑删除持久失效，重新review才恢复；合并绑定SHA |

## 冻结范围

批准工程设计v0.1、protocol v1进入TDD实现。协议PR先提交可执行schema与共享有效/无效fixtures，再实现Python/Swift消费者。接口变化须同步规格、测试和评审。

批准不证明任何目标模型或手机能力。H3许可适用性、真实模型块数值误差与内存、Windows/Mac互联、两台iPhone安装及运行、完整离线视频、恢复及容量对照均仍需实际证据。

## 实施分工与独立评审

1. root：仓库治理/CI/review gate，研究或QA非作者审查。
2. research：Python wire/schema/fixtures/资源规划，QA或product非作者审查。
3. spec_qa：Swift协议/原生计算与iOS应用，research非作者审查。
4. product：控制器持久状态、TLS/配对、任务租约和桌面worker，root/research非作者审查。
5. root：端到端模型图、CLI、H3真实分片适配与验收工具，product/QA非作者审查。

这些是实施工作包，不是阶段完成声明；GitHub PR仍逐个审查实际head SHA。
