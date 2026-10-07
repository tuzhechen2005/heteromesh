# HeteroMesh

把 Windows、Mac、iPhone 与未来设备组成一个异构推理集群，让同一个大模型利用分散的内存和计算资源。首个目标是完全本地的视频生成，MiniMax H3 为首个目标模型。

**当前状态：设计与基础设施开发中。没有完成 H3 或 iPhone 真机推理验证。** 本仓库不把模拟节点、小型模型或绿色 CI 宣称为目标模型已能在四台设备运行。

## 设计与开发

- [产品需求](docs/product/PRD.md)
- [技术调研与事实边界](docs/research/feasibility.md)
- [系统规格](docs/spec/system-spec.md) / [协议 v1](docs/spec/protocol-v1.md)
- [测试计划与需求映射](docs/testing/test-plan.md)
- [两轮设计评审决议](docs/reviews/design-decision.md)
- [SDD / TDD 与跨 agent PR 流程](docs/development/workflow.md)

我们先验证跨平台张量与真实小型计算图，再适配真实视频模型分片，并在指定硬件上验证完整生成、断网运行、故障恢复及容量收益。文档中的阶段不缩小最终产品范围。

## 验证状态

| 能力 | 状态 |
|---|---|
| 产品、系统规格、测试计划与两轮设计审查 | 已形成文档 |
| 非作者 agent 审查门禁与需求追踪检查 | 自动化实现中，参见各 PR |
| Windows + Mac 物理设备共同推理 | NOT RUN |
| 两台 iPhone 实际模型计算 | NOT RUN，设备暂不可用 |
| H3 跨设备离线视频生成 | NOT RUN |
| 同配置单节点与集群容量对照 | NOT RUN |

H3 权重受其上游社区许可约束，公开权重不等于所有地区和用途均获授权；本仓库不打包权重。实际模型部署前确认适用条件。

## 开发检查

治理检查只需 Python 3.11+：

```sh
python -m unittest discover -s tests/governance -v
python tools/check_docs.py
```

PR 必须由非作者 agent 审查当前提交，且 CI 通过才合并。共享 GitHub 登录无法产生原生独立账号 APPROVED；本仓库明确使用可审计的 agent 审查记录与 CI 门槛，详细边界见开发流程。
