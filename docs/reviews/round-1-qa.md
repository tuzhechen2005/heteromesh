# 第一轮交叉评审：架构与质量

日期：2026-10-06；评审者 `/root/spec_qa`。已阅读 PRD、feasibility、workflow、AGENTS 和产品第一轮审查；不自批本人 system spec/test plan。

## 结论

产品目标与技术研究边界一致，支持进入修订。冻结前必须补齐以下事项；没有任何真机完成声明。

| 优先级 | 发现 | 处理要求 | 责任 |
|---|---|---|---|
| P1 | workflow 仅描述结构化批准，未冻结机器schema和事件优先级 | 明确评论schema、维护者校验、author/reviewer、完整SHA、最新有效结论、撤销/删除/编辑和阻塞优先；读取失败fail closed | root |
| P1 | review CI 若随PR来自不可信分支，会允许PR修改自己的门禁实现 | gate执行信任主分支代码；只读PR内容，不能在有写权限的评论workflow执行PR代码；精确SHA合并 | root |
| P1 | PRD暂停/生命周期/资源控制/干净环境复现尚未全部映射spec/test | 已在本人文档新增REQ-018..021、PAUSE/LIFE/POLICY/INSTALL及PRD映射，交独立review确认 | spec_qa，product复核 |
| P1 | 协议最初是字段清单，不足以跨语言独立实现 | 本人文档已补HTTP routes、status/errors、dtype字节金样、canonical摘要、shape上限；下一协议PR先提交schema与共享fixtures | spec_qa，research复核 |
| P2 | 研究Mac内存与可访问性陈述滞后 | 引用workflow单一硬件事实；16GiB已确认≠可分配内存已实测；iPhone现在NOT RUN | research |
| P2 | 发布门禁缺机器可读硬件证据 | test plan已加入PASS/FAIL/NOT_RUN与commit/模型/profile/设备绑定；开发PR可以明确NOT_RUN，最终能力发布不可以 | spec_qa，product复核 |

## 可执行性与安全检查

- Windows/macOS/Linux CPU矩阵可跑协议和状态机，但不能证明CUDA/MPS运行，需另列硬件测试。
- 本机无Xcode，Swift纯逻辑通过不能证明iOS app编译；CI iOS SDK编译及模拟器测试与真机签名/运行分开。
- H3许可是实际部署门槛；研究已给一手来源。通用协议与自有小模型可以继续，不能用它们替代目标模型最终验收。
- checkpoint必须涵盖视频/audio、两条scheduler及RNG；步内临时结果和持久步检查点区分，否则“已提交不重算”会阻止恢复。本人spec已用execution_epoch澄清。
- 同GitHub账号的非作者agent review是可审计流程记录，不是原生独立账号APPROVE；工作流必须继续明确这一限制。

## 第二轮要求

检查root修订的治理schema、安全触发与SHA失效语义；由product/research复核本人新增契约。设计冻结仅覆盖已确定工程接口，H3平台可行性仍待实验。
