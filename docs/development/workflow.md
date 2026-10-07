# 开发与评审流程

状态：设计草案。2026-10-06。

## 原始目标与完成条件

Windows、macOS、iPhone 与后续设备通过局域网协作执行同一个大模型的推理；首个真实目标是视频生成。多设备扩展可用模型容量是项目目的。单机优化、远程调用付费生成 API、进程模拟集群、仅交换心跳都不能代替最终结果。

H3 是用户指定的首个目标模型。替代模型只验证基础设施，更换最终验收模型须由用户明确变更目标。框架、权重、许可、设备内存和正确性必须逐项核验。未获得支持的设备/算子应报告不可运行，不伪装为远端执行，也不静默回退到主节点。

## 文档先行

1. 产品经理编写 PRD，技术调研记录原始来源，架构与 QA 定义带编号的规格和测试矩阵。
2. 第一轮会议由其他 agent 检查可行性、产品目标和测试覆盖；记录发现、责任人、处理结果。
3. 修订后第二轮复核，明确决定、仍未知的事项及证明方法。冻结的是可执行的工程契约，不把尚未实验的硬件能力写成事实。
4. 功能开发引用需求 ID、测试 ID 和设计决定；接口变更同时修改规格。

## SDD 与 TDD

每个行为变更先写一个有意义的失败测试，记录 RED 的命令与原因，随后最小实现，再记录 GREEN 与必要重构。文档修正不需要虚构 RED 测试。测试覆盖协议边界、数值误差、资源不足、断线、重复投递、取消、恢复、跨平台以及真实模型输出。

单元测试、模拟网络、真实跨进程、跨物理设备和目标模型生成分别记录证据等级。mock、合成张量和小网络用于开发，不可证明 H3 或真机能力。无法访问的设备保持 NOT RUN，不记为 PASS。

## GitHub 与多 agent

- 仓库：https://github.com/tuzhechen2005/heteromesh （public）。
- 每项变更使用独立 `codex/` 分支；并行实现使用独立 checkout/worktree。
- 功能作者提交 PR，正文写明 author agent、需求、RED/GREEN 证据、风险和实机状态。
- 非作者 agent 读取实际 diff、复查测试与规格后给出结论；有阻塞项先修改再审。
- 评审绑定完整 head commit SHA。后续修改使之前的批准失效。
- CI 必须通过才允许合并。不能把 push 前 CI 当作 GitHub 功能：本地检查先于 push，远端 CI 在 push/PR 后运行，门槛控制合并主分支。
- 除 GitHub 自动生成的初始 README 外，主分支变更都经 PR；不强推主分支，不跳过失败的门槛。

## 共享账号限制

当前所有 agent 使用同一个 GitHub 登录。GitHub 不允许同一个账号批准其创建的 PR，因此不能声称已有独立 GitHub 账号的 APPROVED。

本项目在此环境使用实际独立 agent 审查，在 PR 评论留下结构化的批准记录，由 CI 核对 reviewer agent 与 author agent 不同、SHA 相同，以及评论者有仓库维护权限。这是共享凭据环境下的 agent 交叉审查流程；agent 身份是流程记录，不是 GitHub 独立身份认证。将来有独立维护者账号时启用 GitHub 原生 required approvals。

CI 不执行 PR 评论中的代码。合并操作者检查测试与评审在最新提交上成立，并使用精确 SHA 防止合并期间变更。

## 测试与交付

持续集成至少覆盖 Linux/Windows/macOS 的核心行为，以及 Apple 客户端的编译和可自动执行的测试。GPU、iPhone 真机和 H3 权重实验属于独立硬件验收，不能由 CPU runner 的绿色状态替代。

持续交付构建可安装制品，先发布开发版；在真实设备矩阵完成前不发布“已支持 H3 四设备协作”的稳定能力声明。不得自动下载几十 GB 权重或分发受限模型到公共仓库。

完成开发后由非实现作者做完整回归，复核原始目标、所有需求及真实设备证据，输出未解决问题。只有完整验收成立才能宣布完成。

## 当前硬件事实

- 当前 Mac：系统查询 `hw.memsize = 17179869184`，16 GiB；型号 `Mac16,12`。用户称 M4 MacBook Air。
- 用户游戏本：RTX 4060 Laptop，8 GB VRAM，16 GB RAM；用户确认可开机并接入局域网，尚未建立工具连接。
- 两台 iPhone 15：用户确认目前不可用于真机测试；OS 版本未知。
- 本机 Swift 6.2 / Command Line Tools 可用；`xcodebuild -version` 表明未选择完整 Xcode，不能在本机假称完成 iOS 编译或真机签名。
- 设备接入凭据、局域网地址不写入公共文档或测试产物。

## 可机器校验的审查记录

PR正文必须有唯一 `Agent-Author: /root[/name]` 行。reviewer提交一条以 `<!-- heteromesh-agent-review-v1 -->` 开头的评论，随后一个JSON代码块：

```json
{"version":1,"reviewer_agent":"/root/research","head_sha":"完整40位小写Git提交SHA","decision":"approve","reviewed_paths":["docs/spec/protocol-v1.md"],"summary":"实际审查与验证说明"}
```

`decision` 只允许 `approve / changes_requested / withdraw`。reviewer与PR author必须不同，且在项目登记的agent列表中；评论者的GitHub账号必须有write/maintain/admin权限。此权限检查不证明agent身份独立，实际会话中的非作者审查记录才是其依据。

Gate读取当前PR head、正文、完整评论分页和权限；任何读取失败必须fail closed。只接受当前SHA、未编辑且格式完整的记录。按reviewer采用最新创建的记录；当前SHA任一reviewer的最新结论为changes_requested或withdraw即阻止合并，至少一个approve且无阻塞才通过。旧SHA不能批准新提交；新push、PR正文变化、review评论创建/编辑/删除都触发重算。审查记录采用只追加约定，编辑或删除事件立即使门槛失败；必须由非作者重新追加审查记录后才能恢复。修改原批准不会静默保持通过。

合并前操作者重新读取PR head、全部检查和当前评论，使用 `--match-head-commit` 绑定提交。实现阶段为上述规则编写FLOW测试。初始治理PR的门槛工作流尚未在默认分支时，使用本地同一验证器校验并上传commit status；由非作者另行审计，记录这一次启动例外，之后全部由默认分支可信工作流执行。

开发制品与稳定能力发布使用不同门槛：开发包可以附NOT RUN硬件矩阵，稳定能力声明必须具备当前commit/profile对应的全部必需真机PASS。发布门槛读取机器可读证据记录，缺字段、缺原始制品或版本不符即失败。
