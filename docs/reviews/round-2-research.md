# 第二轮复核：protocol-v1 技术契约

日期：2026-10-06。评审者：`/root/research`。对象：新增 `docs/spec/protocol-v1.md`（D-01 至 D-06，修订进行中）。此为设计评审，不是代码 PR 批准。

## 结论

HTTPS、证书带外指纹、授权制品、二进制长度检查、平台内存预算和真实计算边界已经具体化，较首轮显著可实施。`float32 tiny Transformer` 明确只是基础验证，未冒充 H3，认可。以下 BF16 契约建议本轮直接纳入；暂停与恢复 epoch 正由作者补充，需完成后再冻结。

## BF16 建议：现在接受字节传输，计算能力独立声明

推荐 D-03 从第一版加入 `dtype:"bfloat16"`，而非将它留作不确定的未来协议升级。每元素两字节、小端、IEEE BF16 位模式：符号 1 位、指数 8 位、尾数 7 位。tensor codec 只验证和保持字节，不自动转 FP16。Swift 首轮可以使用 `UInt16` 保存原始 bits，仍只声明 float32 模型运算。这样可以提前验证真实 H3 的数据交换，而不冒充已有 BF16 计算能力。

能力字段应拆分 `wire_dtypes` 与 `compute_dtypes`；若保留现有 `supported_dtypes`，必须明确定义为 compute dtype 并增加 wire 字段。执行计划要求 `wire_dtype`、`compute_dtype` 与可选的明确转换阶段。拒绝从“能decode bfloat16”推导“能运行bfloat16 block”。

建议 manifest 转换契约限定为：`none`、`bfloat16_to_float32_exact`、`float32_to_bfloat16_rne`，并分别声明输出 dtype。首轮无需实现所有转换，声明不支持即拒绝执行。BF16→FP32 对有限数值可由 raw bits 左移16位得到精确值；反向舍入会有误差，必须独立测试 ties-to-even、overflow、subnormal 和正负零。不要把这些转换隐含在网络 codec 中。

NaN/Inf 策略分层：codec 保持并可 roundtrip 任意合法位模式；模型输入/结果检查默认拒绝非有限值并报告数值错误。BF16 规范传输金样本至少包括 `0x0000`（+0）、`0x8000`（-0）、`0x3f80`（1）、`0xc000`（-2）、`0x0001`（最小正subnormal）、`0x7f7f`（最大有限正数）、`0x7f80`（+Inf）、`0x7fc0`（NaN）。例如 `0x3f80` 线上字节为 `80 3f`。校验 fixture payload/hash，不仅校验转换后的浮点相等。

如果作者选择本轮明确拒绝 BF16 也不阻碍 G1，但必须将 H3 边界扩展列为未完成必需项；不得把首轮 wire protocol 的五种 dtype 覆盖宣称 H3 互操作完成。新增支持字段时，旧端必须可识别并拒绝自己不支持的类型，不能因 protocol_version 同为1而盲目接收。

## 冻结前其余具体核对项

| ID | 严重度 | 处理要求 |
|---|---|---|
| R2-R-01 | P1 | D-02 GET等无body请求是否仍需Content-Length应明确：只对有body请求必需更易互操作。64 KiB应明确只约束控制JSON，不限制artifact整个二进制body；artifact限制由tensor/header/会话预算共同决定。 |
| R2-R-02 | P1 | D-04 的 manifest/profile/input/plan digest 需要明确规范化方式：可限定 RFC 8785 JCS，或固定内部受限规范（ASCII keys、sorted键、禁止float/重复keys、UTF-8、无空白等），不能Python/Swift各自JSON序列化后直接hash。普通artifact ID可直接hash原文件bytes，无需重排header。 |
| R2-R-03 | P1 | schema/完整请求响应fixtures仍须落地，至少涵盖register、lease、result和tiny manifest/plan；纯字段列表不能确定required、null、未知字段和response格式。可作为第一个spec-only契约提交，之后再写服务实现。 |
| R2-R-04 | P1 | 补pause/resume、重启epoch后复核：暂停在安全边界停止新租约，取消与恢复使用事务；重启旧attempt拒绝规则与“重复已提交结果返回原commit”优先级须确定。不可简单将所有旧session结果拒绝而破坏已成功提交请求重试的幂等性。 |
| R2-R-05 | P2 | D-03 shape有0时，逐维仍须限定范围（推荐每维<=2^31-1），避免`[0,巨大整数]`因乘积0通过却令不同语言解析/转换溢出。字段整数需统一范围、禁止bool及JSON指数/小数冒充整数。 |

## 不需要再次改变的决策

首版集中协调、静态显式分片、worker主动领取任务、CPU参考、iOS原生前台执行、真实硬件另行验收均可继续。H3支持仍需真实权重/算子/边界/单步与完整视频证据；本轮协议批准也不能消除授权和真机访问缺口。

## 修订后 system-spec / test-plan 复核

实际读取了本轮更新的 system-spec 与 test-plan：已加入 RFC8785 摘要、整数范围、BF16 编码/显式 FP32 扩展、暂停/恢复 epoch、模型生命周期测试及 H3 许可引用，首轮相应缺口基本关闭。

发现新的 **P1 R2-R-06：两个规范之间存在冲突**。system-spec 的 `/hello`、`/nodes/{id}/poll`、`/attempts/{id}/result`、`/jobs/{id}/actions`、`/tensors/{id}` 与 protocol-v1 的 `/health`、`/work/lease`、`/work/result`、独立 cancel/resume、`/artifacts/{sha256}` 不同；BF16 接受/拒绝也冲突。冻结前请指定并保留唯一规范，建议 system-spec 引用 protocol-v1 的具体路由/帧格式，避免两个作者分别维护相同细节。schema/fixture 必须从唯一规范实现，不能让桌面和Swift各自选一份。

状态：等待作者整合后最终复核。未将任何研究待验证项视为已完成。

## 最终整合复核（2026-10-06）

实际再次读取 D-01–D-07 及修订 system-spec。确认 BF16 raw bits、wire/compute能力拆分、逐维范围、body上限例外、暂停/恢复epoch、先查幂等账本再拒绝迟到结果均已补齐；路由已统一引用 protocol-v1。首协议 PR 先冻结 schema/共享 fixtures/失败测试，再实现的顺序满足 SDD/TDD；不要求设计阶段先写 schema 执行程序。

最后发现三处需机械同步的文字：system-spec 仍写 RFC8785，但 D-07 定义的是本项目受限 canonical JSON（Unicode码点排序、有符号64位整数），两者不可混称；`execution_epoch` 应统一为 D-05 的 `recovery_epoch`；D-03 dtype 列表重复一次 bfloat16。另建议 D-07 明确 JSON 字符串最短 escaping 与拒绝孤立surrogate，避免Python/Swift输出不同bytes。完成这些同步后，架构与首轮开发契约可冻结；真实硬件/H3门槛继续单列，不阻碍G1实施。

最终核验：system-spec 已统一 D-07 canonical规则与 recovery_epoch。**研究评审结论：通过 G1 设计冻结。** 重复 dtype 为不改变语义的排版错误；canonical 字符串 escaping 在首协议 PR 共享 fixture/失败测试阶段具体冻结并交叉审查。此通过仅授权进入已设计范围实现，不表示 H3 或 iPhone 真机能力通过；原完整目标及其验收仍保留。

随后已实际核验 protocol-v1 清除了重复 dtype，并明确最短 escaping、slash不转义及拒绝孤立surrogate。**本轮发现全部已处置，最终设计评审通过；无剩余 G1 设计阻塞项。** schema与共享fixture仍按既定顺序在首协议PR先行创建、测试与独立审查。
