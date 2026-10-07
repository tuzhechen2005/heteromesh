# v1 协议与首轮实现契约

状态：待双轮评审冻结；本文件把 system-spec 的需求细化为可实现接口。兼容版本：`1`。

## D-01 技术与阶段

Python 3.11+ 桌面控制器/worker，NumPy CPU 参考，PyTorch CUDA/MPS 可选后端；Swift 6 Package 实现协议、客户端和实际计算，iOS App 前台运行。第一段可执行模型为自有确定性 tiny Transformer；它是完整系统的数值与通信试验，不是 H3 支持声明。H3 另有固定版本组件适配、单 block 和全模型验收。

第一版 coordinator 是一个受用户信任的节点。worker 主动通过 HTTPS 领取任务，iPhone 不要求后台监听。单个作业按片段顺序推进，所有远端结果影响后续计算。worker 无执行任意代码/动态插件接口。

## D-02 HTTP 与信任

所有 v1 路由为 HTTPS。创建集群时本机生成/保存自签证书，用户把 SHA-256 证书指纹通过带外方式交给 worker。客户端必须在发送秘密或请求体之前验证 peer 证书原始 DER 指纹；不得因为没有公网 CA 而静默忽略校验。证书改变要求重新配对。

控制请求和响应使用 JSON，最大 64 KiB；artifact请求/响应为二进制流，使用该路由独立的payload上限而非64KiB；带 body 的请求必须有 `Content-Length`；无 body 的 GET/DELETE 可以省略该字段，重复或无效长度拒绝，首版不支持 chunked transfer encoding。回复含 `protocol_version: 1`。HTTP 400 参数错误、401 身份错误、403 权限错误、404 不存在、409 幂等冲突/状态冲突、413 超限、422 不支持的计划/算子、503 等待资源。错误 JSON 为 `{ "protocol_version": 1, "error": { "code": "INVALID_TENSOR", "message": "...", "retryable": false } }`，不包含 traceback 或秘密。

凭据为至少 32 bytes 随机值，以 `Authorization: Bearer ...` 发送。admin 凭据只保存在创建集群设备；node 凭据按节点独立，撤销立即拒绝所有新请求。一次性配对 token 默认 300 秒过期，每 token 只消费一次，成功注册 node 后返回其长期凭据。配对请求重复不重发已丢失的凭据，须重新创建 token。配对 token 不在 URL 或日志中出现。

| Route | 权限 | 请求/效果 |
|---|---|---|
| `GET /v1/health` | 无秘密信息 | 返回 version 与服务状态，不列节点/任务 |
| `POST /v1/pairing` | admin | 创建一次性配对 token 和 expiry；只返回给 admin |
| `POST /v1/nodes/register` | 配对 token | body 包含 capabilities；返回 node_id 与 node token |
| `GET /v1/nodes` | admin | 返回节点状态与脱敏能力 |
| `DELETE /v1/nodes/{node_id}` | admin | 撤销凭据并使未完成租约失效 |
| `POST /v1/heartbeat` | node | 更新自身能力/活动状态；不能替其他节点更新 |
| `POST /v1/work/lease` | node | 返回分配给自身的 task 或 `task: null`，短轮询不持有无限连接 |
| `POST /v1/work/result` | node | 校验租约、attempt、输入/模型摘要与输出制品后至多提交一次 |
| `POST /v1/work/error` | node | 报告受限错误码；OOM/unsupported 默认不重试 |
| `POST /v1/jobs` | admin | 提交已知模型/profile/显式设备计划；`Idempotency-Key` 必需 |
| `GET /v1/jobs/{job_id}` | admin | 状态、当前片段、进度、脱敏执行证据 |
| `POST /v1/jobs/{job_id}/cancel` | admin | 原子取消；取消后不再接受新结果 |
| `POST /v1/jobs/{job_id}/pause` | admin | 设置pause_requested，在最近完整推理步边界落盘后进入paused，暂停后不再发新lease |
| `POST /v1/jobs/{job_id}/resume` | admin | 仅从有效已提交检查点/状态恢复；重新检查资源 |
| `PUT /v1/artifacts/{sha256}` | admin 或获授该制品写权限的 node | 有界二进制流上传，哈希一致才原子发布 |
| `GET /v1/artifacts/{sha256}` | admin 或持有该制品读取授权的 node | 只允许访问自身租约输入或已提交输出；不能按哈希猜其他任务数据；授权按job/epoch/lease检查，即使字节digest相同也不能自动继承其他job授权 |

所有 task/作业相关消息以 `job_id, recovery_epoch, task_id, step_index, fragment_id, attempt_id, manifest_digest, input_digest` 绑定身份；task 附明确 `inputs`（张量名→制品 digest）、运行参数、输出张量规格、deadline。worker 不得自行改变 fragment、模型或输入。

控制时间字段 `deadline`、配对 `expires`、节点 `last_seen` 均为整数 Unix 秒；数据库内部可保留更高精度。控制 DTO 不依赖浮点时间解码。工作节点对模型输入/契约错误报告 `INVALID_TENSOR`，运行后端不可恢复错误报告 `EXECUTION_FAILED`，内存不足报告 `OUT_OF_MEMORY`；这些错误默认终止当前作业，不自动重试。

## D-03 tensor-v1 二进制

线上帧：4 bytes 大端无符号 header 长度 + UTF-8 JSON header + payload。header 最大 65536 bytes；payload 默认上限 256 MiB，实际限值取节点、会话和计划的最小值。禁止负数、bool 伪整数、非有限数字、重复 JSON keys、重复张量名以及压缩负载。

header 必需字段：`version:1, name, dtype, shape, byte_order:"little", layout:"contiguous", payload_bytes, sha256`。身份/租约与模型由包裹该制品的 task 绑定，内容寻址使同一张量可复用。`name` 为1..128个ASCII字母数字、点、下划线或横线。dtype 取 `float32/float16/bfloat16/int32/int64/uint8`；bfloat16仅声明线上原始位模式支持，未声明该compute dtype的节点不得直接运算；允许profile显式指定bfloat16→float32的精确值扩展，禁止隐式转float16。shape 最多8维，每维为0..2147483647整数（即使已有0维也检查其他维上限），0元素合法，空shape代表标量；乘积在分配前检查 <= payload预算/dtype字节宽度。

payload 仅紧密排列的小端字节；hash 为 payload SHA256 的64位小写十六进制。整个文件的SHA256另作为 artifact ID。读取必须拒绝多余尾字节、截断、不匹配长度/类型、无效UTF-8和哈希错误。先检 header 再分配/写入。流传输每次最多4MiB，每会话最多2个在途4MiB块（总缓冲上限8MiB）；一个或多个数据流共享此上限，磁盘中的制品不计为常驻内存；禁止按声明长度预分配未经检查的 buffer。

跨语言金样本：float32 形状 `[2,3]`、值 `[0,1,-2,0.5,3.25,-4]`；另含标量、空数组、float16/bfloat16/int32/int64/uint8。Python 与 Swift 都读取同一 fixture 并验证字节；Swift 模型运算首轮仅float32，其余仅证明传输，不宣称算子支持。

## D-04 资源与片段契约

capabilities 必需字段：`protocol_version, platform, runtime, backend, supported_ops, wire_dtypes, compute_dtypes, memory`。节点名可选且非身份。memory 使用 bytes，独显设备分别 `host_budget_bytes` 和 `accelerator_budget_bytes`；统一内存设备设置 `unified:true` 并仅用 host预算，accelerator为null。unknown为null，未知关键预算的计划不准入。

manifest 中每片段声明输入输出tensor规格、所需算子/dtype/backend、权重哈希与固定revision。内存 estimate 包含 `resident_weights_bytes, peak_activations_bytes, workspace_bytes, transfer_buffers_bytes, runtime_overhead_bytes, safety_margin_bytes, loading_peak_bytes`，分别归属 host/accelerator；两者不能盲目合并。加总运行峰值与独立加载峰值取max，再与预算比较。多个常驻片段须累加权重；顺序片段仅可共享已证明不重叠的临时空间。

已验证profile与实验profile明确区分。H3 未有实測profile前不准声称自动规划可运行。tiny-transformer是自有fixture，具有明确输入shape、确定性权重、参考结果与数值门槛。

## D-05 幂等、租约、恢复

coordinator 以 SQLite 事务持久化作业、attempt、制品授权、节点撤销与提交账本。logical key `(job_id, recovery_epoch, step_index, fragment_id, input_digest)` 唯一；任何物理重算都记录新attempt。lease 只授予显式计划中的node；先检查幂等账本：重复相同已提交结果返回既有commit，冲突结果返回409；未提交且deadline到期的结果拒绝。取消不撤销取消前已完成的提交，但禁止任何新提交。重新配对/服务重启废弃未提交旧lease。

心跳与长计算分线程/任务，默认间隔5s；30s无心跳进入suspect，60s进入unavailable（可配置），不因估计时间过短取消健康慢节点。task deadline默认30分钟、用户可增大；重试上限3次，只有transient重试。取消和成功使用同一事务竞争，终态不可逆。节点缺席时作业等待，不隐式改用控制器CPU。

已提交片段输出先落盘并通过哈希校验，再原子写入账本，重启可重用。若从较早步骤检查点显式恢复，增加recovery_epoch，允许重新计算旧epoch已提交片段但仅新epoch结果可推进作业；旧epoch结果不能污染当前状态。用户暂停在完整推理步边界保存已提交状态（包含H3完整scheduler），取消优先于未完成暂停。暂停期间不发新lease；重启保留paused。

首轮tiny模型每个完整循环写checkpoint；H3必须保留两条scheduler和音视频latent等完整状态，尚无适配时明确unsupported。checkpoint引用全文件校验通过的制品；采用临时路径+flush/fsync+原子索引，不覆盖上一有效版本。恢复先比对manifest/profile/input与计划，换节点时产生新计划版本并重新prepare。

## D-06 首轮边界及后续验收

实现可分为协议/资源、控制服务/worker、Apple客户端、模型与CLI四个PR工作流；先合并共享协议fixture，再并行实现依赖方。各PR只声明已实现条款。任何未实现route明确404/unsupported，CLI不能暗示成功。

静态显式放置先落地，随后自动计划、模型安装缓存、可观测界面、H3真实分片和完整设备验收。所有后续项仍属产品目标，不能把首轮开发当最终完成。

## D-07 规范化摘要与样本

协议header自身不作canonical digest；artifact以实际完整文件字节摘要为准，重排header可改变artifact ID。manifest/profile等结构摘要使用本项目限制的canonical JSON：UTF-8，key按Unicode码点排序，无空白，不转义非ASCII字符，禁止浮点数、非有限数、重复key与非字符串key，只接受有符号64位整数、字符串、bool、null、数组、对象。浮点模型参数以规范字符串存储（例如 `"0.125"`），避免Python/Swift浮点JSON格式差异。摘要为该字节串SHA256。需要浮点数组的计算输入放tensor制品。

有效/无效JSON和tensor金样本作为首个protocol PR的一部分提交，必须在实现前冻结测试期望。纯Swift协议测试和Python协议测试读取同一文件；不能两边各自产生“各自正确”的fixture。wire支持不意味着计算支持。


bfloat16 fixture以UInt16小端原始位模式保存：`0000, 8000, 3f80, c000, 0001, 7f7f, 7f80, 7fc1`（此列表为数值hex而非线上byte顺序）。codec保留Inf/NaN位模式，不擅自改值；模型执行准入另拒绝非有限激活。测试必须区分“可传输”与“可计算”，并验证finite check不能把bf16 reinterpret为fp16。float32小网络compute初始容差 `atol=1e-5, rtol=1e-4`。

字符串escaping：引号与反斜线必须转义，使用 `\b \t \n \f \r`，其他U+0000..001F用小写 `\u00xx`，slash不转义，其他Unicode字符直接UTF-8输出，拒绝unpaired surrogate。金样覆盖中文、非BMP字符、控制字符和key排序；不得以RFC8785替代本限制规范。
