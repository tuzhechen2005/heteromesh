# 异构分布式视频推理：可行性与证据

调研日期：2026-10-06。状态：设计输入，尚无目标设备实测。作者：独立技术调研 agent。

## 1. 结论与范围

目标保留为 Windows RTX 4060 Laptop（8 GB VRAM / 16 GB RAM）、M4 MacBook Air（主代理已用本机 sysctl 确认 16 GiB 统一内存）和两台 iPhone 15 共同完成同一次视频生成，并通过模型分片扩大可运行模型容量。连接、心跳、远程矩阵乘法、小型模型测试只是工程里程碑，不能证明 H3 已能跨这些设备生成视频。

**事实**：MiniMax 官方已公开 H3 权重与社区许可；Diffusers 有真实 H3 modular 实现。**推断**：按连续 Transformer block 切分能避免每个算子都跨网同步，适合家庭局域网的容量优先设计。**未实测**：完整 H3 在此四节点组合上的可完成性、耗时、可承受分辨率、iPhone 可运行分片大小及数值偏差。不能承诺所有闲置设备加入都会加速。

优先实现显式分片执行图和独立平台执行器，保留 H3 为首个模型适配目标。iPhone 必须在最终硬件验收中执行影响生成结果的模型计算，仅展示控制界面、传输文件或生成后的视频转码不满足这一目标。

## 2. 已查证的模型事实

| 内容 | 证据及其边界 |
|---|---|
| 公开权重 | MiniMax 官方模型库同时包含原始 `FL2VA/`、`Ref2VA/` 及 Diffusers 格式。2026-10-06 metadata API 返回 revision `42ed227ee7df40d41602854ae760620d6eb651fe`。仅读取了公开文档、配置和文件列表，未下载权重。[模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3) / [metadata API](https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3) |
| 本地目标 | 本地 H3-Base 与官方完整 Context-IR/Base/Regenerate-2K 系统有区别，不能把 API 参与的完整流程称完全离线。[官方 README](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/README.md) |
| 正确框架入口 | 采用 `ModularPipeline` 与显式 workflow；不能复制 Hub 自动生成的通用 `DiffusionPipeline(...).images[0]` 片段作为已验证视频用法。[Diffusers H3](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3) |
| 权重规模 | Diffusers 说明单个 BF16 denoiser 61.7 GB、conditioner 62.1 GB，int8 流式卸载示例仍预期约 75 GB 主机 RAM。此为上游报告，不是本项目测量；GB 与 GiB 不混用。[同一文档 Memory](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3#memory) |
| block 参数 | 已读取官方转换配置：50 个主 block、hidden size 5376、2 个 refiner 层、text dim 5120、video channels 24、audio channels 32。[固定 revision 配置](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/transformer/config.json) |
| 生成依赖 | 音视频共同去噪、拥有两条 scheduler。分片不能丢掉音频状态或把生成视为独立逐帧任务。[Diffusers H3](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3) |

### 许可会影响部署范围

H3 属于有条件开放权重，不能称为全球无条件使用的开源模型。官方 LICENSE 的 I.5 列出美国、欧盟、英国、韩国为排除区域；II 说明这些地区可联系 MiniMax 申请许可；III 包含再分发约定。项目不能将权重纳入自己代码的许可证，也不应默认再分发。用户环境的 America/Chicago 时区不是所在地或授权的证据，需要在真实 H3 部署前确认适用情况。通用调度与协议开发、自有小模型验证可以继续；若授权尚未满足，H3 硬件验收保持未完成，不能靠替换成另一个模型宣称原目标完成。[官方完整许可](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE)

## 3. 平台运行时选择

| 平台 | 首选实现 | 备选路线 | 必须验证 |
|---|---|---|---|
| Windows NVIDIA | Python/PyTorch CUDA，复用 H3 PyTorch 结构 | CPU 部分卸载，磁盘逐 shard 加载 | 驱动/torch 版本、量化 kernel、峰值显存、加载期 RAM；8 GB 不能直接加载完整 BF16 denoiser |
| M4 Mac | Python/PyTorch MPS，首先保持同一模型结构 | MLX 重新实现特定分片；CPU fallback 必须显式报告 | 不支持的算子、dtype、attention 工作区、统一内存峰值、fallback 比例 |
| iPhone 15 ×2 | 原生 Swift 节点；对确定形状的模型分片试验 Core ML 导出 | ExecuTorch Core ML delegate；MLX Swift 实现分片；必要时直接 Metal | 真机编译/安装、算子语义、可用内存、峰值、中途后台/锁屏、温控、连续运行 |

PyTorch MPS 是 Mac 上的 GPU 后端，不能把桌面 Python worker 原样视为 iOS 应用。[PyTorch MPS](https://docs.pytorch.org/docs/stable/mps.html)

Core ML Tools 支持直接转换 PyTorch，但动态形状与算子转换有约束。应先固定少数生成档位，导出一个真实 block，验证输出后再增加模型范围。转换成功不代表一定跑在 ANE，也不代表完整 H3 可转换。[转换文档](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html) / [形状约束](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)

ExecuTorch 官方提供 Core ML delegate，支持 CPU/GPU/ANE 与 fp16/fp32；它解决部署路径，不提供 H3 的自动兼容保证。[ExecuTorch Core ML](https://docs.pytorch.org/executorch/stable/ios-coreml.html)

MLX Swift 的 Package manifest 声明 iOS 支持，适合作为动态张量计算备选；这也不等于已有可运行的 iPhone H3 port。选择替代执行器前应比较同一个 block 的精度、内存和维护成本，避免同时维护三套未经验证的 iOS 实现。[MLX Swift manifest](https://github.com/ml-explore/mlx-swift/blob/main/Package.swift)

## 4. 真实内存与通信预算

每个节点的准入条件应分别检查：`权重驻留 + 输入输出激活 + 算子临时工作区 + 传输缓冲 + runtime开销 + 安全余量 ≤ 实测可用预算`。加载期还要独立核查原始 shard、转换中间结果、量化后缓存同时存在的峰值。不能先构造完整大模型再切层；必须按文件索引、参数名和分片清单加载本节点需要的权重。

Mac CPU/GPU 共用统一内存，不把 16 GiB 重复计算两次。PC 系统 RAM 与 GPU VRAM 分属不同预算；可用于卸载但不等价。iPhone 的可分配预算取真机反馈与小步探测，不能用物理 RAM 减固定常数猜一个可靠上限。Metal 的建议工作集也只是参考，不是全系统内存授权。[Metal 建议工作集](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)

量化可降低权重，不能等比例降低激活与 attention 临时内存。即使全部权重总大小小于集群内存之和，某一层的单节点峰值仍可能超限；遇到这种情况需要更小视频档位、attention/FFN 分块或更细模型拆分，不能让调度器仅凭总和准入。

**通信示例，非 H3 实测**：假设 packed sequence 为 20,000 个 token、hidden 5376、每元素 2 bytes，一份边界激活约 215 MB。三处跨设备边界、30 次模型评估将传约 19.4 GB，仅计算此张量；还没有辅助张量、重试、GPU/CPU 拷贝。有效吞吐 50 MB/s 时裸数据传输约 387 秒。真实 sequence 长度由具体工作流推导，不能使用此示例估计用户视频耗时。

第一版采用连续层分片：不同设备顺序执行各自层，不要求同时满载。对单个视频，这主要增加可运行容量；对多个视频才能进一步考虑流水线重叠。tensor parallel/sequence parallel 会引入更频繁通信，留作后续研究，不是第一版的默认方案。

## 5. H3 边界契约与数据面

源码中的真实 block 输入包含 `hidden_states`、`temb`、`adaln_indices`、`rotary_emb`，以及可选 mask；分片不能只发视频 latent。整个模型外围还涉及模态索引、位置与音频状态。应从固定源码 revision 生成 adapter 契约，并逐项与上游单机轨迹对照。[Diffusers Transformer 源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_minimax_h3.py)

建议边界格式：版本化 JSON 元数据 + 有界二进制张量；声明 dtype、shape、连续布局、字节序、内容长度、SHA-256、task/step/stage/attempt、模型及分片哈希。fp16、bf16、fp32 分别标识，禁止把 bf16 字节当 fp16。量化格式作为模型制品的一部分记录；激活不默默有损压缩。网络传输无 pickle、无任意代码执行，无未校验远程路径。

建议控制面用配对后的 HTTPS 请求，数据面用有界分块上传与内容寻址缓存。Swift 节点可主动领取任务，避免要求手机后台监听服务；桌面与手机都遵守同一 schema。首版手动地址或二维码配对即可，自动发现不是推理正确性的前提。对 LAN 的访问也需要认证；发现不等于信任。

重试采用至少一次执行、单次结果提交：`task+step+stage+attempt` 绑定租约，过期结果不能污染新尝试。保存最近完整去噪步的视频/audio latents、两条 scheduler 状态、RNG 状态、模型/配置哈希、输入条件引用。若节点退出，从完整边界恢复；缺少相容替代节点则可恢复地等待，不假装仅剩电脑能装下丢失分片。检查点采用原子提交与校验，避免恢复半写文件。

## 6. iOS 持续执行的边界

iPhone 首版前台参与、显示任务进度和退出状态；用户可主动停止。锁屏或后台必须反映到租约和调度状态，不能把心跳过期等同永久故障。Apple 的 continued processing 可延续用户启动工作，但受系统调度与取消约束；后台 GPU 需要 entitlement，并查询 `BGTaskScheduler.supportedResources`。不能根据手机型号或 API 存在就宣称 iPhone 15 支持后台 GPU。[Apple 长任务文档](https://developer.apple.com/documentation/BackgroundTasks/performing-long-running-tasks-on-ios-and-ipados) / [Apple DTS 解释](https://developer.apple.com/forums/thread/794072)

真正的发布路线需要 Mac/Xcode、真机安装签名、用户授予局域网访问；CI 的模拟器成功不证明 GPU 或内存可用。iPhone 的后台限制不能作为永不实现手机计算的借口，也不能在产品验收中隐去。

## 7. 从开发到真实验证的证据阶梯

1. **协议验证**：自有小模型及固定张量金样；Windows/macOS/iOS 解码一致，拒绝错误长度、超限形状、未知 dtype、损坏哈希。验证超时、重复、乱序、陈旧结果、部分上传及取消。该层不证明任何视频模型能运行。
2. **平台验证**：Windows CUDA、Mac MPS、两台 iPhone 真机分别执行同一小型分片，比较 CPU FP32 基准；记录平台/版本、实际后端、峰值和原始日志。模拟器只能证明构建与部分逻辑。
3. **H3 最小真实分片**：取得适用权重授权后，以固定权重和真实形状验证一个 H3 block、refiner 或 VAE 子图的 CUDA/MPS/iPhone 输出。FP32→FP16/BF16→量化分别记录误差，不能跳过。
4. **H3 跨机单步**：桌面/手机执行真实分片，比较同一初始状态的完整去噪步，包括视频与音频 scheduler；验证全部边界张量及峰值。固定种子并不足以保证不同后端逐 bit 相同，误差阈值需依据基准提前写入 spec，失败后不得任意放宽。
5. **H3 完整生成**：至少 Windows + Mac + iPhone 在同一次生成中执行模型计算，之后扩展两台手机；无云推理调用，导出可播放视频（音频能力按已批准规格），保存分片来源、内容哈希、时间线、内存和网络指标。目标设备参与证据应能排除主节点偷偷算完整模型。
6. **容量与恢复实验**：对固定模型版本、精度、画幅、帧数和内存上限，比较单机与集群；说明收益属于更高容量还是速度。人为断开手机/网络，证明检查点恢复及输出有效性。强制人为限制单机预算的结果只能称“受限预算实验”，不能宣称物理单机绝不可能完成。

每阶段做 SDD 契约先行和 TDD：先明确可观察的失败/成功，再实现，再加入真实设备集成测试。最终完整验收必须附生成产物与设备记录，不能以 CI 全绿、mock 后端或 toy 输出代替。缺真实设备/权重/许可时状态为待验证，原目标保持未完成。

## 8. 资源与未决问题

- 已有硬件足以开始协议、桌面异构执行与 iOS 原型研究；是否足够完整 H3 取决于分片峰值、量化和视频档位，当前无保证。
- 需要 Windows 节点可执行环境、Mac Xcode/命令行工具与 iPhone 真机签名安装、同一局域网、足够磁盘空间。模型大文件仅在确定 checkpoint/格式/许可后按清单下载，避免原始与转换格式全部重复下载。
- 网络先测有效吞吐/延迟/抖动，不先购买设备。有线桌面网络可能降低等待，但不解决手机算子与峰值内存问题。
- H3 的语言编码器也非常大，必须与 denoiser 一样考虑分片或分阶段加载；不能以替换较小编码器后成功，宣称完整原版模型等价运行。
- 主代理本机 sysctl 已确认 Mac 为 16 GiB；用户确认 Windows 可开机并接入局域网，但尚未建立节点执行会话。用户确认两台 iPhone 暂不可用于真机验证，iOS 版本与安装条件仍未知。本机只有 Swift 6.2 / Command Line Tools，无完整 Xcode；Swift Package 测试可本地进行，iOS 模拟器构建拟用 CI macOS/Xcode runner，均不能替代真机。还需实际可用磁盘与节点预算。这些条件不阻碍通用系统实现，最终 H3/四节点认证保持未完成。

来源均为开发者/模型发布方的一手文档或源码。文档 `main` 会变化：实现前固定具体依赖和模型 revision，发布前重新校验许可与转换兼容性。
