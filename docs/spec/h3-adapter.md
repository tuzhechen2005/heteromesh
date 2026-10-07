# H3 真实分片适配契约（源码核验版）

日期：2026-10-06；状态：适配开发输入，真实权重与设备实验 NOT RUN。

## 来源与固定身份

- Diffusers源码 commit：`c6df88a511a98740646ee55577b590c9852650ce`。
- 模型 metadata revision：`42ed227ee7df40d41602854ae760620d6eb651fe`。
- [Transformer源码](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/models/transformers/transformer_minimax_h3.py)。
- [去噪循环源码](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/modular_pipelines/minimax_h3/denoise.py)。
- [权重索引](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/transformer/diffusion_pytorch_model.safetensors.index.json)。
- [配置](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/transformer/config.json)。

本次仅读取源码、546-byte配置、64,488-byte权重索引；无权重下载。适用H3许可条件见研究文档，不能因读取metadata而声称已获部署授权。

## 查证结果

官方索引列出638个张量key，14个safetensors文件，`total_size=66,280,430,080 bytes`，约61.73 GiB。上游文档使用“61.7 GB”，容量计算必须使用精确bytes，不能混淆单位。文件索引提供key→shard映射，不提供每个tensor的shape/dtype/offset，不能凭索引保证所有参数的实际峰值或直接按整个shard加载。

50个主block各有12个权重key。本revision的关键分组为 `transformer_blocks.{i}.`；其余为输入projection、text refiner、时间embedding、输出norm/head。原始FL2VA格式和转换格式key不可混用。本适配第一目标固定转换格式 `transformer/`（t2va/fl2va），ref2va另作明确profile。

## H3-A01 分片语义

执行范围为半开区间 `[start,end)`，满足 `0 <= start < end <= 50`。不得构造完整50层模型后才切片。节点只构造并加载自己负责的blocks；prepare返回配置/source revision、key清单与制品哈希。缺少/多出key拒绝，不使用非strict load掩盖错误。

主block边界由以下值组成：

| 输入 | 语义 |
|---|---|
| hidden_states | `[B,S,5376]`；包含text/video/audio packed rows |
| temb | `[U,2688]`；原始路径保留FP32时间embedding |
| adaln_indices | `[S]`整数；值对应noise-level及模态表 |
| rotary_cos / rotary_sin | 与S匹配的位置旋转缓存；类型/形状按固定源码推导 |
| attention_mask | 默认不使用；启用必须是另一个受测profile |

输出只替换hidden_states，其余边界状态保留到下一分片；输入/输出模态索引与两条scheduler由外围阶段保管。源模型先完成输入投影/packed scatter/refiner，再进入连续blocks，最后执行输出norm/head并选取音视频rows。不得把每帧当可独立生成任务。

## H3-A02 精度与参数预算

原始路径包含FP32 projection、time embedding、输出head和rope，以及BF16主block，不能整个model直接统一转低精度。BF16传输/FP32扩展沿protocol-v1明确转换，转换后算子语义与误差需独立验证。temb在activation前降成BF16会改变所有层的调制，不能作为透明优化。

从此配置与模块声明推导，每个主block约645,571,840参数，若所有主block权重均BF16，纯权重为1,291,143,680 bytes（约1.202 GiB）。这是理论预算，不是可运行峰值；还需额外激活、attention/FFN工作区、加载/解量化缓冲、runtime。不能由1.202GiB推断iPhone15一定可执行一层。

H3-A02 inspector应输出key→文件清单、理论主block参数量和“未测激活”的明确标志。禁止将理论估计标记validated profile。safe loading必须逐tensor或有界读取所需tensor，读取全部约5GB shard可能超过小设备加载预算。

## H3-A03 单步与恢复

外围denoiser每步调用分布式Transformer，获得视频及音频velocity；两条scheduler各自仅更新target rows，conditioning rows保持。检查点保存两条scheduler、完整latent、条件引用、packed row布局、步索引和RNG/版本。只保存最后一份hidden_states不足以从完整步恢复。

本阶段未实现完整encoder、VAE、scheduler远程编排。小尺寸自有权重的同结构测试只能证明adapter控制与拼接，不证明真实H3质量、内存或iPhone支持。

## H3-A04 验证顺序

1. metadata inspector对固定配置/索引识别block key，拒绝缺key、重复/越界范围、路径穿越、不支持结构；不请求网络。
2. 用上游真实block类的缩小配置和自有随机权重，比较全序列与多个range组合；这属于结构测试，报告为E1。
3. 在适用许可与资源就绪后，单个真实权重block做CUDA/MPS/iPhone数值与峰值测量，保存输入输出参考。
4. 完整单步再完整视频；最终H3双桌面、两台iPhone和容量收益要求不变。

H3-A01–04的schema/测试先于实现；需要重新review的语义改变不得只改代码。

## H3-A05 第一轮独立审查补充

审核者 `/root/spec_qa` 要求明确精确结构及失败边界；2026-10-06 已据固定源码补充如下。

固定metadata原始字节SHA256：config `74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae`；weight index `ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e`。固定Transformer源码原始hash `92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef`，denoise源码 `4c2bc8b9856c38ba1692b02e35aa68ec92beda4db273e2e71f0ebbf997e19c79`。inspector输出同时携带来源revision、raw hashes及`evidence_level:metadata_only`。仅给相同revision字符串但内容不符必须拒绝。

正式profile限定B=1、S>0、U>0；hidden `[1,S,5376]` BF16，temb `[U,2688]` FP32，adaln_indices `[S]` int64且值在`[0,3U)`；rotary_cos/sin均`[S,96]` FP32（2×3×rope_freq_dim16）。NaN/Inf拒绝。首轮不支持mask、LoRA或任意改变维度的config；必须显式错误。缩小配置测试另标`synthetic_structure`，不能通过修改official profile偷偷降低模型规模。

每层恰含以下suffix：`norm1.weight`、`norm2.weight`、`attn.to_q.weight`、`attn.to_k.weight`、`attn.to_v.weight`、`attn.norm_q.weight`、`attn.norm_k.weight`、`attn.to_out.0.weight`、`ff.net.0.proj.weight`、`ff.net.2.weight`、`adaln_proj.linear.weight`、`adaln_proj.linear.bias`。检查必须比较集合，不能仅验证数量12；JSON重复key在解析时拒绝。

测试范围补充：各合法切分位置、空/反转/越界range、缺权重/多权重、重复key、错误dtype、S/U不匹配和索引越界、全0/非有限数、partitions重叠/空洞/顺序错乱、不同模型metadata hash。真实weight loading与完整分布式去噪仍需后续专门测试，不以metadata检查替代。

## H3-A06 主block运行包装器（下一增量）

依赖固定 `torch==2.8.0` 与上述Diffusers git commit；安装后核对PEP610 commit记录与Transformer源码原始hash。无此可选依赖时显式报缺依赖，默认轻量CI不安装大框架；专用CPU结构CI必须实际执行上游算子测试。

包装器只接收本地已经获得的主block权重，绝不下载或调用from_pretrained。初始化先验证range、官方/合成profile、输入state_dict的精确key集合、dtype、shape、有限值和权重预算，再按meta-device构建所需block并严格赋权；不构造50层完整模型。每个节点只保留本range权重。权重统一为主block原始BF16；其他精度要另增受测profile。

`synthetic_structure`允许小尺寸配置与自有随机权重，报告必须携带该标签；`official`维度固定原配置，但即使通过校验也只说明本地包装器接受了给定张量，不证明权重来源真实或完整H3质量。官方模型来源验证仍依manifest哈希/授权链。

执行时检查所有输入形状、dtype、device、finite及索引范围；FP32 temb和rotary不预先降精度，hidden保持BF16。返回同shape/dtype hidden，禁止非有限输出。no_grad/eval，无autograd，不把运行失败静默改在CPU上重跑。

测试：使用真实上游MiniMaxH3TransformerBlock与自有随机BF16权重，固定随机种子及非零位置/时间状态，对三层的每个合法分区比较单段/分段结果；校验辅助张量未被修改，坏key/shape/dtype/预算/索引/NaN及不兼容revision拒绝。此测试为E1 CPU结构正确性；CUDA/MPS/iPhone/原模型权重及去噪视频仍未验证。
