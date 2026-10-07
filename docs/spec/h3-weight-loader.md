# H3 本地选定权重加载器

状态：实现输入；H3-A01/A02、REQ-006/007。真实 H3 权重/硬件 NOT RUN。

## W01 输入及来源

`load_selected(root, shards, weight_map, expected, *, max_selected_bytes, max_loading_bytes)` 返回独立存储的 CPU torch tensors。`ShardRecord(path, size_bytes, sha256)` 来自已被本地批准的 manifest，不能从待加载文件自身计算摘要后称已验证来源。`expected` 精确指定当前 stage/range 的 key→`WeightSpec(shape, dtype)`；dtype 仅 `BF16`、`F32`，不自动转换。调用方从固定上游 meta module 推导 shape，不从待加载 header 信任 shape。这里只验证所选 key；完整模型 index 身份由 metadata inspector 验证。

只读取包含所选 key 的 shard。相对路径只接受普通目录及文件，无空组件、`.`、`..`、绝对路径、反斜线、盘符或符号链接；root 由可信本地配置提供。文件必须 regular、长度与 manifest 一致。按 1 MiB 块计算整个相关 shard SHA256；不通过下载、不使用 pickle、不调用 load_file/from_pretrained。

## W02 header 及选择

safetensors 8-byte little-endian header 长度必须在 2..16 MiB 且不超过文件。UTF-8 JSON 必须无重复 key/非有限数；metadata 只允许字符串映射。每项恰含 dtype/shape/data_offsets，shape rank<=8、每维 0..2^31-1；offset 为非负整数，区间大小必须等于 shape×dtype，所有张量（包括非选定项）区间完整覆盖 payload 且不交叠、无尾随字节。零长度项允许位于当前边界。未知 dtype 明确拒绝，此加载器只覆盖 H3 原始 BF16/FP32 profile。

先验证所有相关 header、所选 expected shape/dtype 及预算，再 safe_open(framework='pt', device='cpu')，只对所选 key 调用 get_tensor，立即 clone 得到独立 CPU storage；不保留 mmap 视图。保持原 dtype，无精度转换。返回 mapping 可供 H3BlockRange 严格验证；finite 数值由执行器检查。

## W03 内存与并发边界

selected bytes 是全部输出 CPU 张量总大小。loading accounting 至少为 selected bytes + 最大所选单 tensor bytes（当前 mmap 源页与 clone 重叠）+ 1 MiB hash buffer + 所有相关 header 原始长度；两预算均须显式提供且先校验。该 accounting 是明确的数据开销下界，不是物理 RSS 硬上限：JSON/Python 对象、torch/runtime、allocator、mmap 页预读与系统文件缓存另需 caller 预留/测量。GPU 转移期间 CPU 输出仍存活，GPU 权重、激活及工作区必须另算；不能以 resident weight 预算替代总峰值。

文件目录须由调用方保证加载期间不被并发写入。读取前后比对文件身份/长度/mtime/ctime，并在 clone 后重新流式 hash；检测到修改拒绝并不返回结果。这不能对抗拥有本机文件写权限的恶意竞态（safe_open 按路径重新打开），不是安全沙箱；不可把网络上传的可变文件直接交给此接口。返回的 clone 不受之后文件修改影响。

## W04 验证

自有小 shard 验证跨 shard BF16/FP32 位值、所选 get_tensor 集合、独立 storage；错误摘要/长度/路径/重复 JSON/header/dtype/shape/offset/预算均在 materialize 前拒绝。测试原始 header 变体使用匹配 fixture manifest 摘要，避免只测到 hash 错误。文件变动注入必须被拒绝。CPU torch/safetensors 专用 CI 实测；默认轻量 CI 可 skip 可选依赖，不能将 skip 当执行通过。小 shard 与调用跟踪不能证明 5 GiB shard 在实际设备的物理 RSS；真实权重与峰值测试保持 NOT RUN。
