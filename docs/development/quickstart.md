# Mac / Windows 双机接入（开发版）

此流程运行同一个自有小 Transformer 的分段推理，用于验证设备互联与数值正确性。
H3 视频生成、超出单设备容量的模型、iPhone 真机仍未验收。

## 安装

两台设备均需要 Python 3.11+、Git；协调节点还需要 `openssl` 命令。
以下在仓库根目录执行。Windows 使用 PowerShell，并将 `python3` 改为 `py -3.12`。

```sh
git clone https://github.com/tuzhechen2005/heteromesh.git
cd heteromesh
python3 -m venv .venv
```

Mac 激活：`source .venv/bin/activate`；Windows 激活：`.venv\Scripts\Activate.ps1`。
然后执行 `python -m pip install -e '.[torch]'`。若先测 CPU，可只安装 `-e .` 并选择
`--backend numpy`。MPS / CUDA 不可用会直接失败，不会自动换成 CPU。

## 在 Mac 启动协调节点

```sh
python -m heteromesh coordinator --host 0.0.0.0 --port 7443
```

数据默认保存在当前用户的 `~/.heteromesh/coordinator`。在另一个终端，为每台 worker
各创建一次邀请，将 `<MAC_LAN_IP>` 替换为 Mac 的局域网地址：

```sh
python -m heteromesh pair --host <MAC_LAN_IP> --out ~/mac-pair.json
python -m heteromesh pair --host <MAC_LAN_IP> --out ~/windows-pair.json
```

邀请含短期一次性凭据和协调节点证书指纹，只通过可信方式复制到对应设备，不提交仓库。
有效期五分钟。过期或已使用时重新生成邀请，输出文件不能覆盖已有文件。
Windows 文件权限继承用户目录 ACL；POSIX 文件采用 0600。

## 启动真实计算节点

Mac 本机：

```sh
python -m heteromesh join --pairing-file ~/mac-pair.json --config ~/.heteromesh/mac-node.json --backend mps
python -m heteromesh worker --config ~/.heteromesh/mac-node.json
```

游戏本 PowerShell，在复制好的邀请所在目录：

```powershell
python -m heteromesh join --pairing-file .\windows-pair.json --config "$HOME\.heteromesh\windows-node.json" --backend cuda
python -m heteromesh worker --config "$HOME\.heteromesh\windows-node.json"
```

两台设备需要能访问 Mac 的 TCP 7443 端口。GPU 后端由 worker 本地验证。
预算默认最多 1 GiB，且不超过探测到可用内存的一半；这是小 profile 的接入预算，
不是对 H3 所需内存的估算。M4 统一内存不会重复计入 GPU 容量。

## 提交及检查

在 Mac 另一个终端列出节点，复制两个 node_id：

```sh
python -m heteromesh nodes
python -m heteromesh submit-tiny --nodes <MAC_NODE_ID> <WINDOWS_NODE_ID> --steps 2
python -m heteromesh status --job <JOB_ID>
python -m heteromesh verify-tiny --job <JOB_ID>
```

最后命令会下载真实最终输出，与 NumPy 完整参考计算比较。`numerical_match: true` 只表示
该任务数值一致；物理设备、后端和运行版本还需单独记录，不能凭两个注册 ID 声称双机验收。
`pause` 在完整 step 边界停，`resume` 继续，`cancel` 取消；均接收 `--job <JOB_ID>`。

## 当前边界

- 固定小 profile，2–8 个节点、1–8 个 step；不同节点实际运行同一模型的不同连续 block。
- 节点掉线不在 Mac 上代算。任务会等待节点/租约超时，并遵循账本重试上限。
- 受限模型权重、iPhone 签名与真机安装、H3 调度器恢复不由本流程证明。
- 不自动开放防火墙，不扫描局域网，不把邀请或节点配置写进公共测试报告。
