# 暂停分支整理与恢复入口

日期：2026-10-08。范围：整理既有分支、保留未提交代码、合入已审增量。没有开展新的模型功能或硬件部署。完整项目目标仍未完成。

## 既有 PR 整合

| PR | 增量 | 整合状态 |
|---|---|---|
| #12 | 固定上游 H3 主计算层分段执行 | 已合入 `e13c1b0` |
| #14 | 完整 Transformer 前段/计算段/后段及双 scheduler 更新 | 已合入 `4abdbc0` |
| #15 | 只加载选定 safetensors 权重及显式加载预算 | 已合入 `b347f8f` |
| #8 | Swift worker 与前台 iOS App 原型 | 已合入 `3d96a0a` |
| #11 | 恢复分支隔离与旧结果失效 | 已合入 `3ee71d8` |
| #13 | 开发 wheel/sdist 构建与干净安装验证 | 已合入 `6ae1d9f` |
| #16 | 原始目标缺口审计（历史快照） | 已合入 `1ff1bf9` |

每次主分支变更后，下一个 PR 同步 main；非作者核对新的精确 SHA，所有检查通过后才合并。旧批准和旧 CI 不能批准新的 SHA。

## 本次软件验证

在整合代码提交 `b3f99c0b218e2fa0d4802039307c8332b5b05363` 上，以 Python 3.12 和固定 H3 可选运行时执行 `python -m pytest -q -rs`：266 PASS、1 SKIP（本机无物理 CUDA）。后续整理仅增加文档，不修改该已验证代码。远端各 PR 同步主分支后重新通过自己的全部 CI，包括三平台协议/任务测试、Apple 编译及跨语言 TLS、H3 自有权重结构测试、Windows/Linux loader 和三平台开发包安装。它们属于软件验收；不代表完整项目或目标模型的硬件验收。

## 尚未提交的工作

以下文件由原作者继续拥有，本次未代提交或合入。原工作区保留；另将完整文件内容复制到私有本地目录 `~/.codex/heteromesh-recovery/2026-10-08/`。该目录的 `manifest.json` 记录分支、原 HEAD、状态、路径、字节数与 SHA256；副本已逐文件比对。`paused-branches.bundle` 另保存两条本地暂停分支的完整 Git 历史（包括网络 spec 和 RED 提交），已通过 `git bundle verify`；其大小和 SHA256 也在 manifest 中。已从 bundle 克隆到两个全新的临时仓库，在原 HEAD 覆盖对应备份后，两个工作区的 git status 和全部 7 个文件哈希均与原状态一致；网络 spec 也成功恢复。临时验证仓库已清理。它不是 GitHub 制品，也未复制模型权重或节点凭据。临时工作区消失时可从副本恢复，但应先新建相应分支工作区，避免覆盖已有修改。

| 原作者/分支 | 原工作区 | 保留文件 |
|---|---|---|
| `/root/product`，`codex/model-checkpoints` | `/tmp/heteromesh-checkpoints` | `.github/workflows/h3-checkpoints.yml`、`docs/spec/h3-checkpoint-store.md`、`src/heteromesh/h3_checkpoints.py`、`tests/test_h3_checkpoints.py` |
| `/root/research`，`codex/h3-network-stages` | `/tmp/heteromesh-h3-network` | `src/heteromesh/h3_network.py`、`tests/h3_process_node.py`、`tests/test_h3_network.py` |

检查点分支最后新增回归后的完整测试结果缺失，不能把此前通过结果用于现有全部代码。网络分支引用旧 loader 依赖，也没有 PR。它们都需要同步已合依赖、完成测试、作者提交 PR、独立审查和 CI 后才能合入。

恢复时先核对备份 manifest 与 bundle 哈希；原仓库中的分支/提交若已不存在，先从 bundle 导入到新的恢复分支，再从 `manifest.head` 建立独立工作区，覆盖对应 `checkpoints/` 或 `network/` 子目录中的 4 或 3 个文件。不要直接从 main 检出后覆盖：网络 spec 仅存在于本地暂停分支的已提交历史中。保存原始状态后，再处理 main 依赖同步和冲突。

恢复时先读取 `docs/spec/h3-checkpoint-store.md` 和 `docs/spec/h3-network-stages.md`，检查工作区 diff，再在相应环境运行：

```sh
python -m pytest tests/test_h3_checkpoints.py tests/test_h3_graph.py tests/test_h3_runtime.py -q
python -m pytest tests/test_h3_network.py -q
```

上面是两个不同工作区的恢复命令，不是当前主分支已存在这些模块或本次运行成功的声明。既有测试环境为 Python 3.12；H3 可选运行时应按 `requirements/h3-runtime.txt` 安装，网络/loader 还需 `requirements/weight-loader.txt`，且不得为默认测试下载正式模型权重。先保存修改再同步，禁止对脏工作区使用 reset/clean。

## 证据边界与后续工作

小配置、自有权重、CPU 测试和 iOS 模拟器构建只证明各自的软件行为。官方完整 H3 权重未下载，Windows–Mac 实机协作、两台 iPhone 实际计算、完整视频生成、容量对照和最终离线恢复验收均未完成。

本次整理之后，下一项开发仍是 H3 网络阶段与完整检查点的集成；随后连接真实 Windows/Mac 节点，完成相同计算的数值与内存验证。文本编码器、VAEs、逐层/分片加载、实际峰值与手机模型后端仍需实现和验证。
