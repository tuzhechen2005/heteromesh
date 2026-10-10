# README 信息架构调研

日期：2026-10-09。范围：开源项目首页的结构、导航和入门体验；不据此推断 HeteroMesh 的模型、平台或性能能力。

## 参考项目

仅使用项目自己的 README 作为来源，借鉴组织方式，不复制品牌、文案或能力声明。

| 项目与来源 | 可借鉴的组织方式 | 本仓库的采用方式 |
| --- | --- | --- |
| [exo](https://github.com/exo-explore/exo/blob/main/README.md) | 从日常设备组成推理集群的使用场景切入；快速开始按环境展开；运行和配置入口明确。 | 开头说明同一模型的异构分片目标；本机 CPU 示例与实际双机/GPU 接入分开；保留物理设备验收边界。 |
| [vLLM](https://github.com/vllm-project/vllm/blob/main/README.md) | 简洁居中的项目头部与导航；能力分组；上手、文档、贡献入口容易找到。 | 使用项目名、短定位、真实工作流徽章和锚点导航；以能力/证据矩阵呈现当前进展。 |
| [Ray](https://github.com/ray-project/ray/blob/master/README.rst) | 先说明价值，再建立架构抽象与组件入口；更多资料和参与渠道集中整理。 | 用协调器、worker、存储解释执行路径；以按任务选择的文档表格连接规格、证据和开发流程。 |

## 面向 HeteroMesh 的取舍

- **保留中文主文档。** 与现有规格、测试计划及开发文档一致，头部附简短英文定位。
- **首页回答完整的入门问题。** 说明项目目标、目前能运行什么、如何安装与验证、如何贡献。
- **让读者可以复现。** 提供无需正式模型权重的本机双 worker 流程；标明终端分工、占位符替换、成功状态和数值验证输出。
- **把实现与验证环境分开。** Swift App 代码与 CI 模拟器编译不能表示 iPhone 真机通过；H3 Transformer 小配置组件测试不能表示完整视频生成通过。
- **用架构图解释实际路径。** 当前张量和任务经协调器流转；不画成已经实现的设备间直接传输或自动网状调度。
- **不补造项目背书。** 不加入虚构的基准、下载量、稳定版、社区渠道、发行包链接或许可证徽章。仓库缺少项目级 LICENSE 的事实直接披露，不替维护者选择授权条款。
- **路线图依照证据验收。** 已实现项限定为对应软件能力；物理设备、完整目标模型、离线和容量实验保持待验收。

## 本地事实核对入口

| README 内容 | 本地依据 |
| --- | --- |
| Python 版本、安装 extras | [pyproject.toml](../../pyproject.toml) |
| CLI 参数、凭据文件与默认地址 | [cli.py](../../src/heteromesh/cli.py) · [CLI 规格](../spec/local-cli.md) |
| tiny 节点/step 范围、profile 与容差 | [demo.py](../../src/heteromesh/demo.py) · [Tiny Transformer 规格](../spec/tiny-transformer.md) |
| 已合入增量与仍未完成的集成 | [整合记录](../development/pause-recovery.md) |
| 同机 NumPy/MPS 与 Python/Swift 验证 | [CLI 证据](../testing/evidence/cli.md) · [Apple 说明](../development/apple.md) |
| H3 Transformer 和权重加载范围 | [结构证据](../testing/evidence/h3-fullgraph.md) · [加载证据](../testing/evidence/weight-loader.md) |
| CI 徽章 | [Python 工作流](../../.github/workflows/python.yml) · [Apple 工作流](../../.github/workflows/apple.yml) |
| 贡献与验收规则 | [开发流程](../development/workflow.md) · [测试计划](../testing/test-plan.md) |

本次仅修改文档，按仓库流程检查链接、格式和事实，不为排版变更虚构行为测试的 RED。
