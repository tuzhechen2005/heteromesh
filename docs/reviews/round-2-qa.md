# 第二轮交叉评审：架构与质量

日期：2026-10-06；评审者 `/root/spec_qa`。实际重读root修订的workflow与protocol-v1、研究第二轮记录。本记录批准设计进入有测试门禁的实现，不批准本人spec/test，不是GitHub代码PR批准。

## 已解决第一轮问题

- H3已固定为用户首个目标，不能由实现自行替换。
- workflow明确完整head SHA、非作者review、维护权限、append-only评论、最新reviewer结论、阻塞/撤销优先、编辑/删除失效及重新审批；共享账号限制不再被掩盖。
- gate后续从可信默认分支执行，启动例外要求非作者复核同一验证器；精确SHA合并防止检查后变更。
- protocol-v1明确HTTPS证书校验、主动pull、制品授权、二进制边界、控制64KiB和二进制例外、预算、BF16 wire/compute分离、pause/recovery_epoch、幂等收据优先。
- system-spec重复wire规范已移除，仅引用protocol-v1；本人文档已把epoch名称与canonical摘要规则对齐，不保留RFC8785另一路径。
- 产品新增生命周期/政策/安装/暂停追踪以及21需求测试矩阵已交非作者复核。

## 实现时必须验证的具体门槛

1. 评论删除事件不能仅靠当前评论列表推导，gate需持久保存事件失效记录或使用可信事件状态；直至新审查创建才能解除。FLOW测试必须覆盖“删除批准后重跑旧workflow仍失败”。
2. 评论workflow只能读取PR数据，不能checkout/执行PR head代码获取写权限。workflow变更本身也需独立审查。
3. D-07受限canonical编码的字符转义规则由共享字节fixture冻结，至少包括换行、引号、反斜杠、中文、非BMP键、边界int64；不能只测ASCII字典。
4. 明确测试旧epoch相同已提交结果重试仅返回原收据，不推进新epoch；旧epoch未提交结果拒绝。撤销凭证仍须在幂等查询前拒绝身份。
5. protocol-v1当前dtype列表重复一次bfloat16，建议清理笔误，不改变语义。
6. schema、有效/无效共享fixture仍属于第一协议PR的先行交付；运行时不能凭本设计批准跳过这些文件。

## 结论

上述门槛纳入实现测试后，可冻结基础工程方向并开始协议/资源PR。最终完成仍要求Windows、Mac和两台iPhone同次H3真实计算、离线视频、容量与恢复证据；当前硬件和授权缺口保持未验证。没有把设计同意解释成H3硬件可行性已获证明。
