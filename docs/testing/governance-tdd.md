# 治理门禁 TDD 记录

作者：`/root`。2026-10-06。覆盖 FLOW-01/FLOW-02 的代码逻辑，不证明真实模型或设备能力。

1. RED：先创建 `tests/governance/test_review_gate.py`，运行 `python3 -m unittest discover -s tests/governance -v`。退出1，`ModuleNotFoundError: No module named 'tools.review_gate'`，此时门禁实现不存在。
2. GREEN：实现parser/evaluate/可信GitHub元数据adapter后同一命令退出0，12个测试通过。
3. 需求追踪：`python3 tools/check_docs.py` 退出0，21个REQ均在测试计划出现。该检查只证明文档映射，不证明测试已经实现。

行为测试覆盖：缺批准、自审、旧SHA、非维护者评论、不同reviewer阻塞优先、撤销、编辑无效、删除的持久失效、缺scope、伪bool版本、未知agent和多个作者。网络adapter的实际运行证据由本PR提交状态及后续review工作流运行记录提供。

代码审查绑定PR具体SHA，文档审查不能替代门禁代码的非作者审查。初始PR使用同一验证器本机发布agent-review status，因可信workflow尚未进入main；此启动例外不能延伸到后续PR。

非作者代码审查后追加RED/GREEN：先加事件权限与编辑身份回归测试，缺invalidation_agents导入失败；实现仅维护者记录可失效且优先读取编辑前reviewer，15个测试通过。修复普通外部评论编辑/删除可阻塞合并的漏洞。
