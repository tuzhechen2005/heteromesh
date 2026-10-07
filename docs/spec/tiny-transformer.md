# 自有 tiny Transformer 执行契约

用途：G1跨后端真实模型计算，不是H3、视频模型或容量最终证明。operation=`tiny_transformer_block_v1`。

一个block输入：hidden、condition形状[N,D]；wq/wk/wv/wo[D,D]；w1[D,F]、b1[F]、w2[F,D]、b2[D]；ln1_weight/ln1_bias/ln2_weight/ln2_bias[D]。全部float32、小端、有限值。N为1..512，D为1..256，F为1..1024，首profile N=3,D=4,F=8。epsilon在parameters中用十进制字符串，范围1e-8..0.01，默认"0.00001"。输入集合必须精确匹配，拒绝未知参数/操作。

计算顺序：

```
z = layernorm(hidden + condition, ln1_weight, ln1_bias, epsilon)
scores = (z @ wq) @ transpose(z @ wk) / sqrt(D)
attention = stable_softmax(scores, axis=-1) @ (z @ wv) @ wo
h = hidden + attention
z2 = layernorm(h, ln2_weight, ln2_bias, epsilon)
output = h + relu(z2 @ w1 + b1) @ w2 + b2
```

layernorm每行使用总体方差（除D）；输出tensor名hidden、shape[N,D]。float32跨后端atol=1e-5/rtol=1e-4。非有限结果必须报错。CPU NumPy与真实PyTorch CPU/CUDA/MPS后端实现同一计算；请求不可用backend失败，不能静默回退。所有权重是输入制品，手机不能用预计算答案替代运算。

图中后一个block的hidden引用前一个block实际输出；条件与权重可各自存放。测试包括解析可手算attention、独立FFN残差、非零condition、非方形F、重复迭代、坏shape/dtype/NaN和不可用后端。原生Swift/Metal消费者使用相同契约；模拟器与主机结果不等于iPhone实测。
