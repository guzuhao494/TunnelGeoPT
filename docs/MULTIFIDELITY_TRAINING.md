# 多保真学习契约

`multifidelity_learning.py` 实现 v0.3 的公平学习与统计基础。所有方法共享一个 14 通道、3 输出的 DeepSets 点算子：11 个几何/载荷基础特征加 3 个粗网格应力分量。Scratch 的粗应力通道固定为零；Direct 与 Residual 使用完全相同的输入和骨干，区别仅是预测 fine stress 还是 `fine - coarse`。

Fine-label 子集以 `geometry_group_id` 为单位、按断面族平衡并严格嵌套。Mismatched-Coarse 使用同族无固定点置换，防止把增加数值通道误解释为正确的物理信息。指标先按 query 权重计算每 case 张量 Frobenius 相对 L2，再在父几何内平均载荷，最后对断面族等权。

训练只用 train，early stopping 只用 dev。每个 checkpoint 原子写入 CPU state dict、method、fine fraction、seed、config hash 和完整 train geometry id，并返回文件 SHA-256。正式 test 必须在这些 checkpoint 全部冻结后由数据层单独授权。

层级配对 bootstrap 的第一层为训练 seed，第二层为各断面族内的父几何；同一父几何的全部载荷须在调用前已聚合。该模块本身不读取文件或解锁 test，因此不能绕过数据层访问审计。
