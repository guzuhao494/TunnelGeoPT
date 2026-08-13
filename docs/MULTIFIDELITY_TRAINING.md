# 多保真学习契约

`multifidelity_learning.py` 实现 v0.3 的公平学习与统计基础。所有方法共享一个 14 通道、3 输出的 DeepSets 点算子：11 个几何/载荷基础特征加 3 个粗网格应力分量。Scratch 的粗应力通道固定为零；Direct 与 Residual 使用完全相同的输入和骨干，区别仅是预测 fine stress 还是 `fine - coarse`。

Fine-label 子集以 `geometry_group_id` 为单位、按断面族平衡并严格嵌套。Mismatched-Coarse 使用同族无固定点置换，防止把增加数值通道误解释为正确的物理信息。指标先按 query 权重计算每 case 张量 Frobenius 相对 L2，再在父几何内平均载荷，最后对断面族等权。

## Formal 训练身份

正式训练不得把已经切好的裸数组直接交给训练器。`build_training_selection` 从一个具体 `LearningBatch` 的 `splits / geometry_group_ids / section_families / case_group_ids` 派生身份：selector 只能选择 `train` 父几何，被选父几何的全部载荷会自动纳入；early stopping 自动使用全部且仅使用 `dev` 行。实际 fine fraction 按“选中 train 父几何数 / 全部可用 train 父几何数”计算，调用者给出的 fraction 只能作为一致性断言。因此把 100% 父几何伪报为 50% 会在训练前失败。

`build_training_contract` 把该 selection 与 method、冻结 config SHA-256 绑定。`train_formal_with_dev_selection` 会在计算前重新由 batch 推导 selection，并拒绝 stale、伪造、跨 split 的 optimizer 或 early-stopping 行。formal `TrainingOutcome` 记录 contract SHA-256；`save_formal_checkpoint_atomic` 只接受同一 contract 的 outcome，checkpoint v2 中的 fine fraction、train geometry IDs、train case IDs 与 selection hash 全部来自真实 selection，没有自报参数入口。

加载 formal checkpoint 时使用 `load_formal_model_from_checkpoint(path, contract=...)`。它同时核对外部期望的 config hash、selection hash 和 training-contract hash，并验证 checkpoint 内部 identity payload 的哈希一致性。仅调用 `checkpoint_payload` 时，formal 复用也必须传 `expected_config_sha256`、`expected_selection_sha256` 和 `require_formal=True`。

旧 `train_with_dev_selection` 与 `save_checkpoint_atomic` 保留给 v0.3 smoke runner 的管线/QC 检验，明确属于 legacy smoke API：它们接收裸数组或调用者提供的 subset 元数据，不能作为正式效果实验的防泄漏证据。

正式 test 必须在全部 formal checkpoint 冻结后由数据层单独授权。

## 统计输入契约

层级配对 bootstrap 的第一层为训练 seed，第二层为各断面族内的父几何。`hierarchical_paired_bootstrap` 要求 seed 唯一、父几何 ID 唯一；重复父几何会直接失败，防止把同一父几何的多载荷误当成独立样本。先用 `aggregate_case_errors_by_parent` 对 `[seed, case]`（或单 seed 的 `[case]`）按父几何平均全部载荷，再把返回的唯一 geometry IDs 和 section families 交给 bootstrap。

该模块本身不读取或解锁 test labels，因此不能绕过数据层访问审计。
