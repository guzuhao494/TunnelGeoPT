# TunnelGeoPT 验证协议（辅助开发版）

## 1. 协议状态

- Version: auxiliary_validation_v0.1
- Date: 2026-08-13
- Verification status: PREREGISTERED / NOT RUN
- Scope: 合成硬岩隧洞卸荷代理模型的数据效率筛查

本文所有门槛均为建议并需在首个 smoke 运行前冻结。当前没有结果。“至少 20% 高保真样本节省”是 Go/No-Go 标准，不是已验证事实。

## 2. 可检验主张

唯一主张候选为：在合成分布和锁定高保真测试 case 上，动态提升式预训练使用 80% 高保真训练 case 时，不劣于同架构从头训练使用 100% 高保真训练 case，且不增加物理违例。

不得由本协议推出：模型学会真实岩爆机理、能从试样迁移到工程尺度、能替代数值求解器、能用于现场预警或控制。

## 3. 数据版本与 case 原子性

### 3.1 `case_group_id`

规范化序列至少包含：

`section_family + section_parameters + material_field_seed + joint_network_seed + dimensionless_material_parameters + initial_stress_tensor + stress_orientation + excavation_schedule + unloading_schedule`。

对规范化 JSON 做 SHA-256 得到 `case_group_id`。网格、时间步、保真度、求解重启、采样点和增强不是新的独立 case，只能作为该组的子记录。

### 3.2 冻结拆分

- 三类断面内分别按 hash 排序并拆分 70% train、15% dev、15% locked test。
- 所有保真层和派生数据继承父 case split。
- 预训练池只能来自 train；这是一项归纳评估，不允许用 test 几何做无标签预训练。
- HF 40/60/80/100% 训练子集从同一顺序嵌套抽取；每个方法和随机种子共享子集。
- 保存 `manifest_hash`、`config_hash`、生成器版本、求解器版本和容器/环境摘要。

## 4. 数据与求解器门禁

### 4.1 Critical（任一失败即整批 No-Go）

- 跨 split 存在相同或近重复父 case。
- 非有限字段、错位张量分量、时间倒序或单位/无量纲化不可逆。
- test 几何/轨迹/统计量被预训练或超参选择读取。
- 同一 case 的低保真与高保真落入不同 split。
- locked test 已用于门槛、超参、早停或模型选择。

### 4.2 每 case 物理 QC（pilot 建议门槛）

| 检查 | 无量纲定义 | 门槛 |
|---|---|---:|
| 有限性 | NaN/Inf 数 / 字段元素数 | 0 |
| 平衡残差 | `||div(sigma)+b||_2 * R / (UCS*sqrt(N))` | <= 1e-2 |
| 边界残差 | `||sigma*n-t||_2 / (UCS*sqrt(Nb))` | <= 2e-2 |
| 能量闭合 | `|Wext-(dEstrain+Ekin+Ediss)| / max(|Wext|, eps)` | <= 5e-2 |
| 损伤不可逆 | `count(d[t+1] < d[t]-1e-8) / N` | <= 1e-3 |
| 负耗散 | `count(delta_Ediss < -1e-8) / N` | <= 1e-3 |
| 节理穿透 | `max(penetration)/R` | <= 1e-5 |
| 网格审计 | 再加密后关键标量相对差 | <= 5% |

求解器离散形式不同可替换残差的实现，但不能放松量纲一致性或在看过结果后改阈值。每类断面至少 95% 的目标高保真 case 需通过；失败 case 必须保留原因，不能只补生成直到数量好看。

### 4.3 对称性、旋转与零载荷 sanity checks

- 对称断面 + 对称应力/卸荷：镜像点标量误差 <= 2%，向量/张量按正确奇偶性比较。
- 整体旋转输入：标量不变，向量和张量按旋转矩阵等变；相对误差 <= 3%。
- 零卸荷或零时间增量：响应增量应接近零，超过训练集响应尺度 1% 即失败。
- 增加网格密度时，峰值应力、总释放能、损伤区面积不能出现无界单调漂移。

## 5. 模型评估指标

### 5.1 主误差（越低越好）

先在每个 case 内计算，再等权跨 case 平均：

`S_field = 0.4*RelL2(sigma/UCS) + 0.25*RelL2(u/R) + 0.20*RelL2(damage) + 0.15*RelErr(total_dissipated_energy)`。

权重必须在 smoke 前冻结。分母接近零的字段使用训练集预注册尺度而非逐 case 极小真值，避免相对误差爆炸。除组合分数外必须逐项报告，不能用组合分数掩盖损伤局部化失败。

### 5.2 次要指标

- 损伤局部化：阈值由 train 固定的 damage mask IoU / boundary F1。
- 峰值：峰值环向应力误差、峰值响应时刻误差、总释放能误差。
- 校准：若输出概率分布，报告 90% 区间覆盖率与宽度。
- 物理：预测平衡、边界、能量、不可逆和负耗散违例率。
- 运行：训练 GPU-hours、推理时间和峰值显存；这些是工程指标，不参与科学主门槛。

## 6. 基线与负对照

主基线为相同骨干的 `Scratch`。还必须运行 `Static geometry pretrain`、仓库当前的 `Random lift` 和 `Shuffled Stress-Lift`。所有方法共用数据 split、HF 子集、下游训练预算和超参选择空间。Multi-fidelity warm start 可作补充消融，但不能替代 Random lift。

需要满足的 sanity 条件：

1. Scratch 随 HF case 增多的学习曲线总体改善；若不改善，先排查标签/训练流程。
2. TunnelGeoPT 的同预算误差需方向性优于 Static；否则不能归因于 dynamics lifting。
3. Shuffled dynamics 不能复现正确动态条件的收益；若能复现，优先解释为额外训练/正则化而非物理耦合。
4. metadata-only 模型不得可靠预测 case 难度或目标；若能，检查 solver 状态和文件组织泄漏。
5. 标签置乱后模型应回到无技能水平。

## 7. 主统计比较

### 7.1 冻结比较

- Candidate: `TunnelGeoPT`, 80% HF train cases。
- Reference: `Scratch`, 100% HF train cases。
- Test: 同一 locked high-fidelity case 集。
- Seeds: 至少 5 个训练随机种子。
- Statistic: `R = S_field(candidate) / S_field(reference)`。

按断面分层、以 `case_group_id` 为重采样单位做 paired bootstrap；禁止按节点或时间步 bootstrap。报告中心估计、双侧 95% CI，并用上侧 95% 界执行门槛。若有多个 checkpoint，只允许使用 dev 预先选定的一个。

### 7.2 预注册 Go 条件

全部满足才 Go：

1. `HF_candidate / HF_reference <= 0.80`；这定义了至少 20% 标签节省。
2. 主误差比的上侧 95% 置信界 `R_upper <= 1.02`。
3. 至少 4/5 seed 单独满足 `R <= 1.02`。
4. 至少 2/3 断面分层满足 `R <= 1.02`，且任一断面 `R <= 1.05`。
5. Candidate 的物理违例率不高于 reference 超过 1 个百分点，且二者均通过绝对 QC。
6. Candidate@80% 对 Static@80% 为方向性胜出，且 shuffled-dynamics 不满足主 Go 门槛。
7. 数据、求解器、拆分和网格审计无 critical failure。

### 7.3 No-Go 与 Inconclusive

- No-Go：critical leakage/QC 失败，或主误差门槛明确失败；不得事后把门槛改为 10% 节省。
- Inconclusive：置信区间跨越门槛、有效 case/seed 不足或训练不稳定；可扩大预注册 pilot，但必须保留本次结果。
- Go：只许可进入下一阶段，不等价于论文主结论或现场部署许可。

## 8. 防 solver bias 验证

同一求解器的低/高保真可能共享系统误差，因此：

- 预留不少于 locked test 10% 的 case 做独立离散方案或第二求解器复算（若资源允许）。
- 报告两求解器之间的差异下限；模型误差低于求解器间差异时，不宣称更高物理精度。
- 对 material law、damage regularization、边界截断半径和阻尼做局部敏感性分析。
- 如果模型只在训练求解器上表现好，而跨求解器性能显著恶化，则结论限定为 solver emulation。

## 9. 最小报告包

报告必须包含：冻结配置与 manifest hash、全部 QC 失败清单、case 级 split 证明、5 seed 原始表、HF 学习曲线、逐断面指标、负对照、物理违例、网格审计、资源实测，以及明确的 `Go / No-Go / Inconclusive`。不得只展示最佳 seed、最佳断面或精选云图。
