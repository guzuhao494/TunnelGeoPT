# v0.3 多保真残差正式实验预注册

标识：`v0.3-mf-residual-prereg-v2`

日期：2026-08-13

唯一机器可读契约：`configs/multifidelity_formal.json`

当前状态：`frozen_preregistered_pre_generation`。独立 dev-only 网格收敛和优化轨迹校准均已完成；formal 数据尚未生成、formal 效果尚未计算、locked label 尚未打开。该配置现已允许开始 formal 生成，任何效果门槛均未因开发结果改变。

## 1. 研究问题与结论边界

在二维、均质各向同性、小应变平面应变隧洞开挖中，测试时允许运行一次粗网格 FEM。主问题是 `Residual+Coarse@50% fine-train parents` 能否在未见父几何及预定义 OOD 上达到 `Scratch@100%` 与 `Direct+Coarse@100%` 的非劣性能，并明显修正 raw coarse 场。

本实验只检验“指定粗网格到指定细网格的合成线弹性离散修正”。Fine FEM 是项目内的数值标签，不是真实世界真值。即使全部门槛通过，也禁止声称模型学习了破裂、损伤、岩爆、三维动力响应、微观到工程迁移或现场预测。

## 2. 冻结证据与禁止重写

冻结前只允许用与 formal seed、salt、几何和标签完全独立的 train/dev 开发样本完成两件事：

1. 证明候选 fine/ultrafine 网格满足本文件第 7 节的收敛门槛；
2. 用 dev label 校准优化超参数，并在新配置 hash 中留下修改记录。

两项门禁已经通过并写入配置：

- `mf-convergence-dev-v0.3.0` 在 24 个 dev-only case 上得到 fine-ultrafine median `2.116%`、p95 `3.036%`，三族 median 为 `1.769%/2.204%/2.564%`；24 个标签读取均为 dev，locked 读取为 0。其规范 config hash、config snapshot、metrics 和 manifest 文件 hash 已写入 `preformal_development_evidence`。
- 独立 smoke train/dev 优化轨迹中，Scratch100 和 Direct100 的最佳 epoch 仍分别刷新至 194 和 198，说明候选 200-epoch 上限可能截断；Residual50 的最佳 epoch 为 138，并在 164 停止。仅据这些 train/dev 轨迹，将 `max_epochs` 从 200 冻结到 300、patience 从 25 冻结到 35；learning rate、weight decay、batch size、min delta、模型和全部效果门槛均未改变，formal/locked label 读取为 0。

配置状态已更新为 `requires_dev_convergence_pass=false`、`eligible_to_generate_formal_data=true`。从此不得据任何 formal 结果重写网格、优化超参数、方法、阈值或统计规则。若需重跑，必须新建版本与 salt，声明旧 locked 集已见且不再具有 locked 身份。

配置 hash 使用 UTF-8、key 排序、紧凑分隔符及 `allow_nan=False` 的规范 JSON，再取 SHA-256；digest 存入 formal manifest，避免在配置内递归嵌入自己的 digest。

## 3. 原子身份与文件级防泄漏

- `geometry_group_id = hash(section family, normalized macro parameters, roughness realization, exact float64 boundary, outer-domain rule)`；
- `load_group_id = hash(normalized far-field stress tensor, material, boundary-condition convention)`；
- `case_group_id = hash(geometry_group_id, load_group_id, material)`；
- split 单位固定为父几何；同一父几何的全部载荷、coarse/fine/ultrafine、query、重网格和增强继承同一 split；
- train/dev/各 locked 分区之间，以及与旧 v0.2 locked test 之间，geometry、boundary、case 和 load hash 均须零交集；
- 归一化器和任何数据依赖预处理只在 train 上拟合。

Python 对象的下划线字段不是安全边界。正式数据必须拆成 public coarse/input store、train/dev fine-label store 和四个独立 locked fine-label store。训练进程不得收到 locked 文件路径。只有 35 个预期 checkpoint 均由 `TrainingContract` 绑定到真实训练行、原子落盘并由 `CheckpointRegistry` 验证唯一 SHA-256 后，独立 evaluator 才能授权读取。每个 locked 分区最多一次批量评价，访问日志必须 append-only 并参与 hash。

## 4. 几何、载荷、数据规模与种子

三类断面是 circle、horseshoe、straight-wall-arch。每族至少两个连续宏观参数改变归一化边界；边界 192 点、特征半径 `R=1`、外域尺度 8，学习集合粗糙度为 `[0.008R,0.025R]`。ID 参数位于形状参数全范围的归一化位置 `[0.15,0.85]`；geometry-OOD 至少一个参数位于 `[0,0.10]` 或 `[0.90,1]`。

材料固定 `E/sigma_ref=500`、`nu=0.25`。ID 载荷为 `sigma1/sigma_ref in [0.30,0.80]`、`sigma3/sigma1 in [0.45,0.85]`、主应力角 `[-45,45] deg`。Load-OOD 每父几何固定各取一个低侧压比、一个大转角及一个二者联合样本。

| 分区 | 父几何 | 每族父几何 | 每父载荷 | case |
|---|---:|---:|---:|---:|
| train-ID | 72 | 24 | 4 | 288 |
| dev-ID | 18 | 6 | 4 | 72 |
| locked-IID | 30 | 10 | 4 | 120 |
| locked-geometry-OOD | 30 | 10 | 3 | 90 |
| locked-load-OOD | 30 | 10 | 3 | 90 |
| locked-joint-OOD | 15 | 5 | 3 | 45 |

总计 195 个父几何、705 个 case。生成 seed 固定为 train/dev `310031`、locked-IID `310037`、geometry-OOD `310049`、load-OOD `310061`、joint-OOD `310073`；训练 seeds 固定为 `[103,211,307,401,509]`；split salt 固定为 `tunnelgeopt-v0.3-mf-residual-20260813-v1`。所有内部随机流由 seed、purpose、section、parent index 和 load index 的 SHA-256 确定派生。

## 5. 网格、公共 query 与 solver QC

同一 case 的边界、外域、材料、载荷和 query 完全相同，只允许网格尺寸变化：

| tier | `mesh_size/R` | `wall_size/R` | `farfield_size/R` |
|---|---:|---:|---:|
| coarse | 0.8 | 0.25 | 0.8 |
| fine | 0.4 | 0.0625 | 0.4 |
| ultrafine audit | 0.25 | 0.03125 | 0.25 |

每父几何在任何求解前生成 512 个公共 query：384 个距壁 `0.05R–2.0R` 的 nearfield 点、64 个位于岩体侧 `0.02R` 的 wall-offset 点和 64 个 farfield 点。每一网格都必须独立执行 point-in-triangle 定位和重心坐标复验；query hash 在各保真度间必须一致。

每个 required solve 必须同时满足：非有限比例为 0；自由自由度相对代数残差 `<=1e-9`；Clapeyron 相对能量误差 `<=1e-9`；最小有向三角形面积除以 `R^2` 不小于 `1e-12`；最小三角形质量不小于 `0.02`；wall/farfield tag 完整；洞内无单元质心；全部 query 可定位。单 case 未通过即标为无效并完整记录，不得静默补样；每个“分区 × 断面族”的有效 case 率必须 `>=95%`，低于该门槛时实验进入 `ABSTAIN`，而不是模型 No-Go。

## 6. 方法、训练集合与公平性

父几何子集按族平衡并严格嵌套：25/50/75/100% 分别是每族 6/12/18/24 个父几何，即总计 18/36/54/72 个。正式 checkpoint 组合固定为每个 seed 七个：

- `Scratch@100%`；
- `Direct+Coarse@100%`；
- `Residual+Coarse@25/50/75/100%`；
- `Mismatched-Coarse@50%`。

五个 seed 共 35 个 checkpoint。Coarse-only 无 checkpoint。Direct 和 Residual 使用相同的 14-to-3、hidden width 64、3 个 global-context block 的骨干；Scratch 的 coarse 三通道固定为零；Mismatched 使用同族无自匹配的固定错配。正式优化契约已冻结为 AdamW、learning rate `1e-3`、weight decay `1e-4`、case batch 8、最多 300 epochs、patience 35、min delta `1e-5`。上述 epoch/patience 变化仅来自第 2 节的 dev-only 轨迹，不涉及 formal 或 locked label。

训练损失区域质量固定为 nearfield/wall-offset/farfield `0.80/0.15/0.05`，每一区域内部权重归一化到对应质量；避免 wall-offset 与 farfield 因主指标只覆盖 nearfield 而在训练中被零权重忽略。模型选择只使用 dev-ID，绝不使用 locked 指标。

## 7. Fine-ultrafine 审计

开始 formal 生成前，独立 dev-only 收敛审计必须先通过相同门槛。正式审计再按“分区 × 断面族”对 case hash 排序并向上取整预选至少 20%，预计 144 个 case；locked 审计只能在 sealed generator 内运行，模型开发进程只能看到 pass/ABSTAIN，不得看到 case 值。

审计指标为 nearfield 面积加权张量 Frobenius fine-vs-ultrafine 相对 L2。全部预选 case 的中位数须 `<=3%`、p95 须 `<=5%`、任一断面族中位数须 `<=4%`。失败即 `ABSTAIN`。

## 8. 主指标、聚合与 bootstrap

主指标是每 case 的 nearfield 面积加权张量 Frobenius 相对 L2，剪应力平方权重为 2。先在父几何内平均全部载荷，再在每族内平均父几何，最后令三族等权。同步报告每族、每 OOD 类型、均值/中位数/p90、wall-offset 诊断、非有限值、失败 case、耗时和峰值内存。

正式置信区间用 20,000 次层级配对 bootstrap：先配对重采样五个训练 seed，再在每族内配对重采样父几何；同一父几何的全部载荷始终同行。主门槛使用单侧 95% 上界；同时报告双侧 95% 区间。任何主比率区间总宽度大于 `0.10` 均为 `ABSTAIN`。

## 9. Wall-offset 物理诊断的正确定义

Wall query 位于岩体侧 `0.02R`，不在精确洞壁上。因此这里既不能把绝对 traction-free 当作真值，也不能把离散点环的合力称为全域平衡残差。只计算相对于 fine 标签的 wall-offset 牵引与合力差异代理。

令 `a_i` 为和为 1 的弧长权重，`n_i` 为与 wall-offset 点配对的冻结边界岩体侧法向，`S_inf` 为仅由已知远场载荷得到的应力尺度，`M/F/C` 分别表示候选、fine 与 raw coarse：

```text
D_t(M,F) = sqrt(sum_i a_i ||(Sigma_M_i-Sigma_F_i)n_i||^2) / S_inf
D_r(M,F) = ||sum_i a_i (Sigma_M_i-Sigma_F_i)n_i|| / S_inf
```

`Residual50` 必须同时满足绝对 cap 与 coarse 非恶化门槛：

```text
D_t(M,F) <= 1.10 D_t(C,F) + 0.005
D_r(M,F) <= 1.10 D_r(C,F) + 0.0025
```

绝对 cap 为：IID `D_t<=0.10, D_r<=0.05`；geometry-OOD 与 load-OOD 各为 `D_t<=0.15, D_r<=0.08`。Joint-OOD 的 `0.20/0.10` 只报告，不进入主 GO。

## 10. Scientific GO、NO-GO 与 ABSTAIN

记 `R_s=Residual50/Scratch100`、`R_d=Residual50/Direct100`、`R_c=Residual50/CoarseOnly`。全部有效性与效果条件必须同时成立：

1. 数据、solver、mesh、leakage、sealed evaluation 和 fine-ultrafine QC 全部有效；
2. locked-IID 的单侧 95% 上界：`R_s<=1.02`、`R_d<=1.02`、`R_c<=0.70`；
3. geometry-OOD 与 load-OOD 分别：`R_s<=1.05`、`R_d<=1.05`、`R_c<=0.80`；
4. 至少 4/5 seeds 在 IID 上同时满足 `R_s,R_d<=1.05`，并在两个 OOD 上分别同时满足 `<=1.10`；
5. IID 三族的 `max(R_s,R_d)` 均不超过 `1.10`，至少 2/3 族不超过 `1.02`；任一 OOD load subtype 相对两个完整标签基线的点比率均不超过 `1.15`；
6. 预测非有限值为零，第 9 节两个 wall-offset 门槛全部通过；
7. checkpoint/config/manifests/access log/results/decision 均有 hash，四个 locked 分区各至多评价一次。

全部通过才是 `GO`，且只允许表述“在此合成线弹性分布上观察到 50% fine 训练父几何的多保真标签效率”。实验有效但效果或稳健性 gate 失败是 `NO_GO`。数据泄漏、test 调参、收敛失败、solver/mesh/QC 无效、少于五训练 seed、CI 过宽或评价契约无效一律是 `ABSTAIN`，不得伪装成模型 No-Go。
