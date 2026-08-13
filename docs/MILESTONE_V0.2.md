# TunnelGeoPT v0.2 里程碑报告

日期：2026-08-13
判定：**B-elastic 数据层 GO；Stress-Lift 解析迁移筛查 NO-GO**

## 1. 本轮真正完成了什么

本轮把项目从 GeoPT 兼容的几何轨迹生成器推进到了一个可以独立复算的二维隧洞线弹性数据层，并第一次用预注册的正、负对照检验了 Stress-Lift 是否能减少高保真标签。两个结论必须分开：

1. `B-elastic-v0.2.0` 通过，可以用于生成**均质、各向同性、小应变、平面应变线弹性**求解器样本。
2. `analytic-transfer-v0.2.0` 未通过，当前 Stress-Lift 不能支持“减少 20% 高保真标签”的主张。

这不是岩体破裂、动态岩爆、实验室或现场有效性证据。

## 2. B-elastic：GO

正式运行在圆形、马蹄形和直墙拱形三类断面上冻结了 18 个 case。所有 case 均只尝试一次，无失败替换；18/18 通过网格、静力、能量、schema 保存及重载检查。42 个清单文件的 SHA-256、大小和存在性均已重新核验，18 条记录也再次完成语义加载。

| 验证项 | 结果 | 冻结门槛 |
|---|---:|---:|
| 仿射 patch stress relative L2 | `1.704e-14` | 通过 |
| Kirsch 单轴 fine annulus L2 | `0.04184` | `< 0.08` |
| Kirsch 等双轴 fine annulus L2 | `0.02858` | `< 0.08` |
| Kirsch 纯剪 fine annulus L2 | `0.04425` | `< 0.08` |
| 单轴峰值环向应力误差 | `0.00453` | `< 0.10` |
| 三载荷网格收敛 | coarse → medium → fine 均改善 | 必须单调改善 |
| 18-case 通用 QC + schema | `18/18` | 全部通过 |

因此，本轮只授权进入“扩大线弹性求解器样本、训练求解器代理模型”的阶段，不授权生成损伤、裂纹、耗散、AE 或岩爆标签。

## 3. Stress-Lift 解析迁移：NO-GO

冻结任务包含 240 个圆洞 Kirsch 载荷 case、每 case 512 个点，按完整 `case_group_id` 拆为 168/36/36 train/dev/locked-test。六种方法使用同一 113,571 参数 DeepSets 骨干和相同下游优化规则；`*_80` 使用 134 个完整训练 case，`scratch_100` 使用 168 个。三种子为 17、29、43。

主指标越低越好。下表为锁定测试集逐 case 聚合后，三个种子的结果及均值：

| 方法 | seed 17 | seed 29 | seed 43 | 三种子均值 |
|---|---:|---:|---:|---:|
| Scratch@80% | 0.016238 | 0.017375 | 0.016198 | 0.016604 |
| Scratch@100% | 0.016917 | 0.019871 | 0.012685 | 0.016491 |
| Static@80% | 0.021201 | 0.014014 | 0.024812 | 0.020009 |
| Random-Lift@80% | 0.020705 | 0.016186 | 0.013428 | 0.016773 |
| Shuffled-Stress-Lift@80% | 0.019445 | 0.013407 | 0.014981 | **0.015944** |
| Stress-Lift@80% | 0.019725 | 0.012677 | 0.020524 | 0.017642 |

预注册主比较 `Stress-Lift@80% / Scratch@100%` 的分层、配对 case bootstrap 为：

| seed | 中心误差比 | 95% CI | 是否通过 `upper <= 1.05` |
|---|---:|---:|---|
| 17 | 1.166 | [1.119, 1.210] | 否 |
| 29 | 0.638 | [0.612, 0.662] | 是 |
| 43 | 1.618 | [1.557, 1.679] | 否 |

只通过 1/3 种子，低于 2/3 门槛；Stress-Lift 的平均误差也高于 Random-Lift。Shuffled 对照在 seed 29 同样通过，并取得最低的三种子平均误差，所以不能把 seed 29 的收益解释为模型稳定学到了正确的应力—边界耦合。壁面牵引与远场误差相对 Scratch@100% 仅增加 0.00075 和 0.00243，物理违约门槛通过，但这不能抵消主效果失败。

## 4. 完整性审计

- 18/18 checkpoint 均为本轮新训练，文件存在，SHA-256 匹配且互不重复，全部权重有限。
- train/dev/locked-test case 数为 168/36/36，三个集合的 `case_group_id` 交集为零。
- checkpoint 全部冻结前，locked-test 标签物化数和读取数均为零；之后才生成 36 个测试标签。
- 实际记录为 18 次评测调用、54 次完整测试前向、270 个 forward batch、648 次 case-label 读取。
- 六方法 × 三种子的逐 case 与均值指标全部有限；重新执行冻结 gate 得到相同 `NO-GO`。
- 二进制 `.pt` checkpoint 和 `.npz` 场数组保留在本机生成目录并由清单哈希约束，默认不进入 Git；代码、配置、JSON 指标和清单进入版本库。

非阻断限制：checkpoint 恢复目前校验 method、seed 和配置哈希，但尚未把源代码哈希写入恢复身份；预训练方法还额外使用 200 轮廉价预训练，因此本实验比较的是高保真**标签效率**，不是总计算效率。

## 5. 为什么不继续放大当前方案

现有圆洞任务已经接近数据饱和。事后诊断显示 Scratch@80% 的三种子平均误差只比 Scratch@100% 高 `0.68%`，标签减少 20% 几乎没有形成可分辨难度。与此同时，正确 Stress-Lift 的种子方差大，打乱条件的模型反而取得最低均值。这说明当前 vector-distance/sticking 预任务更像不稳定的通用初始化，而不是可信的岩石力学桥梁。

因此不扩大当前圆洞 Stress-Lift 到正式 pilot，也不放宽门槛。

## 6. v0.3 转向

下一阶段改为“低保真机理求解器 + 学习残差/闭合项”，而不是继续堆随机几何轨迹：

1. 用 B-elastic 粗网格场作为廉价输入或预训练标签，用细网格场作为高保真目标，直接学习应力、位移、能量或 coarse→fine 残差。
2. 引入圆形、马蹄形、直墙拱形的连续几何变化，使用 leave-one-geometry-family-out 和 leave-one-load-region-out；新建测试版本，当前测试集不再用于调参。
3. 先冻结 Scratch@40/60/80/100% 标签—误差曲线和“任务未饱和”门槛，再检验预训练数据效率。
4. 正对照改成应力/牵引/能量监督，继续保留 Random、Shuffled 和 Static；必要时比较全量微调、冻结骨干和分层学习率，检验是否存在灾难性遗忘。
5. 只有跨几何、跨网格的线弹性迁移稳定后，才接入相场/FDEM/DEM 的损伤张量、断裂能、过程区尺度和耗散等桥变量；仍需实验室与工程数据校准。

## 7. 可复算入口

- B-elastic 配置：`configs/elastic_milestone.json`
- B-elastic 运行：`scripts/run_elastic_milestone.py`
- B-elastic 判定：`artifacts/experiment/b-elastic-v0.2.0/decision.json`
- 解析迁移配置：`configs/analytic_transfer_smoke.json`
- 解析迁移运行：`scripts/run_analytic_transfer.py`
- 解析迁移判定：`artifacts/experiment/analytic-transfer-v0.2.0/gate.json`
- 完整锁测指标：`artifacts/experiment/analytic-transfer-v0.2.0/locked_test_results.json`
