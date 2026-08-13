# TunnelGeoPT 辅助开发检查表

> 本表是执行门禁，不代表任何项目结果。`至少 20% 高保真样本节省`仅为预注册 Go 门槛。

## 当前 B-elastic-v0.2 执行门禁

- [x] B 层物理问题、增量边界条件、符号约定和 Kirsch/patch oracle 已在运行前冻结。
- [x] 项目专属 Python 3.12 科学环境完成 SciPy、scikit-fem、Gmsh、PyTorch 的 import 与最小计算检查。
- [x] canonical case、largest-remainder 分层 split 和 frozen manifest 通过测试。
- [x] Gmsh `wall/farfield` 标签、洞壁法向、网格有限性与质量检查通过。
- [x] 平面应变 P1 patch、矩阵对称、代数残差与能量闭合检查通过。
- [x] 圆洞 Kirsch 多网格验证达到冻结门槛，且误差随加密改善。
- [x] 三类断面端到端求解与 18-case B 层 smoke 完成，全部失败原因进入 manifest。
- [x] 独立 B 层 schema round-trip、内容 hash、单位和分量次序通过测试。
- [x] 首个代理模型可比基线与负对照完成；结果按 case 而非节点聚合（正式结果为 No-Go）。
- [x] 所有证据、代码和文档提交推送，远端 Python 3.11/3.12 与 CPU-PyTorch CI 通过。

## 已完成的 A 层软件 smoke（不构成模型效果证据）

- [x] Windows环境、WSL2、Python、Git与两侧GPU可见性已写入 `validation/environment/`。
- [x] 三类断面 × Random/Stress-Lift 共6组小样本完成生成、保存、加载和哈希记录。
- [x] GeoPT兼容形状、dtype、有限性、表面固定点和输入/监督符号约定通过测试。
- [x] Kirsch圆洞自由面牵引与单轴应力集中系数3通过解析不变量检查。
- [x] Windows侧单元测试与静态检查通过；WSL依赖安装因PyPI超时单独记为blocked。

## A. 预注册与配置冻结

- [x] RQ、H0、H1 与主比较已冻结：`TunnelGeoPT@80% HF` vs `Scratch@100% HF`。
- [x] 主误差、物理 QC、置信区间方法和非劣界 `error ratio <= 1.02` 已冻结。
- [x] `configs/pilot.json` 能由严格 JSON 解析器读取，且记录 schema/version/config hash。
- [x] 明确标注本阶段为 auxiliary/dev，而非主实验或工程验证。
- [x] 明确标注高保真为求解器标签，而非真实世界真值。
- [x] smoke 已记录实际墙钟时间和生成数据量；pilot 的时间/费用仍只是待标定估计。

## B. case 身份与拆分

- [x] `case_group_id` 在任何求解/轨迹生成前由规范化物理参数与随机种子生成。
- [ ] 三类断面按 case 等额分层。
- [x] train/dev/locked-test 比例为 70/15/15，清单已冻结并保存 hash。
- [x] 同 case 的全部时间步、节点、网格、低/高保真、重启和增强均在同一 split。
- [ ] 同一材料随机场母体和同一节理网络的派生 case 已并组。
- [x] 解析 smoke 的预训练只读取 train case 的载荷与轨迹；该任务全为同一圆洞，不作未见几何主张。
- [ ] 所有归一化、编码、插值基和采样权重只在 train 上拟合。
- [x] locked test 未参与超参、阈值、早停、模型选择或错误排查。

## C. 防泄漏审计

- [x] 无节点级、单元级、快照级或时间步级随机拆分。
- [x] 精确内容 hash 在 split 间无重复。
- [ ] 几何距离/参数距离/随机场相似度的近重复扫描通过。
- [x] 文件名、目录、solver ID、网格 ID、耗时、迭代数和状态码未进入模型特征。
- [ ] 元数据-only 探针接近机会水平；否则暂停并追踪泄漏源。
- [ ] 标签置乱测试回到机会/无技能水平。
- [ ] shuffled-dynamics 负对照不能复现正确动态预训练的收益。
- [x] 训练日志记录实际读取的 `case_group_id`，与 frozen manifest 一致。

## D. 几何与参数 QC

- [ ] 三类断面均非自交、法向一致、洞周闭合。
- [ ] 外边界距离满足配置下限，断面不触碰计算域。
- [ ] 条件参数有效：无节理 case 不携带可泄漏的节理占位随机数。
- [ ] 应力张量主值有序且满足配置比值；方向变换后不改变本征值。
- [ ] 材料刚度正定，`0 < nu < 0.5`，强度/断裂能为正。
- [ ] 网格无倒置单元，scaled Jacobian 与宽高比达到求解器门槛。
- [ ] 所有无量纲量能通过 round-trip 恢复，单位元数据没有混用。

## E. 求解器物理 QC

- [ ] 无 NaN/Inf；字段形状、分量次序和时间顺序一致。
- [ ] 初始平衡残差、终态平衡残差、边界牵引残差通过门槛。
- [ ] 相对能量闭合误差不超过配置门槛。
- [ ] 损伤不可逆违例率与负耗散违例率不超过配置门槛。
- [ ] 节理接触不存在超容差穿透，开闭状态与法向约定一致。
- [ ] 对称输入的对称性误差通过；旋转后的张量输出满足等变检查。
- [ ] 至少 10% 高保真 case 完成更细网格审计，关键标量变化不超过 5%。
- [ ] 每断面 QC 通过率达到 95%；失败 case 不被静默替换。
- [ ] 所有失败原因、重跑次数和筛除规则进入 manifest。

## F. 训练公平性

- [x] 所有初始化方法使用同一骨干、参数量、输入字段和输出头。
- [x] 下游优化器、最大步数、早停和数据增强规则一致；额外预训练算力单独披露。
- [ ] HF 40/60/80/100% 子集按 `case_group_id` 嵌套，非重新随机抽样。
- [ ] pilot 每个主方法至少 5 个训练种子。
- [x] Scratch、Static、Random lift、Shuffled Stress-Lift 与 TunnelGeoPT Stress-Lift 均成功运行。
- [x] 不以更多下游训练步数、更多 HF case 或测试集调参补偿某个基线。
- [x] 训练失败按预注册规则报告，不静默重启到“好种子”。

## G. 锁定验证

- [x] test 推理只读取冻结 checkpoint 和 test manifest。
- [x] 主误差按 case 先聚合，再跨 case 统计，避免大网格 case 权重更高。
- [x] 置信区间按 `case_group_id` 分层 bootstrap，不按节点 bootstrap。
- [ ] 报告 5 个种子的全部结果、中心估计、95% CI 和失败运行。
- [ ] 同时报三类断面结果，不能只报总体均值。
- [x] 报告预测物理 QC，不只报 relative L2。
- [x] 报告 `TunnelGeoPT@80%` vs `Scratch@100%`，不事后换主比较。
- [x] “20% 节省”仅在全部 Go 条件通过后表述；本次 No-Go 未作该主张。

## H. Stop / No-Go

- [ ] 一旦发现跨 split case 泄漏，停止效果评估并重建数据版本。
- [ ] 一旦 locked test 被用于调参，废弃其“锁定”身份并创建新测试版本。
- [ ] 高保真网格审计失败时，不继续做数据效率主张。
- [x] 若 `TunnelGeoPT@80%` 未达到主门槛，记录 No-Go，不增加事后容差。
- [x] 若收益由 shuffled-dynamics、元数据探针或 solver ID 同样复现，判定机理解释 No-Go。
- [x] 通过辅助 pilot 后仍不声称现场岩爆预警有效；先进入跨求解器/实验室验证。
