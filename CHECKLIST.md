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

## I. v0.3 多保真残差执行门禁

- [ ] 三类断面均有至少两个改变归一化边界的连续宏观参数；smooth circle 只作校准。
- [ ] `geometry_group_id/load_group_id/case_group_id` 三级身份、boundary hash 与父几何 split 继承通过。
- [ ] coarse/fine 使用完全相同边界与外域，只改变冻结的网格尺寸。
- [ ] 公共 query 在求解前确定；粗细场均通过 point-in-triangle 独立定位和重心坐标复验。
- [ ] stress scale 只来自已知远场输入；任何归一化都不读取 fine/test 标签。
- [ ] smoke 完成且只用于接线；formal config 在任何正式效果计算前冻结并提交。
- [x] 独立 dev-only fine-ultrafine 收敛审计通过：24 case median/p95 为 `2.116%/3.036%`，三族 median 均 `<=2.564%`，locked 读取为 0。
- [x] 优化上限仅据独立 train/dev 轨迹冻结为 max 300 epochs、patience 35；其他优化字段与全部效果门槛未改变，formal/locked 读取为 0。
- [x] `configs/multifidelity_formal.json` 已从 candidate 更新为 `frozen_preregistered_pre_generation`；允许生成但当前仍为 `formal_data_generated=false`，未误写成已运行。
- [ ] `Scratch`、`Direct+Coarse`、`Residual+Coarse` 与 `Mismatched-Coarse` 使用公平骨干/预算。
- [ ] 25/50/75/100% 子集按父几何整组、分族平衡且严格嵌套。
- [ ] 方法矩阵严格为 Scratch100、Direct100、Residual25/50/75/100、Mismatched50 × 5 seeds，共 35 个 checkpoint。
- [ ] 每个 checkpoint 的实际训练行/比例/config hash 由 `TrainingContract` 绑定；全部 35 个在 test 解锁前原子落盘并进入带唯一 SHA-256 的 `CheckpointRegistry`。
- [ ] train/dev/locked-IID/geometry-OOD/load-OOD/joint-OOD 的 geometry/case/boundary/load hash 零交集。
- [ ] 至少 20% 预选 case 的 fine-ultrafine 审计通过；失败则 ABSTAIN 而非模型 No-Go。
- [ ] coarse/fine/ultrafine 均满足代数残差 `<=1e-9`、能量误差 `<=1e-9`、最小面积/R² `>=1e-12`、最小三角形质量 `>=0.02` 和全部 query 可定位。
- [ ] 每个“分区 × 断面族”有效 case 率 `>=95%`；失败 case 全部记录且未静默替换。
- [ ] locked fine label 位于独立文件级 store；训练进程没有其路径，Python 私有字段未被当作安全边界。
- [ ] `0.02R` wall-offset 只报告相对 fine 的牵引/合力差异；未声称绝对 traction-free 或全域平衡残差。
- [ ] Residual50 wall-offset 指标同时通过冻结的绝对 cap 与 `1.10 × raw coarse + margin` 非恶化门槛。
- [ ] locked-IID、geometry-OOD、load-OOD 仅评价一次，并报告所有种子、断面和失败切片。
- [ ] 主 gate 使用父几何与训练种子的层级配对 bootstrap，不按点 bootstrap。
- [ ] 只有全部预注册 gate 通过才表述“50% fine 训练标签非劣”；否则如实 No-Go/ABSTAIN。

### v0.3 formal 执行记录

- [x] 三族参数化边界、三级身份、公共 query 与 coarse/fine 边界一致性通过。
- [x] 35 个公平 checkpoint 与嵌套 25/50/75/100% 父几何子集完成并冻结。
- [x] 705/705 case 通过 solver/mesh QC；144-case fine-ultrafine 审计通过。
- [x] 训练进程未获 locked 路径；冻结前 locked read=0；4 个 sealed store 各开一次。
- [x] IID/geometry/load/joint 四分区共 140 次 checkpoint 评估恰好执行一次。
- [x] 最终判定按预注册优先级记录为 ABSTAIN，未放宽区间或效应门槛。
- [ ] 用全新身份完成有功效的 v0.4 复现；当前 v0.3 不作标签效率主张。

## J. v0.4/v0.5 转向执行记录

- [x] v0.4 结构化残差真实 cross-fit 未获授权；未用失败后加算力掩盖不利原型。
- [x] v0.4 参数量、loss 和 17/14 输入通道混杂已披露；消融不作因果解释。
- [x] 九通道载荷基单元契约、归一化不变性、秩与条件数失败检查通过。
- [x] v0.3 seen 数据上 120 parents / 480 held-out loads 的载荷基验证通过。
- [x] P1 应力恢复算子的常量/仿射/叠加/秩亏与严格输入测试通过。
- [x] 15-case coarse/fine/ultrafine 恢复开发运行完成，三断面近场均改善且 QC 通过。
- [x] 原恢复 wall-offset 物理门失败，已记录 STOP，不启动新 confirmatory 数据。
- [x] 边界兼容切向增量投影在同一 seen 选择上完成：近场与 traction/resultant
  通过，但 wall-offset 全应力门失败，因此未升级为 READY。
- [x] 载荷基在三个相对冻结 v0.2/v0.3 排除源为新身份的几何、三个规范基载荷和
  每几何五个 direct-FEM held-out 载荷上独立确认；24/24 求解、17/17 门通过。
- [x] 当前结果与文档继续保留“线弹性数值算子不等于岩爆/破裂模型”的边界；
  后续 C 层仍须单独的损伤/断裂求解器或实验标签和新验证协议。

## K. C-fracture 论文路线开发门禁

- [x] 完成现有 v0.2-v0.5 证据信任分级，明确当前为 `baseline_ready + analysis_ready`，
  不是 fracture-paper-ready。
- [x] 冻结一句 RQ、可证伪效果门、主指标、强基线和不可声称边界的
  Stage-1 研究范围候选。
- [x] 将线弹性载荷基降级为方法引理/固定骨架，不再当作论文核心创新。
- [x] 将旧 `configs/pilot.json` 判定为不可直接复用；新 Phase-1 删除动力、3D、
  节理、AE、随机非均质和旧 Stress-Lift 效果门。
- [x] 用户以持续论文目标确认 Stage-1 范围；已进入独立 C-fracture schema 与新 pilot
  config 的实现阶段，但尚未生成 36 条轨迹。
- [x] 实现独立不可变 fracture trajectory schema、36-case/12-audit 开发协议和本地
  P1 AT2 调试内核；专项测试通过，但三者尚未组成完整数据生成器。
- [x] 完成一次独立实现审计并判定当前 `NO-GO`：P2/P3/P4、裂纹带网格、最终状态
  一致性、累计功/反力、重试账本和外部交叉验证未闭合，禁止生成训练标签。
- [x] 完成配置到 `s, sigma_inf(s), wall_release_by_facet(s)` 的唯一 v3 加载适配器；
  已冻结 `(y,z)` 坐标、主应力角/符号/插值顺序和 P4 实际壁面 facet 分区，并支持
  41 个必需状态及自适应重试所需的任意 `s in [0,1]`。
- [x] 将 v3 加载状态接入 development-only 位移/损伤求解器；P1 与旧路径逐位回归，
  P2/P3 当前远场仿射场与 correction carry、P4 逐 facet 卸荷及 foreign-wall 拒绝均已测试。
- [x] 为 scheduled 求解增加 41 个必需输出、自适应减半、失败重试和全部接受子步账本。
- [x] 冻结独立反力/路径功原语：约束反力符号、壁面与远场梯形功、拒绝尝试回滚、
  瞬时 Neumann 泛函和带显式物理尺度下限的增量能量诊断均已有专项测试。
- [x] 让求解器在同一收敛 `(u,d)` 状态输出内力、壁面力、约束 DOF 与规定位移；
  trajectory schema v3 已从这些原始量重算反力、自由 DOF 残差、分项功、累计功和
  能量不平衡，并绑定完整几何/拓扑/DOF/加载身份。
- [x] 将 scheduled 求解结果适配并保存为 schema v3；保存前对每个接受步重放
   `Phase1LoadSchedule.state_at(s)` 并逐位核对 stress/facet order/release，再接入分项
   累计功、全局合力/力矩、减半重试账本和保存-重载-复验 E2E。
- [x] 完成可选裂纹带 Gmsh Distance/Threshold 背景场及生成后实际最大边长和
  `h/ell` 硬审计；默认 B-elastic 网格路径保持兼容，尚未运行协议规模网格。
- [x] 固定 MOOSE `crack2d_iso` reference self-test 在 WSL 中精确运行 1/1 通过，
  MOOSE/项目 HEAD、环境、输入、gold、runner、executable 与日志哈希均已落盘。
- [x] 本地相场原型与 MOOSE 在同一 TRI3 网格的完好弹性与固定非均匀 P1 损伤
  平衡状态上交验证，6/6 个基载荷均通过 `1e-6` 门。
- [x] 冻结 Miehe-type SENT/SENS development-only 协议；实现零厚度双裂面网格、
  `[y,z]` 位移控制 BVP、完整物理标签与现场哈希复验。六档真实 Gmsh 网格均通过
  拓扑和 `hmax<=1.15h` 审计，但没有运行耦合裂纹轨迹。
- [x] 实现最多 12 个显式状态的完整体 `d=0` 非授权探针；默认 validate-only，
  严格区分自由 DOF 残差、约束反力、合力/力矩和路径功，且不能授权正式计算。
- [x] 在同一 clean/pushed HEAD 下串行完成 SENT+SENS coarse fixed-`d=0` 三状态
  成对实跑；6/6 状态通过适用 QC，峰值 RSS 约 402 MiB。固定损伤完整 coarse
  轨迹耗时下界合计约 113 h，因此该结果只允许 timing/QC triage。
- [ ] 接通 checkpoint 的 `u/d/H` 单步重启并完成 SENT+SENS 各三状态的 bounded
  coupled coarse prefix；在此之前不得启动完整 201/151 状态轨迹。
- [ ] 完成预缺口拉伸/剪切 SENT/SENS 耦合裂纹演化三网格基准。
- [ ] 三网格、加载步、长度尺度、能量、残差与不可逆门全部通过。
- [ ] 完成 36 条 development-only 轨迹和 12 条 ultrafine 复算，无静默替换。
- [ ] 完成资源、方差和计划效能审计；不达标则 STOP，不创建 locked 数据。
- [ ] 只在 development launch 门通过后冻结 EBR-DNO 模型与 formal 240-parent 候选合同。
