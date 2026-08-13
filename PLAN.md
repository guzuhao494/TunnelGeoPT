# TunnelGeoPT 辅助开发计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-13
- Verification Status: B-elastic GO; analytic Stress-Lift smoke NO-GO
- Version Label: milestone_v0.2

## 0. 当前执行节点：B-elastic-v0.2 里程碑

- Run ID: `b-elastic-v0.2.0`
- Tier: B 层求解器 `main/test`；对整个岩爆研究仍属于 `auxiliary/dev`
- Selected idea: 先把 GeoPT 的几何先验接到一个可独立验证的二维平面应变开挖卸载求解器，再讨论预训练迁移。求解器采用拉应力为正的增量形式：远场增量位移为零、洞壁施加 `-Sigma_inf*n`，最终总应力为 `Sigma_inf + delta_sigma`。
- Baseline/oracle: 无孔均匀应变 patch test 与现有圆洞 Kirsch 解析解；二者都是只读验证基准，不是学习结果。
- Dataset boundary: 本节点只产生位移、应变、应力、平面应变轴向应力、应变能及静力诊断；`damage/velocity/dissipation/AE/rockburst` 均为不适用，不允许填零冒充标签。

### 0.1 冻结验收条件

1. `case_group_id` 在网格、保真度和求解重启之间保持不变；三类断面按 largest-remainder 规则做确定性 70/15/15 拆分，小样本每断面 6 个 case 必须得到 4/1/1。
2. Gmsh 网格显式区分 `wall/farfield`；无倒置单元、洞内无单元质心，网格和结果均有内容哈希。
3. P1 平面应变装配通过均匀场 patch、矩阵对称、自由自由度残差和 Clapeyron 能量检查。
4. 圆洞在冻结环带内与 Kirsch 三个应力分量比较；加密必须改善误差，细网格门槛见 `configs/elastic_milestone.json`，不得看结果后放宽。
5. 圆形、马蹄形、直墙拱形各至少完成一个端到端求解；随后生成冻结的 18-case B 层 smoke，并记录全部失败而不是补生成到好看。
6. 只有上述条件通过后才进入同一骨干的 Scratch/Static/Random/Stress-Lift 代理模型比较；该比较只允许声称合成弹性求解器仿真，不允许声称岩爆机理或现场有效。

### 0.2 最小代码变更图

| 路径 | 作用 | 本节点变更 |
|---|---|---|
| `src/tunnelgeopt/cases.py` | 原子 case 与 split | canonical hash、分层冻结 manifest、防泄漏断言 |
| `src/tunnelgeopt/mesh.py` | 带孔域网格 | Gmsh 生成、边界标签、网格 QC |
| `src/tunnelgeopt/elasticity.py` | B 层求解 | P1 平面应变增量求解、场恢复、物理诊断 |
| `src/tunnelgeopt/elastic_schema.py` | 独立 B schema | float64 计算记录、原子保存/加载、结果 hash |
| `scripts/run_elastic_milestone.py` | 真实运行入口 | patch、Kirsch、三断面、18-case smoke 与 manifest |
| `artifacts/experiment/b-elastic-v0.2.0/` | 证据 | 配置、命令、环境、原始指标、判定与失败清单 |

### 0.3 停止与转向条件

- 若洞壁法向、符号或边界组无法被独立测试确认，停止生成数据并修复网格/载荷接口。
- 若圆洞误差不随加密改善，结论是实现或边界截断有误，不能把细网格输出命名为高保真。
- 若静力求解器通过但迁移学习不优于 Scratch，保留 B 层数据与 No-Go 结果，转向机理求解器闭合项，而不是放宽门槛。

### 0.4 已执行结果（2026-08-13）

- `b-elastic-v0.2.0`：**GO**。18/18 三断面 case 通过；Kirsch 三载荷均随网格加密单调改善，fine 环带误差为 0.04184/0.02858/0.04425。
- `analytic-transfer-v0.2.0`：**NO-GO**。Stress-Lift@80% 相对 Scratch@100% 只通过 1/3 种子，且平均误差高于 Random-Lift；Shuffled 对照取得最低三种子均值。
- 决策：冻结 v0.2 负结果，不扩大当前 vector-distance/sticking 预任务；v0.3 转向跨几何、粗到细 B-elastic 场与残差学习。详见 `docs/MILESTONE_V0.2.md`。

## 1. 状态与边界

本计划用于验证“硬岩隧洞版 GeoPT”是否值得进入主实验，不是主实验方案，也不包含任何已取得的模型效果。当前没有真实工程数据；文中的“低保真/高保真样本”都指合成求解器输出，高保真不等于真实世界真值。

本阶段只回答一个开发问题：动态提升式预训练是否能在严格 case 级隔离下，减少学习合成高保真卸荷响应所需的标签数量。即使通过，也只能进入更严格的跨求解器、实验室和工程验证，不能据此宣称模型已经学会岩爆机理或具备现场预警能力。

当前仓库已经实现 A 层几何-边界 lifted 样本、B 层二维平面应变求解器、独立弹性 schema、冻结 case/split 和首个解析迁移结果。尚未实现高保真破裂求解器、损伤/断裂标签或现场迁移；本计划中 C 层字段和正式 pilot 规模仍是下一步接口与门禁，不能当作现有能力。

## 2. 预注册科学问题与假设

### RQ

在圆形、马蹄形和直墙拱形三类硬岩隧洞断面上，给定无量纲化的断面几何、原岩应力、材料/节理参数及开挖卸荷条件，使用训练集专属的 Stress-Lift 合成动态提升任务进行预训练，能否使同架构下游代理模型用不超过从头训练基线 80% 的高保真标注 case，达到从头训练基线使用 100% 高保真标注 case 时的锁定测试性能，同时满足预注册物理质控？

### H0

在 case 级隔离、相同模型容量、优化预算和高保真测试集下，动态提升式预训练不能实现至少 20% 的高保真训练 case 节省，或虽达到误差门槛但违反物理质控、仅由数据泄漏/求解器伪特征造成。

### H1

动态提升式预训练使用不超过 80% 的高保真训练 case 时，其预注册主误差相对“从头训练 + 100% 高保真 case”的误差比上侧 95% 置信界不超过 1.02，并同时通过物理质控、形状分层和负对照检查。

“至少 20% 高保真样本节省”是进入下一阶段的预注册 Go 门槛，不是当前结果或事实陈述。

## 3. 首版任务定义

### 3.1 输入状态与条件

- 几何：断面类型及其无量纲形状参数、计算域边界距离/特征半径。
- 初始状态：初始应力张量、位移/速度初值、材料场、损伤场和节理实现。
- 动力学条件：卸荷比例、卸荷时间、开挖进尺及主应力方向。
- 材料：`E/UCS`、泊松比、`ft/UCS`、`Gc/(UCS*R)`、非均质变异系数和相关长度/特征半径。
- 节理：无节理或 1-2 组节理；强度、刚度、迹长、方向和强度均用无量纲量表达。

`R` 为仓库几何对象记录的 `characteristic_radius`，`UCS` 为参考单轴抗压强度，`cp` 为参考纵波速度。完整首版范围见 `configs/pilot.json`。

### 3.2 动作/边界条件

本阶段不训练自主控制策略。“动作”只指求解器条件：分步移除洞壁支护压力/等效边界牵引，按给定无量纲卸荷时间推进，并保持远场应力边界。任何后续开挖策略优化必须另立 RQ，不能与本阶段数据效率结论混合。

### 3.3 拟议下游输出（当前未实现）

- 场输出：`u/R`、`v/cp`、`sigma/UCS`、损伤变量、应变能密度/UCS、耗散能增量。
- 事件输出：峰值响应时刻、损伤局部化区、总释放能和洞壁动能代理。
- 诊断输出：平衡残差、边界牵引残差、能量闭合误差、损伤不可逆违例率。

AE/微震只允许作为由损伤增量构造的“合成代理量”报告，不能标成真实 AE 或微震观测。

## 4. 模型与基线

所有学习模型必须共享相同下游骨干、参数规模、输入字段、优化器搜索空间、训练步数上限和 early-stopping 规则。

1. `B0 Solver reference`：锁定高保真求解器标签；不是学习基线，也不是工程真值。
2. `B1 Scratch`：同骨干从随机初始化训练；主比较基线。
3. `B2 Static-geometry pretrain`：仅预测 SDF/向量距离；用于检验静态几何预训练是否负迁移。
4. `B3 Random lift`：使用仓库当前实现的随机方向 GeoPT-compatible lifted 任务；隔离 Stress-Lift 条件化本身的作用。
5. `B4 Shuffled Stress-Lift negative control`：在断面类别内打乱应力/卸荷方向后预训练；若仍与正确条件等效，不能解释为学到几何-动力学耦合。
6. `TunnelGeoPT / Stress-Lift`：训练集专属几何和应力对齐动态提升预训练，再以不同高保真标签比例微调。

可选的 multi-fidelity warm start 只能作为补充消融，不能替代 `Random lift` 主基线。

## 5. 数据生成与分层保真

### 5.1 两档开发规模

| 档位 | 基础 case | 低保真 | 高保真 | 每 case 动态提升样本 | 用途 |
|---|---:|---:|---:|---:|---|
| small smoke | 18（每断面 6） | 18 | 6（每断面 2） | 8 | 跑通生成、QC、split、训练和评估；不做效果结论 |
| pilot | 384（每断面 128） | 384 | 144（每断面 48） | 32 | 辅助 Go/No-Go 筛查；仍非主实验 |

资源数仅为待实测的容量预估：smoke 约 2-50 CPU core-hours、0.2-2 GPU-hours、1-20 GB；pilot 约 500-5000 CPU core-hours、12-96 GPU-hours、100-1500 GB。首批 smoke 必须记录实际单 case 时间和存储，再据此重估 pilot；不得把这些区间当作已验证成本。

### 5.2 保真定义

- 低保真：较粗网格/时间离散和较松求解容差，用于预训练与开发，不作为最终测试标签。
- 高保真：更细网格/时间离散、严格残差与能量闭合；所有锁定测试指标只在通过 QC 的高保真 case 上计算。
- 至少 10% 高保真 case 做再加密网格审计；若关键标量变化超过 5%，该求解器设置不能称为本项目内的“高保真”。
- 若条件允许，pilot 的一小部分 case 应交由独立求解器复算；同一求解器的细网格只能降低离散误差，不能消除 solver bias。

## 6. 采样与 case 级拆分

### 6.1 采样

- 三类断面等额分层。
- 连续变量使用带 scramble 的 Sobol 序列；跨数量级参数在对数空间采样。
- 节理参数按 `joint_family_count` 条件采样，无节理 case 不生成无意义的节理随机数作为模型特征。
- 在正式生成前做约束过滤，拒绝自交断面、洞周边界过近、节理几何不可构造和无效材料组合。
- 不按求解结果回填或过采样“好看”的破坏 case；若需要响应分层，规则必须只用训练集并另行记录抽样权重。

### 6.2 原子 case 定义

`case_group_id` 由以下内容的规范化表示哈希得到：父几何参数、材料实现种子、节理实现种子、初始应力、主方向、开挖/卸荷日程和物理参数。下列派生项必须继承同一个 split：

- 同一 case 的所有时间步/场快照；
- 不同网格、时间步长、求解重启和低/高保真版本；
- 旋转、镜像、重采样、点云化和裁剪等增强版本；
- 同一母体随机场或节理网络的不同观测窗口。

先生成并冻结 `case_group_id -> split` 清单，再生成任何轨迹或标签。采用按断面分层的 70% train / 15% dev / 15% locked test。预训练只可访问 train 组；dev 用于选超参；locked test 在 Go/No-Go 前保持封存。

## 7. 防泄漏要求

- 禁止按节点、网格单元、快照或时间步随机拆分。
- 禁止让同一 case 的低保真进入训练、高保真进入测试。
- 禁止预训练接触 dev/test 断面几何、动态轨迹、材料/节理种子或归一化统计量。
- 归一化器、PCA/编码器、插值基、采样权重和缺失值规则只能在 train 上拟合。
- 内容哈希和近重复检索同时执行；参数哈希不同但几何/随机场近重复的 case 仍须并组。
- 文件名、目录、求解器耗时、网格编号和收敛轮数不作为模型输入，且应做可预测性探针，防止标签被元数据泄漏。
- test 结果只运行一次主门槛评估；任何门槛修改都必须生成新版本并声明原 test 已经被看过。

## 8. 训练与验证矩阵

| 因子 | 水平 |
|---|---|
| 初始化 | Scratch / Static / Random lift / Shuffled Stress-Lift / TunnelGeoPT Stress-Lift |
| 高保真训练比例 | 40% / 60% / 80% / 100%（均从同一 train case 嵌套抽取） |
| 随机种子 | pilot 至少 5 个 |
| 测试 | 锁定 IID case；按断面分层；可选参数边界 OOD 挑战集 |
| 保真 | 训练可混合；测试仅 QC 合格高保真 |

主比较预先锁定为 `TunnelGeoPT@80%` 对 `Scratch@100%`。其他比例用于估计标签-误差曲线，不可事后挑选最有利比例替代主比较。

## 9. Go/No-Go 概览

必须同时满足：

1. 数据与求解器 QC 达标，无 critical leakage；
2. `TunnelGeoPT@80%` 对 `Scratch@100%` 的主误差比上侧 95% 置信界不超过 1.02；
3. 至少 4/5 训练种子通过、至少 2/3 断面不劣，且任何单一断面误差恶化不超过 5%；
4. 动态提升在同标签预算下方向性优于静态几何预训练，且 shuffled-dynamics 负对照不能复现收益；
5. 预测物理 QC 达标，收益不是以更大的能量/平衡/不可逆违例换取。

任何 critical leakage、锁定 test 被用于调参、求解器 QC 失败或主门槛失败均为 No-Go。详见 `docs/VALIDATION.md`。

## 10. 通过后仍需补的证据

辅助 pilot 通过后，仅允许进入：跨求解器复算、几何/应力/材料外推、实验室真三轴/卸荷数据校准，以及最终的工程 shadow test。没有这些证据时，项目结论只能是“在特定合成分布上观察到预训练数据效率候选信号”。

## 11. v0.3 多保真残差路线（当前执行锚点）

v0.2 的圆洞 Stress-Lift 已按预注册门槛判为 No-Go，因此本节取代第 8–9 节作为当前实验锚点；旧 locked test 永久退出新模型选择和新结论。

新的唯一科学问题是：在二维均质各向同性小应变平面应变问题中，测试时允许运行一次粗网格 FEM，`coarse field + learned residual` 能否只用 50% 的 fine-FEM 训练父几何，在全新父几何和预定义载荷/几何 OOD 上达到 `Scratch@100%` 与 `Direct+Coarse@100%` 的非劣性能，同时将 raw coarse 误差至少降低 30%。

执行顺序固定为：

1. 增加真正改变归一化边界的连续宏观断面参数，冻结精确边界 hash；
2. 以 `geometry_group_id` 为最小 split 单位，使同一父几何的全部载荷、网格和查询点继承同一 split；
3. 同一条边界只改变网格尺寸，在独立公共查询点分别定位 coarse/fine 单元，不以单元中心最近邻配对；
4. smoke 只验证接线，不生成正式结论；其后以独立 dev-only 数据完成 fine-ultrafine 收敛和优化超参数校准；该门禁已通过且 locked 读取为 0；
5. `configs/multifidelity_formal.json` 已转为 `frozen_preregistered_pre_generation`，允许生成 formal 数据，但仍保持 `formal_data_generated=false`、效果未计算、locked label 未打开；
6. 正式 checkpoint 固定为 Scratch100、Direct100、Residual25/50/75/100、Mismatched50 × 5 seeds，共 35 个；每个 checkpoint 必须绑定 `TrainingContract` 并进入 `CheckpointRegistry`；
7. locked fine label 使用独立文件级 store，训练进程不得收到其路径；全部 checkpoint 冻结并哈希后才由独立 evaluator 解锁，并对每个 locked 分区最多评价一次；
8. `0.02R` wall-offset query 只计算相对 fine 的牵引/合力差异，不冒充精确洞壁 traction-free 或全域平衡残差；
9. solver 代数/能量残差、三角形面积/质量、每分区每族 95% 有效率和 fine-ultrafine `3%/5%/4%` 门槛必须先通过，否则 ABSTAIN；
10. 正式结果通过五种子、三类断面、IID/geometry-OOD/load-OOD、fine-ultrafine 与泄漏审计后，才允许称为“合成线弹性多保真标签效率成果”。

不论结果如何，本阶段都不声称已学习破裂、损伤、岩爆、微观到现场迁移或工程预警。

### 11.1 v0.3 正式结果（2026-08-13）

- 五阶段正式运行已在 clean/pushed implementation HEAD
  `0f4cc0d504b35092928eb33e43bbbca0d213b545` 完成；705/705 case、35/35
  checkpoint、140/140 锁定分区评估以及全部数值/访问合同通过。
- 最终分类为 **ABSTAIN**，而不是 No-Go：IID 与 geometry-OOD 的
  `Residual50/Scratch100` 双侧 95% 区间宽度分别为 `0.131422` 和
  `0.112845`，超过预注册最大宽度 `0.10`。有效性失败优先于效应失败。
- 诊断上，Residual50 在 IID/geometry-OOD 未证明 50% fine-label 效率；在
  load/joint OOD 显著优于学习基线但仍劣于 raw coarse，并且 load-OOD
  wall-offset 门失败。Mismatch50 在全部分区最差，说明正确 coarse-fine
  配对确有信息，但不足以支持主张。
- 当前 705 个 case 的全部身份永久记为 seen。下一次确认性运行必须新版本、
  新 salt、新身份并保持原科学阈值；先依据本次父几何层级方差做功效设计。
- 公开结果与 SHA-256 清单见
  `artifacts/experiment/mf-residual-formal-v0.3.0/RESULT_SUMMARY.md`。

## 12. v0.4/v0.5 数值算子转向（2026-08-13）

- [x] 结构化神经残差原型未达到新 locked 数据启动门槛；生产 cross-fit 保持
  禁用，未生成新 locked case，探索记录标记为不可重放的会话迁移证据。
- [x] 实现九通道线性载荷响应基，并在 120 个 seen 父几何、480 个逐载荷
  leave-one-load-out 上验证；生产设计固定三个条件数为 `sqrt(2)` 的
  张量范数单位规范载荷。
- [x] 实现确定性 P1 节点应力恢复，并在 15 个 seen case 的
  coarse/fine/ultrafine 真实求解中确认近场误差下降。
- [x] 原恢复的 wall-offset 牵引/合力门失败，冻结为负面边界，不启动新的
  unseen/formal 实验。
- [x] 完成边界兼容切向增量投影的独立 seen-development 重设计验证；牵引/合力
  非恶化通过，但 wall-offset 全应力仍恶化，冻结为第二个 STOP。
- [x] 在 clean/pushed 实现 HEAD `44d244e` 上，用三个相对冻结 v0.2/v0.3 排除源
  为新身份的几何完成 `24/24` 次 direct FEM：15 个 held-out 面内总应力重建
  RelL2 中位数/最大值为 `4.886e-15/5.882e-15`，17 个门全部通过，分类为
  `LINEAR_ELASTIC_LOAD_AXIS_FACTORIZATION_CONFIRMED`。该确认仍只适用于每个
  固定几何/材料/网格/查询系统内的二维小应变线弹性层。
- [ ] 恢复路线需在新 development screen 上解决切向应力偏差后才可冻结；载荷基
  已独立完成新几何 direct-FEM 确认。破裂/损伤仍由未来 C 层求解器或实验数据负责。

完整指标与证据边界见 `docs/MILESTONE_V05_NUMERICAL_OPERATORS.md`。

## 13. 论文主线：C-fracture 开发门禁（2026-08-14）

当前论文问题已从“线弹性载荷基本身是否成立”转为：在二维准静态
脆性相场断裂的合成硬岩隧洞范围内，将每个断面的三载荷线弹性响应基作为
固定物理骨架、仅学习不可逆损伤与应力残差，能否用 `50%` 断裂轨迹达到
同架构 `Scratch100`。

当前只允许执行 solver/schema/QC 的 development pilot：

1. 主求解器候选为 MOOSE 相场断裂；本地 scikit-fem 实现只能作交叉验证原型；
2. 先通过完好弹性、MOOSE `crack2d_iso`、Miehe 拉伸/剪切、三网格、能量与不可逆门禁；
3. 然后生成 `3 断面 x 3 材料档 x 4 载荷路径 = 36` 条 development-only 轨迹；
4. 其中 `12` 条按断面×路径平衡做 ultrafine 复算；36 个冻结 cell 不得静默替换；
5. Phase-1 不切 train/dev/test、不训练模型、不声称标签效率；
6. 只有 solver 验证与 36 轨迹资源/方差审计通过，才允许冻结 EBR-DNO 开发训练合同；
7. 只有 development 上 `EBR-DNO50` 相对 `Scratch100` 有稳定趋势且物理门不恶化，
   才可新建 formal locked 身份。

详细 RQ、反证、基线、统计与论文边界见 `paper/RESEARCH_SCOPE.md` 和
`paper/PAPER_OUTLINE.md`。这一阶段仍不是动力岩爆、微震或现场验证。
