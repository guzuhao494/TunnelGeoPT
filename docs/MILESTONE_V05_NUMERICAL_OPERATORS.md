# v0.5 数值算子里程碑：载荷基分解与应力恢复

日期：2026-08-13；独立载荷基确认更新：2026-08-14

本里程碑是在 v0.3 多保真正式实验 `ABSTAIN` 后进行的开发性转向。
v0.3 的 705 个 case 已全部看过，因此本页涉及这些标签的结果都属于
**seen-data development**，不构成新的独立验证或正式效果主张。

## 1. v0.4 结构化神经残差：停止扩大实验

结构化线性残差原型在五个种子上的 IID、geometry-shift、load-shift
平均误差约为 `3.171% / 3.341% / 3.722%`。相对 raw coarse 的点比率约为
`0.945 / 0.971 / 0.965`，远未达到启动新锁定数据前冻结的
`0.68 / 0.78 / 0.78` 安全边际。因此没有运行生产 cross-fit，也没有生成
任何新 locked case。

原型数值只保存在会话记录中，没有原始 checkpoint 或训练日志，仓库中的
迁移记录明确标记为 `conversation_record_only`、`replayable=false`。同时，
候选与 generic 对照存在参数量、loss 和 17/14 输入通道混杂，三项消融只能
作为诊断，不能支持组件必要性的因果解释。

## 2. 九通道线性载荷响应基：开发验证通过

当前 B 层是均质小应变线弹性问题。固定几何、网格、材料、边界条件和查询点
后，输出应力对远场载荷 `[Sigma_yy, Sigma_zz, Tau_yz]` 严格线性。因此每个
查询点可保存一个 `3 x 3` 响应矩阵，展平后恰好是九个通道：

```text
sigma_query = B_9(geometry, query) @ Sigma_farfield
```

在 v0.3 中恰有四个载荷的 120 个 seen 父几何上，使用三个载荷拟合、留出第四个
载荷预测，共得到 480 个逐载荷验证：

| 指标 | 九通道载荷基 | raw coarse 基线 |
|---|---:|---:|
| 平均 near-field tensor RelL2 | `5.481e-7` | `3.4201e-2` |
| 中位数 | `7.276e-8` | `3.4152e-2` |
| 95 分位 | `6.917e-7` | `4.2043e-2` |
| 最大值 | `1.127e-4` | `7.3034e-2` |
| 平均误差之比 | `1.603e-5` | `1.0` |

最大误差来自条件数 `67008` 的近共线随机三载荷组合；误差与载荷矩阵条件数
的对数相关系数约为 `0.83`。生产数据不应随机碰运气，而应固定三个规范载荷
`[1,0,0]`、`[0,1,0]`、`[0,0,1/sqrt(2)]`；最后一项来自对称张量范数中
剪应力的权重 2。其秩为 3、条件数为 `sqrt(2)`。

这一结果支持一个具体的数据构建变化：同一几何不再生成许多随机高保真载荷，
而是求解三个规范载荷并保存每点九通道响应基；其余载荷通过确定性矩阵乘法生成。
这既减少重复求解，也把线性叠加留在可审计的物理层，而不是要求神经网络重新学习。

seen-data 开发之后，冻结确认已在 clean、已推送实现提交 `44d244e` 上完成。
相对于冻结的 v0.2/v0.3 正式身份排除源，圆形、马蹄形和直墙拱形各采用一个
新几何身份；每个几何复用同一 mesh/query，分别直接求解三个规范基载荷和五个
held-out 载荷，共 `24/24` 次求解、零失败。15 个 held-out 面内总应力重建的
RelL2 中位数为 `4.886e-15`、最大值为 `5.882e-15`，17 个身份、求解器、网格、
响应重建门全部通过。独立结果审计复算了全部聚合门和执行账本，结论为
`LINEAR_ELASTIC_LOAD_AXIS_FACTORIZATION_CONFIRMED`。

这证明的是三个**各自固定**几何/材料/细网格/查询系统内的二维小应变平面应变
线弹性载荷轴分解，不证明不同几何之间的泛化、网格或材料泛化，也不覆盖损伤、
接触、断裂或岩爆动力学。

## 3. P1 应力恢复：体内改善，壁面失败

固定的面积/距离加权节点仿射恢复算子在 15 个按“分区 × 断面”预选的 seen case
上进行了 coarse/fine/ultrafine 三档真实 FEM：

| near-field 相对 ultrafine | raw coarse | recovered |
|---|---:|---:|
| 平均误差 | `3.1617%` | `1.3953%` |
| recovered/raw 中心比率 | `1.0` | `0.4413` |

15/15 case 均改善，最大逐 case 比率为 `0.5422`；circle、horseshoe、
straight-wall-arch 三族中心比率分别为 `0.3743 / 0.4468 / 0.4868`。
solver/mesh QC 全部通过，最大代数残差 `2.91e-13`、最大能量闭合误差
`2.01e-14`、最小三角形质量 `0.6883`。

但 wall-offset 诊断未通过：相对 ultrafine 的牵引与合力误差比率分别为
`1.1860` 和 `1.6039`。因此 v0.5 的正式开发路由是
`STOP_OR_REDESIGN_BEFORE_ANY_NEW_UNSEEN_CONFIRMATORY_RUN`，不能只凭体内误差
下降启动新正式实验。

下一候选是边界兼容的切向增量投影：在 wall-offset 仅保留恢复增量的
切向—切向分量，使修正增量严格满足 `Delta sigma n = 0`，从而精确保留 raw
coarse 的牵引；非 wall 查询仍使用原恢复场。这是看过 wall 失败后的新开发候选，
必须另立版本验证，不能倒写成 v0.5 预注册成功。

该 v0.5.1 重设计随后已在完全相同的 15 个 seen case 上执行。投影契约本身
通过：最大 `|Delta sigma n|=2.36e-16`，相对 raw coarse 的 traction/resultant
中心比率均为数值意义上的 `1.0`，近场比率仍为 `0.4413`。但 wall-offset
全应力 RelL2 相对 ultrafine 的比率仍为 `1.0412`，相对 fine 更为 `2.3031`。
因此冻结路由仍为 STOP；不能删除失败的全应力门来宣布 READY。下一次恢复研究若
继续，必须另立开发屏并分析切向应力偏差，而不是在这 15 个 case 上继续调投影。

## 4. 对“GeoPT 式硬岩隧洞数据”的直接影响

当前最稳妥的数据分层是：

1. `A-geometry`：GeoPT 兼容的几何—边界 lift，提供几何表示预训练；
2. `B-linear-basis`：每个几何三个规范载荷，监督九通道线弹性响应基；
3. `B-recovery-candidate`：粗网格 FEM 加边界兼容应力恢复的候选层；v0.5/v0.5.1
   均因 wall-offset 门槛失败而保持 STOP，当前不进入生产样本，只有通过新的
   development screen 后才能重新评估；
4. `C-fracture`：未来相场/FDEM/DEM 或微观实验时序，只学习损伤、裂纹、耗散、
   接触和路径依赖的非线性残差/状态演化。

九通道线性基可以复用 GeoPT 当前九维输出头的接口形状，但语义完全不同：
GeoPT 的九维是三步 vector-distance；这里的九维是 `3 x 3` 应力响应矩阵。
二者不能混用同一个标签说明，也不能据此声称模型已经学习岩体破裂。

## 5. 可追溯证据

- v0.4 停止记录：`artifacts/analysis/v04-structured-prototype-stop/`
- 载荷基开发结果：`artifacts/development/linear-load-basis-v0.5-development/`
- 载荷基独立确认：`artifacts/confirmation/linear-load-basis-v0.5.0/`
- 应力恢复开发结果：`artifacts/development/stress-recovery-v0.5-dev/`
- 边界投影重设计结果：`artifacts/development/stress-recovery-boundary-v0.5.1-dev/`
- 核心实现：`src/tunnelgeopt/load_basis.py`、`src/tunnelgeopt/stress_recovery.py`
- 契约文档：`docs/LINEAR_LOAD_BASIS.md`、`docs/STRESS_RECOVERY.md`
