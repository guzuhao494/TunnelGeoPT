# v0.3 多保真残差预注册

标识：`v0.3-mf-residual-prereg-v1`  
日期：2026-08-13  
状态：smoke 契约已冻结；formal 数量级与门槛已冻结，精确生成 manifest 在 smoke 通过后、正式效果计算前落盘。

## 研究问题与结论边界

在二维、均质各向同性、小应变平面应变隧洞开挖中，测试时允许运行一次粗网格 FEM。主问题是 `Residual+Coarse@50% fine-train` 能否在未见父几何及预定义 OOD 上达到 `Scratch@100%` 与 `Direct+Coarse@100%` 的非劣性能，并明显修正 raw coarse 场。

这只检验“指定粗网格到指定细网格的合成线弹性离散修正”。Fine FEM 是本项目内的数值标签，不是真实世界真值。禁止据此声称破裂、损伤、岩爆、三维动力响应、微观到工程迁移或现场预测有效。

## 原子身份和防泄漏

- `geometry_group_id = hash(section family, normalized macro parameters, roughness realization, exact canonical boundary, outer-domain rule)`。
- `load_group_id = hash(normalized far-field stress tensor, material, boundary-condition convention)`。
- `case_group_id = hash(geometry_group_id, load_group_id)`。
- split 单位固定为父几何；同一父几何的全部载荷、coarse/fine/ultrafine、查询点、重网格与增强继承同一 split。
- coarse 输入在测试时合法；locked fine label 在全部 checkpoint 冻结并哈希前不可被训练器读取。
- 旧 v0.2 圆洞 locked test 与新 geometry/case/boundary/load hash 必须零交集。

## 几何、载荷和保真度

每族至少两个连续宏观参数改变归一化边界，且学习集合的粗糙度严格大于零。ID 参数位于冻结全范围的 `[0.15, 0.85]`；geometry-OOD 至少一个参数位于 `[0,0.10]` 或 `[0.90,1]`。

- circle-like：axis ratio、superellipse exponent、roughness realization；
- horseshoe：span/height、sidewall height/height、crown shape、roughness；
- straight-wall-arch：span/height、springline height/height、crown rise/span、roughness。

材料固定 `E/sigma_ref=500`、`nu=0.25`。ID 载荷为 `sigma1/sigma_ref in [0.30,0.80]`、`sigma3/sigma1 in [0.45,0.85]`、主应力角 `[-45,45] deg`。load-OOD 冻结为低侧压比 `[0.25,0.35]`、大转角 `[-85,-60] U [60,85] deg` 及二者联合。

网格只允许尺寸不同：coarse wall/farfield 为 `0.25R/0.8R`，fine 为 `0.0625R/0.4R`，ultrafine audit 为 `0.03125R/<=0.25R`。共同 query 在求解前由边界和独立 seed 决定；主评价区为距洞壁 `0.05R–2.0R`。

## 数据规模与冻结种子

Smoke 仅验证接线：每族 train/dev/pseudo-test 父几何为 `6/2/2`，每几何 2 个载荷，单训练 seed、短训练；不得据此作效果结论，也不生成正式 locked label。

Formal：train-ID 72 个父几何 x 4 载荷，dev-ID 18 x 4，locked-IID 30 x 4，locked-geometry-OOD 30 x 3，locked-load-OOD 30 x 3，locked-joint-OOD 15 x 3。训练子集按族平衡、父几何整组嵌套为 25/50/75/100%。

生成 seed 固定为 train/dev `310031`、locked-IID `310037`、geometry-OOD `310049`、load-OOD `310061`；训练 seeds 固定 `[103,211,307,401,509]`；split salt 固定 `tunnelgeopt-v0.3-mf-residual-20260813-v1`。

## 基线、指标和统计

必须运行 Fine oracle、Coarse-only、Scratch、Direct+Coarse、Residual+Coarse 和 Mismatched-Coarse。Direct 与 Residual 使用完全相同的 14 通道输入和骨干；Scratch 的 coarse 三通道固定为零。Mismatched 使用同族无固定点置换的错误 coarse 场。

主指标是每 case 面积权重张量 Frobenius 相对 L2（剪应力权重 2），随后先在父几何内平均载荷，再令三族等权。同步报告每族、OOD 类型、均值/中位数/p90、洞壁牵引/合力残差、非有限值和耗时。

正式置信区间用 20,000 次层级配对 bootstrap：先重采样五个配对训练 seed，再在每族内重采样父几何；同一父几何全部载荷始终同行。

## Scientific GO

记 `R_s=Residual50/Scratch100`、`R_d=Residual50/DirectCoarse100`、`R_c=Residual50/CoarseOnly`。

全部条件必须同时成立：

1. 数据、solver、leakage 和 fine-ultrafine QC 有效；ultrafine 审计中位数 `<=3%`、p95 `<=5%`、任一族中位数 `<=4%`。
2. locked-IID 的单侧 95% 上界：`R_s<=1.02`、`R_d<=1.02`、`R_c<=0.70`。
3. geometry-OOD 与 load-OOD 分别：`R_s<=1.05`、`R_d<=1.05`、`R_c<=0.80`。
4. 至少 4/5 seeds 在 IID 上同时满足 `R_s,R_d<=1.05`，两个 OOD 上同时满足 `<=1.10`。
5. IID 三族均无 `>1.10` 灾难退化，至少 2/3 族不超过 `1.02`；任一 OOD 载荷类型点估计不超过完整标签基线 `1.15`。
6. 预测非有限值为零，洞壁牵引与全局合力残差不超过冻结门槛，且不以比 raw coarse 更差的物理残差换取 L2。
7. checkpoint/config/manifest/access log/results 均有 hash，locked fine test 只在全部 checkpoint 冻结后评价一次。

若数据泄漏、test 参与调参、fine-ultrafine 失败、任一族 solver/QC 有效率低于 95%、少于五训练种子或主 CI 总宽度大于 0.10，则为 `ABSTAIN` 并以新 salt 重建测试；不得把这些情况伪装成模型 No-Go。
