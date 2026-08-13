# v0.3 Fine-Ultrafine 开发侧收敛审计

这个审计回答一个很窄、但必须先回答的问题：当前 `fine` 网格能不能作为 v0.3
多保真学习的数值标签。它不训练模型，也不读取 smoke pseudo-test 或任何 locked-test
标签，因此不能被当作模型效果、破裂、损伤、岩爆、现场有效性或迁移证据。

## 冻结设计

- 配置：`configs/multifidelity_convergence_dev.json`。
- 数据命名空间：`tunnelgeopt-v0.3-development-convergence-only`，与 smoke/formal
  测试种子分离。
- 三种断面各 4 个 development-only 父几何，每个 2 个载荷，共 24 case；父几何
  参数位于 ID 范围 `[0.15, 0.85]`，粗糙度严格大于零。
- 每个父几何先冻结 256 个公共 query：192 个 `0.05R–2.0R` 近场点、32 个
  `0.02R` 洞壁偏置点、32 个远场点。fine 与 ultrafine 在完全相同的边界、外域、
  载荷和 query 上计算。
- fine 显式冻结为 `mesh/wall/farfield = 0.4R/0.0625R/0.4R`；ultrafine
  为 `0.25R/0.03125R/0.25R`。这里不允许用隐式默认 `mesh_size`。
- 主指标是近场面积权重的应力张量 Frobenius 相对 L2，剪应力平方权重为 2；先得到
  每个 case 的误差，再让 24 个 case 等权计算 median/p95，并报告各断面族 median。

冻结门槛为：case median `<=3%`、p95 `<=5%`、任一断面族 median `<=4%`；同时
要求 24/24 case、每族 8 case、全部 query 均在两层网格中定位、代数残差与能量闭合
误差均 `<=1e-9`、最小三角形质量 `>=0.02`，并确认边界、外域和 query 身份相同。

## 运行

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_multifidelity_convergence.py --validate-only
.\.venv-gpu\Scripts\python.exe scripts\run_multifidelity_convergence.py
```

产物落在 `artifacts/analysis/mf-convergence-dev-v0.3.0/`：配置快照、环境、逐 case
指标、汇总指标、进度日志和带 SHA256 的 manifest。只有
`current_tiers_eligible_for_formal_freeze` 才表示当前两层网格可以进入正式配置冻结；
它仍然不允许任何模型效果结论。若为 `do_not_start_formal_with_current_tiers`，应先修改
development 网格设计并重新预注册，不得打开正式 locked 标签。

## 已执行结果（2026-08-13）

在 Python 3.12.13、NumPy 2.5.2、SciPy 1.18.0、scikit-fem 12.0.2、gmsh
4.15.2 环境中完成 24/24 case。主指标 case median 为 `2.116%`，p95 为
`3.036%`，最大值为 `3.776%`；circle、horseshoe、straight-wall-arch 三族
median 分别为 `1.769%`、`2.204%`、`2.564%`，全部通过冻结门槛。

数值 QC 同步通过：最大代数残差 `2.96e-13`，最大能量闭合误差 `2.25e-14`，
最小三角形质量 `0.6873`，所有 query 均在两层网格定位，边界、外域和 query hash
一致。审计记录只读取了 24 个 `dev` ultrafine 标签，`locked_test` 读取数为 0，未曾
解锁 locked test。最终决定是 `current_tiers_eligible_for_formal_freeze`，仅说明当前
fine/ultrafine 分辨率可以进入正式配置冻结，仍不构成模型效果证据。
