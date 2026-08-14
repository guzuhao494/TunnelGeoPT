# TunnelGeoPT

[![CI](https://github.com/guzuhao494/TunnelGeoPT/actions/workflows/ci.yml/badge.svg)](https://github.com/guzuhao494/TunnelGeoPT/actions/workflows/ci.yml)

面向硬岩隧洞的物理感知合成数据与预训练实验平台。项目借鉴
[GeoPT](https://github.com/Physics-Scaling/GeoPT)“先学习廉价几何—边界关系，再用较少昂贵物理样本适配”的思路，但不复刻其代码。

> **v0.3 formal result (2026-08-13): `ABSTAIN`.** 705/705 个多保真
> 线弹性 case 和全部数值/访问审计通过，但 IID 与 geometry-OOD 的主比率
> Bootstrap 区间宽度超过预注册上限，因此不允许作 GO 或 NO-GO 效应主张。
> `Residual50` 的诊断结果在 load OOD 上优于两个学习基线，在 IID/geometry
> OOD 上却未稳定优于 `Direct100`。完整证据和边界见
> [`RESULT_SUMMARY.md`](artifacts/experiment/mf-residual-formal-v0.3.0/RESULT_SUMMARY.md)。

> **v0.5 development result (2026-08-13): 数值算子路线取得可复核进展。**
> 在全部视为 seen 的 v0.3 标签上，九通道线性载荷响应基对 120 个父几何做
> 480 次逐载荷留一预测，中位 near-field RelL2 为 `7.276e-8`；固定 P1 应力
> 恢复在 15 个三档 FEM case 上将相对 ultrafine 的平均误差从 `3.1617%`
> 降至 `1.3953%`，但 wall-offset 牵引/合力诊断恶化，故该恢复版本按规则停止并
> 转向边界兼容重设计；重设计虽精确保留了粗网格牵引，但 wall-offset 全应力仍
> 略微恶化，因此同样记录为 STOP。结果、限制和数据路线见
> [`MILESTONE_V05_NUMERICAL_OPERATORS.md`](docs/MILESTONE_V05_NUMERICAL_OPERATORS.md)。

> **v0.5 load-basis confirmation (2026-08-14): `CONFIRMED` within the frozen
> linear-elastic scope.** 在 clean、已推送的实现提交 `44d244e` 上，圆形、马蹄形和
> 直墙拱形三个新身份各执行 `3` 个规范基载荷与 `5` 个 direct-FEM held-out 载荷，
> 共 `24/24` 次求解、`15` 次独立重建。面内总应力 RelL2 的中位数为
> `4.886e-15`、最大值为 `5.882e-15`，全部 17 个身份、数值和物理 QC 门通过。
> 这只确认固定几何/材料/网格/查询条件下的二维小应变线弹性载荷轴分解，绝不表示
> 几何泛化、破裂、损伤或岩爆能力。证据见
> [`RESULT_SUMMARY.md`](artifacts/confirmation/linear-load-basis-v0.5.0/RESULT_SUMMARY.md)。

> **论文 Stage-2 开发状态（2026-08-14）：kernel/schema/protocol 已实现，pilot
> `NO-GO`。** 现有线弹性载荷基只作为
> 后续方法引理和固定弹性骨架，不单独包装为论文创新。推荐主线是先验证二维准静态
> 相场脆性断裂求解器，再检验“精确弹性基 + 只学习不可逆损伤残差”是否真能节省
> 断裂轨迹标签。当前已有本地 P1 AT2 调试内核、严格轨迹 schema、36-case/12-audit
> 冻结开发协议和裂纹带网格审计，但求解器尚不能表达 P2/P3 的变远场应力或 P4
> 分区卸荷。v3 加载适配器及 development-only scheduled API 已覆盖这些路径，
> 求解器也会从最终同一 `(u,d)` 输出结点力；schema v3 则能从原始量重算反力、
> 分项路径功和能量不平衡。development-only 轨迹适配已接入冻结 schedule
> 重放、减半重试、全局平衡和保存-重载-复验。SENT/SENS 的冻结协议、
> 显式零厚度双裂面网格、位移控制 BVP 和受限完整体探针现已实现；六档真实
> Gmsh 网格通过拓扑与尺寸审计。SENT/SENS 成对 coarse fixed-`d=0` 三状态实跑
> 已完成且 6/6 状态通过适用的平衡门，但其完整 coarse 轨迹固定损伤耗时下界已达
> 约 113 h。耦合轨迹、三网格收敛和完整标签生成器仍未闭合，所以禁止生成训练标签。
> 固定版本 MOOSE
> 的官方 `crack2d_iso` 自测已经独立通过；本地与 MOOSE 在同一 TRI3 网格和
> 边界条件下的 3 个完好弹性与 3 个固定非均匀 P1 损伤平衡算例也已通过
> `1e-6` 冻结门。这仍不是裂纹演化或岩爆验证。范围、反证和论文大纲见
> [`RESEARCH_SCOPE.md`](paper/RESEARCH_SCOPE.md)、
> [`PAPER_OUTLINE.md`](paper/PAPER_OUTLINE.md) 与 [`PLAN.md`](PLAN.md)。

v0.2 已从几何数据生成器推进到一个可计算、可验证、可持久化的二维线弹性里程碑：程序可以为圆形、马蹄形和直墙拱形隧洞生成有限元网格，求解均质各向同性平面应变开挖增量，并将位移、应变、应力和应变能保存为独立 B-elastic 数据记录。

## 当前证据边界

| 状态 | 当前证据 |
|---|---|
| 已实现并测试 | A 层三类断面及 GeoPT 兼容 lifted 样本；B 层 Gmsh 网格、scikit-fem P1 平面应变求解、case/split 身份、严格持久化 schema、Kirsch/仿射 patch/残差/能量验证 |
| 本机科学栈 | Windows Python 环境已实测 NumPy、SciPy、scikit-fem、Gmsh 的导入、网格和稀疏计算 |
| GPU 实算 | Windows Python 3.12 + PyTorch `2.11.0+cu128` 已在 RTX 5070 Ti Laptop GPU 上完成 CUDA smoke，并完成 6 方法 × 3 种子的圆洞解析迁移训练；后者结果为 No-Go，只适用于解析圆洞筛查 |
| WSL2 | Ubuntu 24.04、Git 和同一 GPU 的可见性已通过；官方 MOOSE conda 栈与 `combined-opt` 已构建，固定 `crack2d_iso` 自测及本地-MOOSE 同网格固定状态比较均已运行通过并留存哈希证据 |
| C-fracture 调试层 | 本地 P1 AT2 spectral-split 内核、不可逆 active-set、schema v3、冻结开发协议、裂纹带网格、v3 P1-P4 scheduled loading、反力/路径功、development-only 轨迹减半重试与全局平衡已实现并测试；同网格固定损伤 MOOSE 比较已通过；SENT/SENS 六档显式裂面网格及位移控制适配已通过合同审计，成对 coarse fixed-`d=0` 三状态实跑 6/6 通过，但完整 coarse 固定损伤耗时投影约 113 h，耦合轨迹、三网格结果和完整生成器尚未完成 |
| 尚未实现 | 可接受的高保真断裂标签集、动态破坏、微震/AE 波形、FDEM/DEM、实验校准和现场迁移 |

因此，B 层当前是**静态、均质、各向同性、小应变线弹性**结果，不是岩爆、岩体破裂或现场预测结果。schema 明确不允许使用全零 `damage`、`velocity` 或 `dissipation` 字段冒充高保真标签。

环境证据保存在：

- [`validation/environment/physics_stack.json`](validation/environment/physics_stack.json)
- [`validation/environment/windows_gpu_torch.json`](validation/environment/windows_gpu_torch.json)
- [`validation/environment/wsl_python_stack.json`](validation/environment/wsl_python_stack.json)

## 数据路线

```text
程序化断面 + 原岩应力/开挖条件
              │
              ├── A 层：廉价几何 lift（已实现）
              │          x[N,7] + condition[N,4] -> supervise[N,9]
              │
              ├── B 层：解析解 + 线弹性 FEM（v0.2 已实现）
              │          网格、位移、应变、总/增量应力、应变能
              │
              └── C 层：准静态 AT2 调试内核（部分实现，标签生成 NO-GO）
                         待补完整载荷、反力/累计功、重试和外部基准后再生成
```

研究路线、schema 和验证边界分别见：

- [`PLAN.md`](PLAN.md)
- [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md)
- [`docs/CASE_MANIFEST.md`](docs/CASE_MANIFEST.md)
- [`docs/ELASTIC_SCHEMA.md`](docs/ELASTIC_SCHEMA.md)
- [`docs/VALIDATION.md`](docs/VALIDATION.md)
- [`docs/SOLVER_ROADMAP.md`](docs/SOLVER_ROADMAP.md)
- [`docs/MILESTONE_V0.2.md`](docs/MILESTONE_V0.2.md)
- [`docs/FRACTURE_SCHEMA.md`](docs/FRACTURE_SCHEMA.md)
- [`docs/FRACTURE_PHASE1_PROTOCOL.md`](docs/FRACTURE_PHASE1_PROTOCOL.md)
- [`docs/FRACTURE_MESH.md`](docs/FRACTURE_MESH.md)
- [`docs/FRACTURE_WORK.md`](docs/FRACTURE_WORK.md)
- [`docs/FRACTURE_SOLVER_LOADING.md`](docs/FRACTURE_SOLVER_LOADING.md)
- [`docs/FRACTURE_BENCHMARKS.md`](docs/FRACTURE_BENCHMARKS.md)

## 安装与测试（Windows PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
```

B 层依赖属于 `elastic` 可选依赖组；`dev` 已包含它们。环境脚本只记录版本和硬件可见性，不读取或保存令牌、密钥或其他凭据。

## A 层：GeoPT 兼容几何样本

```powershell
.\.venv\Scripts\python.exe -m tunnelgeopt.cli generate `
  --shape horseshoe --n-volume 256 --n-surface 64 `
  --n-prompts 2 --seed 42 --output outputs\smoke\case_000000

.\.venv\Scripts\python.exe -m tunnelgeopt.cli validate `
  outputs\smoke\case_000000 --require-meta
```

每个 A 层 case 保存：

```text
case_xxxxxx/
  x.npy                 float16 [N,7]
  condition_0.npy       float16 [N,4]
  supervise_0.npy       float16 [N,9]
  meta.json
```

`x = [x,y,z,d_wall,g_x,g_y,g_z]`，`condition` 保存 lifted 方向和步长，`supervise` 保存三个时刻的距离向量监督。

## B 层：线弹性求解与持久化

CLI 对用户暴露**压应力为正**的岩石力学输入；进入求解器时一次性转换为内部**拉应力为正**约定。参数名中的 `compression` 是有意设计，避免符号约定被默默猜测。

```powershell
.\.venv\Scripts\python.exe -m tunnelgeopt.cli elastic-solve `
  --shape horseshoe --output outputs\elastic\case_000001 `
  --young-modulus 5.0e10 --poisson-ratio 0.25 `
  --sigma-yy-compression 3.0e7 `
  --sigma-zz-compression 2.0e7 `
  --tau-yz-compression 0

.\.venv\Scripts\python.exe -m tunnelgeopt.cli elastic-validate `
  outputs\elastic\case_000001
```

`elastic-solve` 只有在有限性、矩阵对称性、自由度残差和能量闭合均通过后才保存：

```text
case_000001/
  arrays.npz   # nodes/elements/facets/u/strain/stress/energy 等
  meta.json    # case_group_id/mesh_id/config_hash/env/hash/QC 等
```

默认使用严格 `float64`。如确需发布 `float32`，必须在求解与校验两端显式指定；载入会复验文件哈希、内容哈希、数组清单、拓扑、shape、有限性、分量顺序、符号、SI 单位、本构和能量关系。

## 冻结 Kirsch 多网格验证

```powershell
.\.venv\Scripts\python.exe -m tunnelgeopt.cli elastic-kirsch `
  --output validation\elastic\kirsch-v0.2.json `
  --young-modulus 1.0e9 --poisson-ratio 0.25 `
  --sigma-yy-compression 1.0e6 `
  --sigma-zz-compression 0
```

命令依次运行粗、中、细三档圆洞网格，记录每档网格规模、环带应力相对 L2、洞壁牵引残差、峰值环向应力误差、求解器 QC 和仿射 patch 测试。报告以 SHA-256 冻结写入指定 JSON；任一冻结门槛失败仍保留报告并返回非零退出码 `2`，适合 CI 和数据生成门禁。

## WSL2 路线

高保真求解器和后续 GPU 训练仍优先考虑 WSL2/Linux，但不要在 Windows 与 WSL 之间共用一个虚拟环境：

```bash
python3 -m venv .venv-wsl
. .venv-wsl/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
bash scripts/check_environment.sh validation/environment/wsl.json
```

当前 WSL 已通过本机代理安装官方 MOOSE conda 依赖并完成 `combined-opt` 构建。
固定 MOOSE 与项目提交下的官方 `crack2d_iso` 自测已经运行通过；公开证据见
[`result.json`](artifacts/development/moose-crack2d-iso-v1/result.json)。TunnelGeoPT
同网格、同边界的完好弹性/固定损伤平衡比较也已通过；证据见
[`artifacts/development/moose-local-same-problem-v1`](artifacts/development/moose-local-same-problem-v1)。
官方自测只证明所钉住 MOOSE 版本能执行其自身回归用例，两者都不是耦合裂纹演化或岩爆验证。

## 首个预注册学习比较：No-Go

圆洞 Kirsch 解析筛查已按完整 `case_group_id` 拆为 168/36/36 train/dev/locked-test，并在 GPU 上完成 Scratch@80/100%、Static、Random-Lift、Shuffled-Stress-Lift 和 Stress-Lift 六种方法、三个种子的正式比较。

Stress-Lift@80% 相对 Scratch@100% 的误差比分别为 `1.166 / 0.638 / 1.618`，只通过 1/3 种子；平均主误差 `0.017642` 也高于 Random-Lift 的 `0.016773`。Shuffled 对照平均误差反而最低（`0.015944`），因此不能把单个好种子的收益解释为稳定学到了应力—边界耦合。正式判定为 **No-Go**，不声称节省了 20% 标签。

这一负结果和 B-elastic 的 GO 共同构成 v0.2 里程碑。下一步转向跨断面、粗网格到细网格的弹性场/残差学习，并重新建立未使用的锁定测试集。完整指标、审计和边界见 [`docs/MILESTONE_V0.2.md`](docs/MILESTONE_V0.2.md)。

## 来源与许可

本项目为 MIT 许可的独立实现。未来接入 OpenFDEM、YADE、MOOSE 等外部求解器时，各组件继续遵守自身许可证，本仓库不重新分发这些求解器。
