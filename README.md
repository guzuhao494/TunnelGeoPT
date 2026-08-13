# TunnelGeoPT

[![CI](https://github.com/guzuhao494/TunnelGeoPT/actions/workflows/ci.yml/badge.svg)](https://github.com/guzuhao494/TunnelGeoPT/actions/workflows/ci.yml)

面向硬岩隧洞的物理感知合成数据与几何预训练实验平台。

本项目借鉴 GeoPT 的核心经济性：用大量廉价的几何—边界监督进行预训练，再用少量昂贵的物理仿真监督进行适配。项目不会把随机轨迹等同于岩爆物理；当前代码首先验证“几何—应力先验是否能减少高保真破裂样本”，随后才扩展到损伤、断裂、能量释放和岩爆时序。

## 当前状态

| 状态 | 内容 |
|---|---|
| 已实现 | 三类二维隧洞断面、可控粗糙度、围岩/洞壁采样、GeoPT兼容的三步lifted样本、Kirsch圆洞应力解、数据契约与单元测试 |
| 本机环境已核验 | Windows和WSL2均能识别NVIDIA GeForce RTX 5070 Ti Laptop GPU（12227 MiB）；WSL2为Ubuntu 24.04、Python 3.12 |
| 当前阻塞 | WSL内项目依赖安装因访问PyPI超时未完成；这不影响WSL2、Python和GPU可见性已经通过的结论 |
| 尚未验证 | PyTorch/CUDA训练、Transolver训练、OpenFDEM/YADE/MOOSE求解器运行、高保真破裂数据生成 |
| 研究假设 | Stress-Lift预训练在未见几何/应力/材料划分上，比从零训练、静态几何预训练和原始随机lift预训练更节省高保真标签 |

详细边界见 [科学设计](docs/SCIENTIFIC_DESIGN.md)、[数据契约](docs/DATA_SCHEMA.md)、[验证协议](docs/VALIDATION.md)和[求解器路线](docs/SOLVER_ROADMAP.md)。

## 数据路线

```text
程序化断面 + 原岩应力/开挖提示
              │
              ├── A层：廉价几何lift（当前已实现）
              │          x[N,7] + condition[N,4] -> supervise[N,9]
              │
              ├── B层：解析解/线弹性FEM（Kirsch锚点已实现）
              │          位移、应力、应变能
              │
              └── C层：FDEM/DEM/相场（计划）
                         损伤、裂纹图、耗散能、AE源事件
```

A层只提供几何—边界先验。只有B/C层加入动量平衡、材料本构、损伤不可逆和断裂耗散后，模型输出才可以被解释为岩体力学预测。

## 快速开始（Windows PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest

# 生成一个小型、可复现的GeoPT兼容案例
.\.venv\Scripts\python.exe -m tunnelgeopt.cli generate `
  --shape horseshoe --n-volume 256 --n-surface 64 `
  --n-prompts 2 --seed 42 --output outputs\smoke\case_000000

# 校验已有案例
.\.venv\Scripts\python.exe -m tunnelgeopt.cli validate outputs\smoke\case_000000
```

记录当前环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_environment.ps1 `
  -OutputPath validation\environment\windows.json
```

## WSL2路线

高保真求解器和后续GPU训练建议在WSL2/Linux中运行。不要在Windows与WSL之间共用同一个虚拟环境。

```bash
python3 -m venv .venv-wsl
. .venv-wsl/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
bash scripts/check_environment.sh validation/environment/wsl.json
```

环境脚本只记录版本和硬件可见性，不读取或保存GitHub令牌、密钥或其他凭据。

## GeoPT兼容接口

每个廉价预训练案例保存：

```text
case_xxxxxx/
  x.npy                 float16 [N, 7]
  condition_0.npy       float16 [N, 4]
  supervise_0.npy       float16 [N, 9]
  ...
  meta.json
```

- `x = [x, y, z, d_wall, g_x, g_y, g_z]`，其中围岩点的 `g` 指向最近洞壁，洞壁点保存指向围岩侧的表面法向
- `condition = [direction_x, direction_y, direction_z, step_length]`
- `supervise = [dvec_t0, dvec_t1, dvec_t2]`

坐标约定为 `x=隧洞轴向、y=竖向、z=横向`。当前二维平面应变样本的 `x=0`，同时在元数据中保留特征半径和归一化方式。

## 首个预注册比较

在严格按完整算例划分、不得按点或时间帧泄漏的条件下，比较：

1. 从零训练；
2. 静态距离/法向预训练；
3. 原始随机lift预训练；
4. Stress-Lift预训练。

Go/No-Go门槛预先设为：在未见案例上达到相同误差时，Stress-Lift至少减少20%的高保真训练案例，并在三个随机种子上稳定优于前三个基线。这是后续实验门槛，不是当前结果。

## 项目结构

```text
configs/                 试验参数
docs/                    科学设计、数据与验证契约
scripts/                 环境检查和运行入口
src/tunnelgeopt/         生成器、解析解与数据校验
tests/                   单元和物理不变量测试
validation/              小型、可审计的验证证据
artifacts/experiment/    试验清单与结果摘要
```

## 来源与许可

本项目是独立实现，设计受 [GeoPT官方仓库](https://github.com/Physics-Scaling/GeoPT) 启发，并在 `CITATION.cff` 中保留学术引用。代码采用MIT许可。未来接入OpenFDEM、YADE、MOOSE等求解器时，各外部组件继续遵守其自身许可证，本仓库不重新分发这些求解器。
