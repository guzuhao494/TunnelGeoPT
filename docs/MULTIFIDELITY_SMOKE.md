# v0.3 多保真管线 smoke

运行命令：

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_multifidelity_smoke.py --phase all --device cuda
```

运行器按断面族分别冻结 `6/2/2` 个 train/dev/pseudo-test 父几何，每个父几何两个独立载荷，共 30 个父几何、60 个粗/细 FEM case。每个 case 在同一冻结边界上求解，只改变网格尺寸，并在 256 个预先确定的公共物理点上独立定位粗细单元。

数据被拆成三个文件：`public_inputs.npz`、`train_dev_labels.npz` 和 `sealed_pseudo_test_labels.npz`。训练阶段先写完并哈希全部 5 个 CPU checkpoint，随后才首次打开 sealed 文件；每个 checkpoint 对 pseudo-test 只作一次评价。生成数据和 checkpoint 位于被 Git 忽略的 `outputs/`，可审计的小型 JSON 工件位于 `artifacts/experiment/mf-residual-smoke-v0.3.0/`。

本 smoke 只允许 `pipeline_go/pipeline_no_go`，并强制 `effect_claim_allowed=false`。模型误差只能帮助检查数值有限性和接线，不能作为模型有效、标签效率、OOD 泛化、断裂或岩爆证据。正式实验还必须在新的 formal config 下完成五种子、全新 locked-IID/geometry-OOD/load-OOD、fine-ultrafine 和泄漏审计。
