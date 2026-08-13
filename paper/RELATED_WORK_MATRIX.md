# Related-work matrix

Last source check: 2026-08-14

Publication status is stated explicitly. ArXiv-only work is not described as
peer reviewed.

## Geometry-aware and pretrained neural operators

| Work | Status | Relevant contribution | Boundary relative to this paper |
|---|---|---|---|
| [Geo-FNO](https://jmlr.org/papers/v24/23-0064.html) | JMLR 2023 | learned deformation for operators on general geometries | no tunnel mechanics, fracture trajectory, or mechanics-specific load factorization |
| [GINO](https://papers.nips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html) | NeurIPS 2023 | scalable geometry-informed operator for large 3D PDE domains | primarily industrial fluid mechanics; no brittle-damage rollout |
| [Transolver](https://proceedings.mlr.press/v235/wu24r.html) | ICML 2024 | physics-attention on irregular geometry/mesh points | architecture baseline, not a tunnel-specific physical factorization |
| [PI-GANO](https://doi.org/10.1016/j.cma.2024.117540) | CMAME 2025 | physics-informed, geometry-aware operator learning | general engineering PDEs; no exact load-basis backbone or tunnel fracture split |
| [GeoPT](https://arxiv.org/abs/2602.20399) | ICLR 2026 paper; official [code](https://github.com/Physics-Scaling/GeoPT) | dynamics-lifted geometry pretraining with more than one million solver-free samples | steady-state industrial simulation; its vector-distance transport is not crack evolution, fracture energy, or scale transfer |
| [Latent shape pretraining](https://arxiv.org/abs/2509.25788) | arXiv 2025 | uses cheap shape corpora before expensive physics fine-tuning | geometry autoencoding rather than explicit intact/nonlinear mechanics separation |

## Multi-fidelity and mechanics operators

| Work | Status | Relevant contribution | Boundary relative to this paper |
|---|---|---|---|
| [Multifidelity deep neural operators](https://doi.org/10.1103/PhysRevResearch.4.023210) | Physical Review Research 2022 | LF input augmentation and residual learning | generic fidelity coupling, not a verified mechanics response basis |
| [Multi-fidelity DeepONet residual ROM](https://link.springer.com/article/10.1186/s40323-023-00249-9) | AMSES 2023 | residual correction of reduced-order predictions | closest basis/residual neighbor, but no tunnel geometry or irreversible fracture rollout |
| [Sequential DeepONet for transient vector fields](https://link.springer.com/article/10.1007/s00707-024-03991-2) | Acta Mechanica 2024 | sequential prediction under path-dependent loading | does not combine rich tunnel geometry, exact intact basis, and multi-fidelity labels |
| [Multi-fidelity DeepONet for tunnel settlement](https://doi.org/10.1016/j.engappai.2024.108156) | Engineering Applications of Artificial Intelligence 2024 | fuses process simulations with sparse field monitoring for real-time settlement fields | closest tunnel/operator application, but predicts soft-ground settlement rather than brittle surrounding-rock damage |
| [Bayesian multi-fidelity neural operator for spinodal metamaterials](https://www.nature.com/articles/s41524-026-02112-y) | npj Computational Materials 2026 | sparse HF experiment plus LF mechanics simulations | different mechanics and geometry regime; no load-axis factorization |
| [Benchmarking multi-fidelity neural operators](https://arxiv.org/abs/2608.04708) | arXiv 2026 | shows that MF coupling does not reliably beat direct HF learning | motivates strong direct and transfer baselines rather than assuming LF is useful |

## Variational and phase-field brittle fracture

| Work | Status | Relevant contribution | Use in this project |
|---|---|---|---|
| [Bourdin, Francfort and Marigo (2000)](https://doi.org/10.1016/S0022-5096(99)00028-9) | Journal of the Mechanics and Physics of Solids | foundational numerical regularization of variational brittle fracture | theoretical and numerical origin |
| [Bourdin, Francfort and Marigo (2008)](https://link.springer.com/article/10.1007/s10659-007-9107-3) | Journal of Elasticity | comprehensive variational fracture framework | formulation and claim boundary |
| [Miehe, Welschinger and Hofacker (2010)](https://onlinelibrary.wiley.com/doi/10.1002/nme.2861) | International Journal for Numerical Methods in Engineering | thermodynamically consistent phase-field fracture, history/irreversibility, standard tension/shear benchmarks | main benchmark and tensile/compressive split reference |
| [Borden et al. (2012)](https://doi.org/10.1016/j.cma.2012.01.008) | CMAME | quasi-static shear benchmark and dynamic brittle-fracture extension | independent benchmark source; dynamic results remain outside paper scope |
| [Borden et al. (2014)](https://doi.org/10.1016/j.cma.2014.01.016) | CMAME | higher-order phase-field formulation | related variant, not the selected implementation |
| [MOOSE phase-field fracture material](https://mooseframework.inl.gov/source/materials/ComputeLinearElasticPFFractureStress.html) | official software documentation | executable `crack2d_iso.i` example with spectral decomposition | required independent implementation check |
| [PhAST SENT/SENS setup](https://github.com/CEMS-Lab/PhAST/blob/main/docs/user_guide/setup_problems.md) | official project documentation | public geometry recipes for Miehe tension/shear benchmarks | geometry and boundary-condition cross-check |
| [Phase-field tunnel excavation damage](https://doi.org/10.1016/j.engfailanal.2024.109113) | Engineering Failure Analysis 2025 | anisotropic phase-field model applied to layered-rock tunnel excavation | establishes the tunnel-fracture mechanics setting, but has no reusable learned operator or label-efficiency experiment |
| [Variational DeepONet for brittle crack paths](https://doi.org/10.1016/j.cma.2022.114587) | CMAME 2022 | physics-informed operator for crack-path prediction | strong fracture-surrogate neighbor and baseline source |
| [Phase-field fracture with physics-informed deep learning](https://doi.org/10.1016/j.cma.2024.117104) | CMAME 2024 | Deep Ritz solution of nucleation, propagation, kinking, branching, and coalescence | instance-specific neural solver rather than a reusable tunnel-geometry rollout operator |
| [Two-step DeepONet for brittle fracture](https://doi.org/10.1016/j.cma.2025.117984) | CMAME 2025 | data-driven and physics-informed DeepONets for bars and single-edge-notch specimens | close operator-learning neighbor; varies notch/loading but does not use a verified tunnel elastic backbone |
| [IFENN for phase-field fracture](https://doi.org/10.1016/j.cma.2025.118485) | CMAME 2026 | embeds a physics-informed CNN in FEM and reports transfer across rectangular geometries, cracks, meshes, and loads | strong hybrid-solver neighbor; learns local energy-to-damage coupling rather than allocating labels through a fixed elastic response basis |
| [Robust brittle-fracture surrogate benchmark](https://doi.org/10.1016/j.cma.2025.118526) | CMAME 2026 | 6,000 phase-field simulations with 100 steps and PINN, U-Net, and FNO baselines | closest standardized surrogate benchmark; coupon/crack configurations rather than tunnel-section and load-basis factorization |

## Rockburst and monitoring AI

| Work | Status | Relevant contribution | Boundary relative to this paper |
|---|---|---|---|
| [Rockburst prediction using AI: a review](https://doi.org/10.1016/j.rockmb.2024.100129) | Rock Mechanics Bulletin 2024 | summarizes tabular/classification-oriented rockburst AI | supports the gap between hazard classifiers and geometry-conditioned field models |
| [Long- and short-term rockburst ML review](https://doi.org/10.1016/j.tust.2023.105434) | Tunnelling and Underground Space Technology 2023 | engineering prediction and warning review | does not validate synthetic full-field fracture surrogates |
| [Microseismic multi-task rockburst identification](https://doi.org/10.1016/j.jrmge.2025.07.017) | JRMGE 2025 | real monitoring signal processing and disaster identification | represents the future observation side, not current synthetic solver truth |
| [Diffusion augmentation for rockburst levels](https://doi.org/10.1016/j.jrmge.2026.01.048) | JRMGE 2026 | task-level synthetic augmentation | not mechanics-consistent field generation or micro-to-tunnel transfer |

## Novelty position to test, not assume

The candidate contribution is not a new attention block and not the linear
superposition law. It is the controlled combination of:

1. a tunnel-specific, independently verified intact response basis;
2. a causal operator that learns only irreversible fracture deviations;
3. parent-level geometry/material/load-path OOD evaluation;
4. end-to-end fracture-label and solver-cost accounting.

The literature search has not established a safe “first” claim. The manuscript
will use comparative language unless a later systematic search rules out all
close tunnel-fracture operator work.
