# Solver Roadmap

Last updated: 2026-08-13

This note is intentionally narrow: it only covers a practical solver split for synthetic hard-rock tunnel / brittle-fracture data generation. It does not claim any solver is ready merely because a binary exists on disk.

## Recommendation

### MVP toolchain
- Geometry / meshing: `gmsh`
- Fast synthetic fracture fields: `MOOSE` with TensorMechanics + phase-field fracture
- Optional event-style proxy generation: postprocess damage, strain-energy release, crack-surface increments, or stress-drop bursts into an AE-style proxy timeline
- Why this MVP: open source, documented, scriptable, Linux-first, and strong enough to generate stress-strain-damage-crack trajectories without immediately paying the FDEM contact cost

### High-fidelity toolchain
- Geometry / meshing: `gmsh`
- Near-field explicit fracture and fragment/contact evolution: `OpenFDEM`
- Optional cross-check / alternative family: `Peridigm` or `LAMMPS PERI` when nonlocal fracture or particle-style calibration is more important than tunnel-engineering workflow fit
- Why this high-fidelity route: better aligned with discrete crack initiation, propagation, block separation, and contact after failure, which matters for brittle hard-rock tunnel scenarios

## Solver split

| Solver | Family | License | OS reality | Parallel / GPU | Useful outputs | Deployment difficulty | Recommendation |
|---|---|---|---|---|---|---|---|
| [MOOSE](https://mooseframework.inl.gov/) | FEM multiphysics; phase-field fracture via TensorMechanics | [LGPL 2.1](https://github.com/idaholab/moose/blob/master/LICENSE) | Officially Linux/macOS oriented; Windows usually means WSL or container, not native first-class support | MPI via PETSc/libMesh; no general production GPU path to assume | Stress, strain, damage-like phase field, crack path, energy terms; AE is indirect proxy only | Medium | Best open MVP |
| [OpenFDEM](https://openfdem.com/) | FDEM | [LGPL; official page currently contains inconsistent 2.1-or-later and 3-or-later wording](https://openfdem.com/rst_about_introduction/copyrights.html), so verify the downloaded source notice | Linux-centric build stack; Windows should be treated as non-default unless you prove the toolchain in your environment | Parallel/GPU capability must be verified against the exact build; no default GPU assumption | Stress, strain, damage, explicit fracture, fragment/contact evolution; AE source/event outputs are not sensor-level waveforms | High | Best open high-fidelity tunnel-fracture candidate |
| [Peridigm](https://github.com/peridigm/peridigm) | Peridynamics | [BSD-3-Clause](https://github.com/peridigm/peridigm/blob/master/LICENSE.md) | Linux/HPC oriented in practice | MPI on Trilinos stack; no default GPU assumption | Damage, bond failure, displacement/stress-style field outputs; AE is proxy | High | Strong research fallback for nonlocal fracture |
| [LAMMPS PERI](https://docs.lammps.org/Howto_peri.html) | Peridynamics in particle engine | [GPL-2.0-or-later](https://github.com/lammps/lammps/blob/develop/LICENSE) | Linux strong; Windows possible but usually not the easiest serious workflow | MPI, OpenMP, and package-dependent accelerators exist in LAMMPS, but do not assume PERI workflow is GPU-ready without proof | Bond breakage and particle/state outputs; AE is proxy | Medium-High | Good for scalable synthetic event generation if team already knows LAMMPS |
| [YADE](https://www.yade-dem.org/) | DEM | [GPL v2 license text](https://gitlab.com/yade-dev/trunk/-/blob/master/LICENSE); exact “or later” applicability should be checked from source notices | Linux first; Windows use is secondary | OpenMP; limited HPC expectations compared with FEM/FDEM stacks | Contact forces, breakage with cohesive models, and an [acoustic emission helper concept](https://yade-dem.org/doc/acousticemissions.html) | Medium | Good DEM-side exploratory generator, not my first tunnel MVP |
| [PFC](https://www.itascacg.com/software/pfc) | Commercial DEM | Commercial | Windows and Linux are both supported by vendor distributions | Parallel support exists; GPU availability must be validated against the exact licensed version | Strong bonded-particle fracture/contact workflows; AE-like event counting commonly done in practice | Medium if licensed, impossible if not | Commercial benchmark if team already owns licenses |
| [Abaqus](https://www.3ds.com/products/simulia/abaqus) | Commercial FEM/XFEM/continuum damage | Commercial | Windows and Linux supported by SIMULIA | Parallel supported; some GPU acceleration exists in selected workflows, but never assume feature coverage without checking the installed release | Stress, strain, damage, XFEM cracks; AE is proxy unless separately modeled | High | Commercial high-end baseline, not the first open pipeline |

## What each route is actually good for

### MOOSE MVP
- Best when you need many controlled synthetic runs with explicit boundary conditions, constitutive parameters, and mesh-level fields.
- Best output contract for ML data generation: load step, stress-strain curve, phase-field damage map, crack initiation time, crack growth path, released energy proxy.
- Weak point: post-failure fragmentation/contact is not the natural strength, and AE is not a first-class native waveform output.

### OpenFDEM high fidelity
- Best when brittle crack coalescence, block separation, and contact after failure are part of the signal you care about.
- Best output contract for ML data generation: element/node stress-strain fields, damage/fracture states, crack network evolution, fragment kinematics, contact episodes, energy release proxies.
- Weak point: heavier build chain and higher calibration cost.

### Peridigm / LAMMPS PERI fallback
- Best when you want nonlocal fracture physics or a cleaner research path for bond-failure style datasets.
- Less directly aligned with "hard-rock tunnel engineering workflow" than MOOSE plus OpenFDEM, but very useful as cross-family generators.

## AE proxy boundary

- None of the open tools above should be described as "native laboratory AE waveform simulators" unless you have separately verified that exact workflow in your build.
- The safe claim is:
- They can generate fracture events, damage jumps, bond breaks, crack-surface increments, stress drops, or energy-release bursts.
- Those outputs can be postprocessed into `AE proxy` labels or event sequences.
- That is not the same thing as a validated sensor-level AE forward model.

## License and availability boundary

Do not collapse these states:

- `Installed`: a binary, source tree, or module exists.
- `Runnable`: dependencies resolve and a smoke test completes.
- `Usable`: required features are enabled in that build.
- `Licensed`: commercial entitlements are present and valid.
- `Reproducible`: the exact solver version, mesh pipeline, and postprocessing can be rerun by the team.

In this project, "not installed" also does not mean "not usable in principle"; it only means the current environment check did not prove availability. The inverse is also true: "installed" does not mean the solver is ready for production data generation.

## Recommended division of labor

### If the goal is a fast first dataset
- Use `MOOSE` to generate controlled brittle-fracture trajectories.
- Derive AE proxies in postprocessing.
- Reserve one small `OpenFDEM` campaign for sanity checking crack topology and fragment/contact realism.

### If the goal is the most realistic brittle tunnel failure dataset
- Use `OpenFDEM` as the main generator.
- Use `MOOSE` to sweep parameter space cheaply before expensive FDEM runs.
- Add `Peridigm` or `LAMMPS PERI` only if you need a third physics family for robustness checks.

## Original sources

1. MOOSE official site: <https://mooseframework.inl.gov/>
2. MOOSE license: <https://github.com/idaholab/moose/blob/master/LICENSE>
3. OpenFDEM official site: <https://openfdem.com/>
4. OpenFDEM copyrights/license page: <https://openfdem.com/rst_about_introduction/copyrights.html>
5. Peridigm official repository: <https://github.com/peridigm/peridigm>
6. LAMMPS PERI documentation: <https://docs.lammps.org/Howto_peri.html>
7. LAMMPS license: <https://github.com/lammps/lammps/blob/develop/LICENSE>
8. YADE documentation: <https://www.yade-dem.org/>
9. YADE acoustic emission page: <https://yade-dem.org/doc/acousticemissions.html>
10. PFC product page: <https://www.itascacg.com/software/pfc>
11. Abaqus product page: <https://www.3ds.com/products/simulia/abaqus>
