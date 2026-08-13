# Rendered by tunnelgeopt.fracture_crosscheck. Unresolved placeholders are fatal.
[Mesh]
  [canonical]
    type = FileMeshGenerator
    file = '@MESH_FILE@'
    allow_renumbering = false
  []
  parallel_type = REPLICATED
[]

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Physics]
  [SolidMechanics]
    [QuasiStatic]
      [same_problem]
        add_variables = true
        strain = SMALL
        planar_formulation = PLANE_STRAIN
        out_of_plane_direction = z
        save_in = 'resid_x resid_y'
      []
    []
  []
[]

[AuxVariables]
  [resid_x]
    order = FIRST
    family = LAGRANGE
  []
  [resid_y]
    order = FIRST
    family = LAGRANGE
  []
  [strain_xx]
    order = CONSTANT
    family = MONOMIAL
  []
  [strain_yy]
    order = CONSTANT
    family = MONOMIAL
  []
  [strain_xy]
    order = CONSTANT
    family = MONOMIAL
  []
  [stress_xx]
    order = CONSTANT
    family = MONOMIAL
  []
  [stress_yy]
    order = CONSTANT
    family = MONOMIAL
  []
  [stress_xy]
    order = CONSTANT
    family = MONOMIAL
  []
  [stress_zz]
    order = CONSTANT
    family = MONOMIAL
  []
  [energy_density]
    order = CONSTANT
    family = MONOMIAL
  []
[]

[Functions]
  [farfield_ux]
    type = ParsedFunction
    expression = '@UX_EXPRESSION@'
  []
  [farfield_uy]
    type = ParsedFunction
    expression = '@UY_EXPRESSION@'
  []
[]

[BCs]
  [farfield_ux]
    type = FunctionDirichletBC
    variable = disp_x
    boundary = farfield
    function = farfield_ux
  []
  [farfield_uy]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = farfield
    function = farfield_uy
  []
[]

[Materials]
  [elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = @YOUNG_MODULUS@
    poissons_ratio = @POISSON_RATIO@
  []
  [stress]
    type = ComputeLinearElasticStress
  []
[]

[AuxKernels]
  [strain_xx]
    type = RankTwoAux
    rank_two_tensor = mechanical_strain
    index_i = 0
    index_j = 0
    variable = strain_xx
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [strain_yy]
    type = RankTwoAux
    rank_two_tensor = mechanical_strain
    index_i = 1
    index_j = 1
    variable = strain_yy
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [strain_xy]
    type = RankTwoAux
    rank_two_tensor = mechanical_strain
    index_i = 0
    index_j = 1
    variable = strain_xy
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [stress_xx]
    type = RankTwoAux
    rank_two_tensor = stress
    index_i = 0
    index_j = 0
    variable = stress_xx
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [stress_yy]
    type = RankTwoAux
    rank_two_tensor = stress
    index_i = 1
    index_j = 1
    variable = stress_yy
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [stress_xy]
    type = RankTwoAux
    rank_two_tensor = stress
    index_i = 0
    index_j = 1
    variable = stress_xy
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [stress_zz]
    type = RankTwoAux
    rank_two_tensor = stress
    index_i = 2
    index_j = 2
    variable = stress_zz
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [energy_density]
    type = ElasticEnergyAux
    variable = energy_density
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[VectorPostprocessors]
  [nodes]
    type = NodalValueSampler
    sort_by = id
    variable = 'disp_x disp_y resid_x resid_y'
    execute_on = FINAL
  []
  [elements]
    type = ElementValueSampler
    sort_by = id
    variable = 'strain_xx strain_yy strain_xy stress_xx stress_yy stress_xy stress_zz energy_density'
    execute_on = FINAL
  []
[]

[Executioner]
  type = Steady
  solve_type = NEWTON
  petsc_options_iname = '-ksp_type -pc_type -pc_factor_mat_solver_type'
  petsc_options_value = 'preonly lu petsc'
  nl_abs_tol = 1e-8
  nl_rel_tol = 1e-12
  nl_max_its = 8
  l_tol = 1e-14
  l_max_its = 20
  [Quadrature]
    type = GAUSS
    order = THIRD
    element_order = THIRD
    side_order = THIRD
  []
[]

[Outputs]
  [csv]
    type = CSV
    file_base = '@OUTPUT_BASE@'
    precision = 17
    execute_on = FINAL
  []
[]

[Problem]
  type = FEProblem
  solve = true
[]
