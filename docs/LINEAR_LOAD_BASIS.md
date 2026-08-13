# Nine-channel linear stress-response basis

The current B layer is a homogeneous, small-strain, plane-strain elastic
boundary-value problem. For one fixed tunnel geometry, mesh and query grid,
its in-plane stress response is linear in the prescribed far-field tensor.
`load_basis.py` represents that response as a `3 x 3` map at every query point:

```text
[sigma_yy, sigma_zz, tau_yz]_query
    = B(y, z, geometry) @ [Sigma_yy, Sigma_zz, Tau_yz]_farfield
```

The flattened `B` tensor has nine channels per point. Three linearly
independent high-fidelity load solves determine it; every additional load on
the same fixed geometry is a matrix multiplication rather than a new solve.
The representation is compatible with TunnelGeoPT's current normalized data:
dividing each load and its response by the same prescribed far-field norm does
not change `B`.

This changes the useful learning target. Instead of asking a network to
rediscover load superposition from many random load cases, a geometry model can
predict the nine-channel response basis. Load composition then remains an
exact, auditable physics layer.

For generation, use the three tensor-Frobenius-unit normalized load vectors
`[1,0,0]`, `[0,1,0]`, and `[0,0,1/sqrt(2)]`. The `1/sqrt(2)` factor follows
the symmetric-tensor norm `sqrt(yy^2 + zz^2 + 2 yz^2)`. Their design matrix
has rank three and condition number `sqrt(2)`. This avoids the occasional severe conditioning seen
when three random loads happen to be nearly dependent. The current seen-data
leave-one-load-out audit deliberately used the existing random triples as a
harder numerical check; production data should not repeat that avoidable
conditioning risk.

The guarantee stops at the current linear-elastic solver. Damage, contact,
plasticity, crack growth and rockburst dynamics are nonlinear or path
dependent and require residual/state-evolution models and new evidence.
