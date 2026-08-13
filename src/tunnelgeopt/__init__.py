"""TunnelGeoPT: synthetic pre-training data for hard-rock tunnel mechanics."""

from .cases import (
    build_case_manifest,
    case_group_id,
    freeze_case_splits,
    load_case_manifest,
    verify_case_manifest,
    write_case_manifest,
)
from .elastic_schema import (
    ElasticRecord,
    load_elastic_record,
    save_elastic_record,
    save_elastic_result,
)
from .elastic_validation import kirsch_metrics, run_affine_patch_test, validate_elastic_result
from .elasticity import ElasticResult, solve_plane_strain_excavation
from .geometry import TunnelGeometry, make_tunnel_boundary
from .kirsch import kirsch_stress
from .lift import LiftedCase, generate_lifted_case
from .mesh import TunnelMesh, generate_tunnel_mesh
from .schema import GeoPTSample, load_sample, save_sample

__all__ = [
    "ElasticRecord",
    "ElasticResult",
    "GeoPTSample",
    "LiftedCase",
    "TunnelGeometry",
    "TunnelMesh",
    "build_case_manifest",
    "case_group_id",
    "freeze_case_splits",
    "generate_lifted_case",
    "generate_tunnel_mesh",
    "kirsch_metrics",
    "kirsch_stress",
    "load_case_manifest",
    "load_elastic_record",
    "load_sample",
    "make_tunnel_boundary",
    "run_affine_patch_test",
    "save_elastic_record",
    "save_elastic_result",
    "save_sample",
    "solve_plane_strain_excavation",
    "validate_elastic_result",
    "verify_case_manifest",
    "write_case_manifest",
]

__version__ = "0.2.0"
