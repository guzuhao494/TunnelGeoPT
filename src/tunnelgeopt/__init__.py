"""TunnelGeoPT: synthetic pre-training data for hard-rock tunnel mechanics."""

from .geometry import TunnelGeometry, make_tunnel_boundary
from .kirsch import kirsch_stress
from .lift import LiftedCase, generate_lifted_case
from .schema import GeoPTSample, load_sample, save_sample

__all__ = [
    "GeoPTSample",
    "LiftedCase",
    "TunnelGeometry",
    "generate_lifted_case",
    "kirsch_stress",
    "load_sample",
    "make_tunnel_boundary",
    "save_sample",
]

__version__ = "0.1.0"
