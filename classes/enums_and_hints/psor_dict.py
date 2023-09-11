from typing import NamedTuple
from .optimizers_enum import OptimizersNames

class PSORDict(NamedTuple):
    name: OptimizersNames
    dimensions: int
    population_size: int
    bounds: list[int, int]
    omega: float
    min_speed: float
    max_speed: float
    cognitive_update_factor: float
    social_update_factor: float
    reduce_omega_linearly: bool
    nsubspaces: int
    r_size: int