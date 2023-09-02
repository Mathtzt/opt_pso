from typing import NamedTuple

class DEDict(NamedTuple):
    dimensions: int
    population_size: int
    bounds: list[int, int]
    omega: float
    min_speed: float
    max_speed: float
    cognitive_update_factor: float
    social_update_factor: float
    reduce_omega_linearly: bool