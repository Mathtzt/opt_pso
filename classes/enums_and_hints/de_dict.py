from typing import NamedTuple
from classes.enums_and_hints.optimizers_enum import OptimizersNames

class DEDict(NamedTuple):
    name: OptimizersNames
    dimensions: int
    population_size: int
    bounds: list[int, int]
    perc_mutation: float
    perc_crossover: float
    mut_strategy: str
