from typing import NamedTuple
from classes.enums_and_hints.optimizers_enum import OptimizersNames
from classes.enums_and_hints.params_enum import GACrossoverNames, GAMutationNames, GAParentSelectionNames

class GADict(NamedTuple):
    name: OptimizersNames
    dimensions: int
    population_size: int
    bounds: list[int, int]
    total_pais_cruzamento: int
    tipo_selecao_pais: GAParentSelectionNames
    total_pais_torneio: int
    tipo_cruzamento: GACrossoverNames
    taxa_cruzamento: float
    tipo_mutacao: GAMutationNames
    taxa_mutacao: float
    elitismo: int