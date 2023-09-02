from typing import NamedTuple, List, Union
from .problem_enum import ProblemFuncNames
from .pso_dict import PSODict
from .de_dict import DEDict

class ExperimentsDict(NamedTuple):
    name: str
    nexecucoes: int
    functions: list[ProblemFuncNames]
    optimizers: List[Union[PSODict, DEDict]]