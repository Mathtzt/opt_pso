from classes.experiments import Experiments
from classes.enums_and_hints.experiments_dict import ExperimentsDict
from classes.enums_and_hints.problem_enum import ProblemFuncNames
from classes.enums_and_hints.optimizers_enum import OptimizersNames
from classes.enums_and_hints.pso_dict import PSODict

pso = PSODict(
    name = OptimizersNames.PSO,
    dimensions = 10,
    population_size = 10,
    bounds = [-100, 100],
    omega = .9,
    min_speed = -0.5,
    max_speed = 3.,
    cognitive_update_factor = 2.,
    social_update_factor = 2.,
    reduce_omega_linearly = True,
    reduction_speed_factor = .1
)

exp_dict = ExperimentsDict(
    name = 'exp',
    nexecucoes = 30,
    functions = [
        ProblemFuncNames.F1_BASIC
        ],
    optimizers = [
        pso
    ]
)

exp = Experiments(exp_dict = exp_dict)
exp.main()