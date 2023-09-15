from classes.experiments import Experiments
from classes.enums_and_hints.experiments_dict import ExperimentsDict
from classes.enums_and_hints.problem_enum import ProblemFuncNames
from classes.enums_and_hints.optimizers_enum import OptimizersNames
from classes.enums_and_hints.params_enum import DEMultStrategyNames, GAParentSelectionNames, GACrossoverNames, GAMutationNames
from classes.enums_and_hints.ga_dict import GADict
from classes.enums_and_hints.pso_dict import PSODict
from classes.enums_and_hints.de_dict import DEDict
from classes.enums_and_hints.psor_dict import PSORDict

pso = PSODict(
    name = OptimizersNames.PSO,
    dimensions = 10,
    population_size = 10,
    bounds = [-100, 100],
    omega = .9,
    min_speed = -50,
    max_speed = 50,
    cognitive_update_factor = 2.,
    social_update_factor = 2.,
    reduce_omega_linearly = True,
    reduction_speed_factor = .1
)

de = DEDict(
    name = OptimizersNames.DE,
    dimensions = 10,
    population_size = 10,
    bounds = [-100, 100],
    perc_mutation = .8,
    perc_crossover = .4,
    mut_strategy = DEMultStrategyNames.RANDTOBEST1
)

ga = GADict(
    name = OptimizersNames.GA,
    dimensions = 10,
    population_size = 50,
    bounds = [-100, 100],
    total_pais_cruzamento = 2,
    tipo_selecao_pais = GAParentSelectionNames.ROULETTE_WHEEL_SELECTION, 
    total_pais_torneio = 3,
    tipo_cruzamento = GACrossoverNames.SINGLE_POINT,
    taxa_cruzamento = .9,
    tipo_mutacao = GAMutationNames.SWAP,
    taxa_mutacao = .05,
    elitismo = 1
)

psor = PSORDict(
    name = OptimizersNames.PSOR,
    dimensions = 10,
    population_size = 20,
    bounds = [-100, 100],
    omega = .9,
    min_speed = -50,
    max_speed = 50,
    cognitive_update_factor = 2.,
    social_update_factor = 2.,
    reduce_omega_linearly = True,
    nsubspaces = 2,
    r_size = 50
)

exp_dict = ExperimentsDict(
    name = 'f_psor_10d',
    nexecucoes = 30,
    functions = [
        # ProblemFuncNames.F1,
        # ProblemFuncNames.F2,
        # ProblemFuncNames.F4,
        # ProblemFuncNames.F6,
        ProblemFuncNames.F7,
        ProblemFuncNames.F9,
        ProblemFuncNames.F14
        ],
    optimizers = [
        # ga,
        # pso,
        # de,
        psor
    ]
)

exp = Experiments(exp_dict = exp_dict)
exp.main()