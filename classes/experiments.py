from classes.enums_and_hints.problem_enum import ProblemFuncNames
from classes.enums_and_hints.optimizers_enum import OptimizersNames
from classes.enums_and_hints.experiments_dict import ExperimentsDict
from classes.enums_and_hints.ga_dict import GADict
from classes.enums_and_hints.pso_dict import PSODict
from classes.enums_and_hints.de_dict import DEDict
from classes.enums_and_hints.psor_dict import PSORDict
from classes.helper.utils import Utils

from classes.optimizers.ga import GA
from classes.optimizers.pso import PSO
from classes.optimizers.de import DE
from classes.optimizers.pso_r import PSOR

class Experiments:
    def __init__(self,
                 exp_dict: ExperimentsDict,
                 base_dir: str = './') -> None:
        
        self.exp_dict: ExperimentsDict = exp_dict

        self.base_dir = base_dir
        self.root_path, self.exp_path = self.create_dirs()
        

    def create_dirs(self):
        root_path = Utils.create_folder(path = self.base_dir, name = "results")
        exp_path = Utils.create_folder(path = root_path, name = self.exp_dict.name, use_date = True)

        return root_path, exp_path
    
    def create_opt_dir(self, optimizer_name: str):
        return Utils.create_folder(path = self.exp_path, name = optimizer_name)
    
    def create_exp_dirs(self, path: str, problem_name: str):
        problem_path = Utils.create_folder(path = path, name = problem_name)
        imgs_path = Utils.create_folder(path = problem_path, name = 'imgs')

        return problem_path, imgs_path
    
    def init_optimizer(self, optimizer_dict):
        if isinstance(optimizer_dict, PSODict):
            opt: PSODict = optimizer_dict

            pso = PSO(dimensions = opt.dimensions,
                      population_size = opt.population_size,
                      bounds = opt.bounds,
                      omega = opt.omega,
                      min_speed = opt.min_speed,
                      max_speed = opt.max_speed,
                      cognitive_update_factor = opt.cognitive_update_factor,
                      social_update_factor = opt.social_update_factor,
                      reduce_omega_linearly = opt.reduce_omega_linearly,
                      reduction_speed_factor = opt.reduction_speed_factor)
            
            return pso
        
        if isinstance(optimizer_dict, DEDict):
            opt: DEDict = optimizer_dict

            de = DE(dimensions = opt.dimensions,
                    population_size = opt.population_size,
                    bounds = opt.bounds,
                    perc_mutation = opt.perc_mutation,
                    perc_crossover = opt.perc_crossover,
                    mut_strategy = opt.mut_strategy)
            
            return de
        
        if isinstance(optimizer_dict, GADict):
            opt: GADict = optimizer_dict

            ga = GA(dimensions = opt.dimensions,
                    population_size = opt.population_size,
                    bounds = opt.bounds,
                    total_pais_cruzamento = opt.total_pais_cruzamento,
                    tipo_selecao_pais = opt.tipo_selecao_pais,
                    total_pais_torneio = opt.total_pais_torneio,
                    tipo_cruzamento = opt.tipo_cruzamento,
                    taxa_cruzamento = opt.taxa_cruzamento,
                    tipo_mutacao = opt.tipo_mutacao,
                    taxa_mutacao = opt.taxa_mutacao,
                    elitismo = opt.elitismo)
            
            return ga
        
        if isinstance(optimizer_dict, PSORDict):
            opt: PSORDict = optimizer_dict

            psor = PSOR(dimensions = opt.dimensions,
                      population_size = opt.population_size,
                      bounds = opt.bounds,
                      omega = opt.omega,
                      min_speed = opt.min_speed,
                      max_speed = opt.max_speed,
                      cognitive_update_factor = opt.cognitive_update_factor,
                      social_update_factor = opt.social_update_factor,
                      reduce_omega_linearly = opt.reduce_omega_linearly,
                      nsubspaces = opt.nsubspaces,
                      r_size = opt.r_size)
            
            return psor
    
    def main(self):

        for optimizer in self.exp_dict.optimizers:
            print(f"#### Algoritmo {optimizer.name.value} ####")
            opt_path = self.create_opt_dir(optimizer_name = optimizer.name.value)
            for func in self.exp_dict.functions:
                print(f"#### Função {func.value} ####")
                problem_path, imgs_path = self.create_exp_dirs(path = opt_path, problem_name=func.value)
                for exec in range(self.exp_dict.nexecucoes):
                    print(f"#### Execução {exec} ####")
                    opt = self.init_optimizer(optimizer)
                    opt.main(func_name = func, nexecucao = exec, exp_path = problem_path, imgs_path = imgs_path)