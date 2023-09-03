from classes.enums_and_hints.problem_enum import ProblemFuncNames
from classes.enums_and_hints.optimizers_enum import OptimizersNames
from classes.enums_and_hints.experiments_dict import ExperimentsDict
from classes.enums_and_hints.pso_dict import PSODict
from classes.helper.utils import Utils

from classes.optimizers.pso import PSO

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
    
    def create_exp_dirs(self, problem_name: str):
        problem_path = Utils.create_folder(path = self.exp_path, name = problem_name)
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
    
    def main(self):

        for idx, optimizer in enumerate(self.exp_dict.optimizers):
            for func in self.exp_dict.functions:
                problem_path, imgs_path = self.create_exp_dirs(problem_name=func.value)
                for exec in range(self.exp_dict.nexecucoes):
                    opt = self.init_optimizer(optimizer)
                    opt.main(func_name = func, nexecucao = exec, exp_path = problem_path, imgs_path = imgs_path)