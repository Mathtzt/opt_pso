import os
import time
import pygad as pga
from classes.base.algoritmo import Algoritmo
from classes.enums_and_hints.problem_enum import ProblemFuncNames
from classes.enums_and_hints.params_enum import GAParentSelectionNames, GACrossoverNames, GAMutationNames

class GA(Algoritmo, pga.GA):
    def __init__(self,
                 dimensions: int = 10,
                 population_size: int = 10,
                 bounds: list = [-100, 100],
                 total_pais_cruzamento: int = 2,
                 tipo_selecao_pais: GAParentSelectionNames = GAParentSelectionNames.ROULETTE_WHEEL_SELECTION,
                 total_pais_torneio: int = 3,
                 tipo_cruzamento: GACrossoverNames = GACrossoverNames.SINGLE_POINT,
                 taxa_cruzamento: float = .9,
                 tipo_mutacao: GAMutationNames = GAMutationNames.RANDOM,
                 taxa_mutacao: float = .05,
                 elitismo: int = 1,
                 show_log: bool = True):
        super().__init__(dimensions, population_size, bounds)

        self.total_pais_cruzamento = total_pais_cruzamento
        self.tipo_selecao_pais = tipo_selecao_pais.value
        self.total_pais_torneio = total_pais_torneio
        self.tipo_cruzamento = tipo_cruzamento.value
        self.taxa_cruzamento = taxa_cruzamento
        self.tipo_mutacao = tipo_mutacao.value
        self.taxa_mutacao = taxa_mutacao
        self.elitismo = elitismo
        
        self.total_pais_manter_populacao = -1
        self.salvar_melhores_solucoes = True
        self.salvar_todas_solucoes = False
        self.nout_bounds = 0
        self.show_log = show_log

        self.function = None

    def main(self, 
             func_name: ProblemFuncNames,
             nexecucao: int = 0, 
             exp_path: str = './',
             imgs_path: str = './'):
        
        self.function = self.obter_func_objetivo(func_name = func_name)

        pga.GA.__init__(self = self,
                        num_generations = self.max_evaluations,
                        sol_per_pop = self.population_size,
                        fitness_func = self.avaliar,
                        num_genes = self.dimensions,
                        gene_type = [float] * self.dimensions,
                        allow_duplicate_genes = True,
                        init_range_low = self.bounds[0],
                        init_range_high = self.bounds[1],
                        num_parents_mating = self.total_pais_cruzamento,
                        parent_selection_type = self.tipo_selecao_pais,
                        K_tournament = self.total_pais_torneio,
                        keep_parents = self.total_pais_manter_populacao,
                        crossover_type = self.tipo_cruzamento,
                        crossover_probability = self.taxa_cruzamento,
                        mutation_type = self.tipo_mutacao,
                        mutation_probability = self.taxa_mutacao,
                        mutation_by_replacement = True,
                        mutation_percent_genes = 5,
                        random_mutation_min_val = self.bounds[0],
                        random_mutation_max_val = self.bounds[1],
                        keep_elitism = self.elitismo,
                        stop_criteria = ["saturate_" + str(self.max_evaluations)],
                        random_seed = nexecucao,
                        delay_after_gen = 0.0,
                        save_best_solutions = self.salvar_melhores_solucoes,
                        save_solutions = self.salvar_todas_solucoes,
                        suppress_warnings = False)
        
        if self.valid_parameters:
            print("\nExecução: " + str(nexecucao + 1))
            tempo_execucao = time.time()
            self.run()
            tempo_execucao = time.time() - tempo_execucao

            if self.run_completed:
                self.print_informacoes_gerais_optimizacao(tempo_execucao)
                self.plot_fitness(save_dir = os.path.join(exp_path, './imgs'))
                os.rename(f"{exp_path}/imgs.png", f"{imgs_path}/ga_exec_{nexecucao}.png")
                # self.criar_grafico_evolucao_fitness(hist_best_fitness = best_solutions,
                #                                     hist_avg_fitness = avg_fitness_history,
                #                                     imgs_path = imgs_path, 
                #                                     img_name = f"de_exec_{nexecucao}")
                # self.criar_grafico_evolucao_distancia_media_pontos(hist_dist_pontos = avg_euclidian_distance_history,
                #                                                 imgs_path = imgs_path,
                #                                                 img_name = f"de_distance_particles_{nexecucao}")
                self.criar_registro_geral(nexecucao = nexecucao,
                                          func_objetivo = func_name.value,
                                          best_ind = self.best_solutions[-1],
                                          best_fitness = self.best_solutions_fitness[-1],
                                          igeneration_stopped = self.best_solution_generation,
                                          exp_path = exp_path)

    def print_informacoes_gerais_optimizacao(self, tempo_execucao):
        print("Geração Melhor Solução: " + str(self.best_solution_generation))
        print("Melhor Solução........: " + str(self.best_solutions[-1]))
        print("Melhor Avaliação......: " + str(self.best_solutions_fitness[-1]))
        print("Tempo Execução........: " + "%.2f" % tempo_execucao + " seg")

    def avaliar(self, arg0, x, arg1):
        aux = [x[i] for i in range(self.dimensions)]
        nout_bounds = False
        for i in range(self.dimensions):
            if aux[i] < self.bounds[0]:
                aux[i] = self.bounds[0]
                nout_bounds = True
            elif aux[i] > self.bounds[1]:
                aux[i] = self.bounds[1]
                nout_bounds = True
        
        if nout_bounds:
            self.nout_bounds += 1

        return 1. / abs(self.function[0].evaluate(aux) - self.function[1])
    
    def criar_registro_geral(self, 
                             nexecucao: int, 
                             func_objetivo: str, 
                             best_ind: list,
                             best_fitness: list,
                             igeneration_stopped: int,
                             exp_path: str):
        d = {
            'execucao': nexecucao,
            'funcao_objetivo': func_objetivo,
            'dimensoes': self.dimensions,
            'tamanho_populacao': self.population_size,
            'total_geracoes_realizadas': igeneration_stopped,
            'range_position': self.bounds,
            'total_pais_cruzamento': self.total_pais_cruzamento,
            'tipo_selecao_pais': self.tipo_selecao_pais,
            'total_pais_torneio': self.total_pais_torneio,
            'tipo_cruzamento': self.tipo_cruzamento,
            'taxa_cruzamento': self.taxa_cruzamento,
            'tipo_mutacao': self.tipo_mutacao,
            'taxa_mutacao': self.taxa_mutacao,
            'elitismo': self.elitismo,
            'best_ind': best_ind,
            'best_fitness': best_fitness,
            'out_bounds': self.nout_bounds
        }

        self.salvar_registro_geral(registro = d,
                                   exp_path = exp_path)