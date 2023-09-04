import numpy as np

from classes.base.algoritmo import Algoritmo
from classes.enums_and_hints.problem_enum import ProblemFuncNames
from classes.enums_and_hints.params_enum import DEMultStrategyNames

class DE(Algoritmo):
    def __init__(self,
                 dimensions: int = 10,
                 population_size: int = 10,
                 bounds: list = [-100, 100],
                 perc_mutation: float = .9,
                 perc_crossover: float = .4,
                 mut_strategy: DEMultStrategyNames = DEMultStrategyNames.RAND1,
                 show_log: bool = True):
        super().__init__(dimensions, population_size, bounds)

        self.perc_mutation = perc_mutation
        self.perc_crossover = perc_crossover
        self.mut_strategy = mut_strategy
        self.nout_bounds = 0
        self.show_log = show_log

    def main(self, 
            func_name: ProblemFuncNames,
            nexecucao: int = 0, 
            exp_path: str = './',
            imgs_path: str = './'):
        
        func_ = self.obter_func_objetivo(func_name = func_name)
        # inicilizando população
        population = self.init_population()
        # avaliando a população inicial de soluções candidatas
        fitness = [func_[0].evaluate(ind) for ind in population]
        # guardando informações do melhor vetor
        best_vector = population[np.argmin(fitness)]
        best_fitness = min(fitness)
        prev_fitness = best_fitness
        ## loop ##
        nevaluations = 1
        igeneration_stopped = 0
        finish_optimization = False

        best_fitness_history = []
        avg_fitness_history = []
        avg_euclidian_distance_history = []

        # inicializando loop das gerações
        for i in range(1, self.max_evaluations + 1):
            # iteração de todas as soluções candidatas da população
            for j in range(self.population_size):
                # escolhendo três candidatos (a, b e c), que não sejam o atual
                candidates = [candidate for candidate in range(self.population_size) if candidate != j]
                a, b, c = population[np.random.choice(candidates, 3, replace = False)]
                # realizando processo de mutação
                mutated = self.mutation([a, b, c], best_vector)
                # verificando se vetor que sofreu a mutação saiu do espaço de busca. Se sim, aplica correção
                mutated = self.check_bounds(mutated)
                # realizando crossover
                vtrial = self.crossover(mutated, population[j])
                # calculando o valor da função objetivo para o vetor alvo
                fitness_target = func_[0].evaluate(population[j])
                # calculando o valor da função objetivo para o vetor candidato escolhido
                fitness_trial = func_[0].evaluate(vtrial)
                # realizando seleção
                if fitness_trial < fitness_target:
                    # substituindo o individuo da população pelo novo vetor
                    population[j] = vtrial
                    # armazenando o novo valor da função objetivo
                    fitness[j] = fitness_trial
                
                # atualizando número de avaliações, verificando limites de tempo de otimização permitidos
                nevaluations += 2
                stop_cond1 = abs(best_fitness - func_[1]) < 10e-8
                stop_cond2 = nevaluations >= self.max_evaluations
                if stop_cond1 or stop_cond2:
                    if stop_cond1:
                        best_fitness = 0.0
                    finish_optimization = True
                    igeneration_stopped = i
                    break
                
            # encontrando o vetor com melhor desempenho em cada iteração
            best_fitness = np.min(fitness)
            avg_fitness = np.mean(fitness)
            std_fitness = np.std(fitness)
            # armazenando o valor mais baixo da função objetivo (problema de minimização)
            if best_fitness < prev_fitness:
                best_vector = population[np.argmin(fitness)]
                prev_fitness = best_fitness

            # printando progresso
            if self.show_log and i % 500 == 0:
                print(f'Gen {i} | Min {best_fitness} | Avg {avg_fitness} | Std {std_fitness}')
                #%d:  %s = %f' % (i, list(np.around(best_vector, decimals=5)), best_fitness))

            # guardando informações para registro da otimização
            if i == 1 or i % 500 == 0:
                best_fitness_history.append(best_fitness) if best_fitness < prev_fitness else best_fitness_history.append(prev_fitness)
                avg_fitness_history.append(avg_fitness)
            avg_euclidian_distance_history.append(self.calc_distancia_euclidiana_media_da_populacao(population))

            if finish_optimization:
                break

        self.print_informacoes_gerais_optimizacao(best_vector, best_fitness, igeneration_stopped)
        self.criar_grafico_evolucao_fitness(hist_best_fitness = best_fitness_history,
                                            hist_avg_fitness = avg_fitness_history,
                                            imgs_path = imgs_path, 
                                            img_name = f"de_exec_{nexecucao}")
        self.criar_grafico_evolucao_distancia_media_pontos(hist_dist_pontos = avg_euclidian_distance_history,
                                                           imgs_path = imgs_path,
                                                           img_name = f"de_distance_particles_{nexecucao}")
        self.criar_registro_geral(nexecucao = nexecucao,
                                  func_objetivo = func_name.value,
                                  best_ind = best_vector,
                                  best_fitness = best_fitness,
                                  igeneration_stopped = igeneration_stopped,
                                  exp_path = exp_path)

    def print_informacoes_gerais_optimizacao(self, best_vector, best_fitness, igeneration_stopped):
        print("-- Melhor individuo = ", best_vector)
        print("-- Melhor fitness = ", best_fitness)
        print("-- Geração que parou a otimização = ", igeneration_stopped)
        print("-- Qtd de vezes que saiu do espaço de busca = ", self.nout_bounds)

    def mutation(self, X, best):
        x_base = X[0]
        x_r2 = X[1]
        x_r3 = X[2]
        
        if self.mut_strategy.value == DEMultStrategyNames.RAND1.value:
            return x_base + self.perc_mutation * (x_r2 - x_r3)
        elif self.mut_strategy.value == DEMultStrategyNames.RANDTOBEST1.value:
            return x_base + self.perc_mutation * (x_r2 - x_r3) + self.perc_mutation * (best - x_base)
    
    def get_bounds_of_all_dim(self):
        # gera vetor de limites por dimensão do problema
        return np.array([self.bounds] * self.dimensions)
    
    def check_bounds(self, mutated):
        # verificando se vetor de mutação passou dos limites do espaço de busca. Se sim, altera valor para o limite mais próximo.
        nout_bounds = False
        for idx, val in enumerate(self.get_bounds_of_all_dim()):
            if mutated[idx] < val[0]:
                mutated[idx] = val[0]
                nout_bounds = True
            if mutated[idx] > val[1]:
                mutated[idx] = val[1]
                nout_bounds = True
        
        if nout_bounds:
            self.nout_bounds += 1
            
        return mutated
    
    def crossover(self, mutated, target):
        p = np.random.rand(self.dimensions)

        vtrial = [mutated[i] if p[i] < self.perc_crossover else target[i] for i in range(self.dimensions)]
        
        return vtrial
    
    def init_population(self):
        # buscando vetor de limites x dim
        all_bounds = self.get_bounds_of_all_dim()
        # inicializando a população de soluções candidatas de forma aleatória dentro dos limites especificados.
        pop = all_bounds[:, 0] + (np.random.rand(self.population_size, len(all_bounds)) * (all_bounds[:, 1] - all_bounds[:, 0]))
        return pop

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
            'mutation_strategy': self.mut_strategy.value,
            'mutation_perc': self.perc_mutation,
            'crossover_perc': self.perc_crossover,
            'best_ind': best_ind,
            'best_fitness': best_fitness,
            'out_bounds': self.nout_bounds
        }

        self.salvar_registro_geral(registro = d,
                                   exp_path = exp_path)