import pandas as pd
import numpy as np
from classes.base.algoritmo import Algoritmo

from classes.enums_and_hints.problem_enum import ProblemFuncNames
from deap import base, creator, tools

class PSOR(Algoritmo):
    def __init__(self,
                 dimensions: int = 10,
                 population_size: int = 10,
                 bounds: list = [-100, 100],
                 omega: float = .9,
                 min_speed: float = -0.5,
                 max_speed: float = 3.,
                 cognitive_update_factor: float = 2.,
                 social_update_factor: float = 2.,
                 reduce_omega_linearly: bool = False,
                 nsubspaces: int = 4,
                 r_size: int = 20,
                 show_log: bool = True):
        assert population_size >= 1, "Quantidade de partículas deve ser superior ou igual a 1."
        assert population_size >= nsubspaces, "Quantidade de subespaços deve ser menor ou igual ao tamanho da população."

        super().__init__(dimensions, population_size, bounds)

        self.omega = omega
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.cognitive_update_factor = cognitive_update_factor
        self.social_update_factor = social_update_factor
        self.reduce_omega_linearly = reduce_omega_linearly
        self.nsubspaces = nsubspaces
        self.r_size = r_size
        self.show_log = show_log
        self.use_hypeshere_control = False                  # Tentativa do raio de controle [Não funcionou]

        self.nout_bounds = 0
        self.toolbox = base.Toolbox()

    def main(self, 
             func_name: ProblemFuncNames,
             nexecucao: int = 0, 
             exp_path: str = './',
             imgs_path: str = './'):
        func_ = self.obter_func_objetivo(func_name = func_name)
        self.toolbox.register(alias = 'evaluate', 
                              function = func_[0].evaluate)
        
        ## inicializações
        self.define_as_minimization_problem()
        ### 1. Criação dos subespaços ###
        subspaces = self.create_subspaces_bounds()
        ### 2. Criar subpopulaçoes ###
        self.creating_particle_class()
        subspaces_division_particles = self.distr_population()

        population = []
        for idx, subspace in enumerate(subspaces):
            subpop = self.create_particle(pbounds = subspace,
                                          nsize = subspaces_division_particles[idx],
                                          subpop = idx)
            population.append(subpop)

        self.register_to_update_particles()

        ## criando objeto para salvar as estatísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('min', np.min)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'evals'] + stats.fields

        ## loop
        ###
        best_pcentrals = [None] * self.nsubspaces
        best_subspaces_without_pcentral = [None] * self.nsubspaces
        best_subspaces_w_pcentral = [None] * self.nsubspaces
        best_subspace = [None] * self.nsubspaces
        ###
        best = None
        omega = self.omega
        igeneration_stopped = 0
        nevaluations = 0
        finish_optimization = False

        best_fitness_history = []
        avg_fitness_history = []
        avg_euclidian_distance_history = []

        counter = 0
        for idx, generation in enumerate(range(1, self.max_evaluations + 1)):
            # reduzindo omega linearmente
            if self.reduce_omega_linearly:
                omega = self.omega - (self.omega - 0.4) * idx / (self.max_evaluations / self.population_size)
                if counter == 2:
                    omega = 0.9
                    counter = 0
            
            ### 1. Iteração nas populações ###
            for isubpop, subpopulation in enumerate(population):
                improved = (None, False)
                # Avaliar todas as partículas na subpopulação
                for ipart, particle in enumerate(subpopulation): 
                    # calcular o valor de fitness da partícula / avaliação
                    particle.fitness.values = (self.toolbox.evaluate(particle), )
                    # atualizando melhor partícula global
                    if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                        particle.best = creator.Particle(particle)
                        particle.best.fitness.values = particle.fitness.values
                    # atualizando valor global
                    # if self.check_best_subgroup_particle(best_subspace[isubpop], particle):
                    #     best_subspace[isubpop] = creator.Particle(particle)
                    #     best_subspace[isubpop].fitness.values = particle.fitness.values
            
                    if self.check_best_subgroup_particle(best_subspaces_without_pcentral[isubpop], particle):
                        if not particle.is_pcentral:
                            best_subspaces_without_pcentral[isubpop] = creator.Particle(particle)
                            best_subspaces_without_pcentral[isubpop].fitness.values = particle.fitness.values

                            best_subspaces_w_pcentral[isubpop] = creator.Particle(particle)
                            best_subspaces_w_pcentral[isubpop].fitness.values = particle.fitness.values
                    if self.check_best_subgroup_particle(best_subspaces_w_pcentral[isubpop], particle):
                        if particle.is_pcentral:
                            best_subspaces_w_pcentral[isubpop] = creator.Particle(particle)
                            best_subspaces_w_pcentral[isubpop].fitness.values = particle.fitness.values
                    if best is None or best.size == 0 or best.fitness < particle.fitness:
                        best = creator.Particle(particle)
                        best.fitness.values = particle.fitness.values
                        best.speed = particle.speed
                        improved = (ipart, True)
                        if particle.is_pcentral:
                            best_pcentrals[isubpop] = creator.Particle(particle)
                            best_pcentrals[isubpop].fitness.values = particle.fitness.values

                    # atualizando número de avaliações, verificando limites de tempo de otimização permitidos
                    nevaluations += 1
                    stop_cond1 = abs(best.fitness.values[0] - func_[1]) < 10e-8
                    stop_cond2 = nevaluations >= self.max_evaluations
                    if stop_cond1 or stop_cond2:
                        finish_optimization = True
                        igeneration_stopped = idx
                        break

            # atualizando velocidade e posição
            for isubpop, subpopulation in enumerate(population):
                # Avaliar todas as partículas na subpopulação
                for ipart, particle in enumerate(subpopulation):
                    if improved[1] == True and improved[0] != ipart:
                        self.toolbox.update(particle, best, best_subspaces_w_pcentral[isubpop], subspaces[isubpop], omega, True)
                    else:
                        self.toolbox.update(particle, best, best_subspaces_without_pcentral[isubpop], subspaces[isubpop], omega, None)

            pop_list = [part for subspace in population for part in subspace]
            avg_euclidian_distance_history.append(self.calc_distancia_euclidiana_media_da_populacao(pop_list))
            # salvando as estatísticas
            if generation == 1 or generation % 500 == 0:
                best_fitness_history.append(best.fitness.values)
                pop_fitness_list = [part for subspace in population for part in subspace]
                if generation > 500 and best_fitness_history[-1] == best_fitness_history[-2]:
                    counter += 1
                logbook.record(gen = generation,
                               evals = self.population_size,
                               **stats.compile(pop_fitness_list))
                if self.show_log:
                    if self.reduce_omega_linearly:
                        print(logbook.stream + f" | omega = {omega}")
                    else:
                        print(logbook.stream)
            
            if finish_optimization:
                break

        avg_fitness_history = [logbook[i]['avg'] for i in range(len(logbook))]

        self.print_informacoes_gerais_optimizacao(best, igeneration_stopped, best_pcentrals)
        self.criar_grafico_evolucao_fitness(hist_best_fitness = best_fitness_history,
                                            hist_avg_fitness = avg_fitness_history,
                                            imgs_path = imgs_path, 
                                            img_name = f"pso_exec_{nexecucao}",
                                            use_log = True)
        self.criar_grafico_evolucao_distancia_media_pontos(hist_dist_pontos = avg_euclidian_distance_history,
                                                           imgs_path = imgs_path,
                                                           img_name = f"pso_distance_particles_{nexecucao}")
        self.criar_registro_geral(nexecucao = nexecucao,
                                  func_objetivo = func_name.value,
                                  best_particle = best,
                                  best_fitness = best.fitness.values[0],
                                  igeneration_stopped = igeneration_stopped,
                                  exp_path = exp_path)
        
        self.salvar_historico_em_arquivo_txt(best_fitness_history, exp_path, f'best_fitness_{nexecucao}')
        self.salvar_historico_em_arquivo_txt(avg_fitness_history, exp_path, f'avg_fitness_{nexecucao}')
        self.salvar_historico_em_arquivo_txt(avg_euclidian_distance_history, exp_path, f'dist_particles_{nexecucao}')

        del creator.FitnessMin
        del creator.Particle

    def check_best_subgroup_particle(self, bsubgroup_particle, particle):
        return bsubgroup_particle is None or \
               bsubgroup_particle.size == 0 or \
               bsubgroup_particle.fitness < particle.fitness

    def print_informacoes_gerais_optimizacao(self, best, igeneration_stopped, pcentrals):
        print("-- Melhor partícula = ", best)
        print("-- Melhor fitness = ", best.fitness.values[0])
        print("-- Geração que parou a otimização = ", igeneration_stopped)
        print("-- Qtd de vezes que saiu do espaço de busca = ", self.nout_bounds)
        print("-- Fitness das particulas centrais = ", [val.fitness.values for val in pcentrals if val is not None])

    def define_as_minimization_problem(self):
        creator.create(name = "FitnessMin",
                       base = base.Fitness,
                       weights = (-1., ))
        
    def creating_particle_class(self):
        creator.create(name = 'Particle',
                       base = np.ndarray,
                       fitness = creator.FitnessMin,
                       speed = None,
                       best = None,
                       subpop = int,
                       is_pcentral = bool)
        
    # def create_particle(self, pbounds: list, subpop: int, is_pcentral: bool):
    #     particle = creator.Particle(np.random.uniform(low = pbounds[0],
    #                                                   high = pbounds[1],
    #                                                   size = self.dimensions))
    #     speed = abs(pbounds[1] - pbounds[0]) // 2
    #     particle.speed = np.random.uniform(low = -speed,
    #                                        high = speed,
    #                                        size = self.dimensions)
    #     particle.subpop = subpop
    #     particle.is_pcentral = is_pcentral
        
    #     return particle

    def creating_particle_register(self):
        self.toolbox.register(alias = 'particleCreator',
                              function = self.create_particle)
                        
        self.toolbox.register('populationCreator', tools.initRepeat, list, self.toolbox.particleCreator)

    def up_speed_particle(self, particle, local_update_factor, global_update_factor, best_subspace_without_pcentral, omega):

        local_speed_update = local_update_factor * (particle.best - particle)
        global_speed_update = global_update_factor * (best_subspace_without_pcentral - particle)

        nspeed = (omega * particle.speed) + (local_speed_update + global_speed_update)

        return nspeed
    
    def up_speed_pcentral(self, particle, local_update_factor, global_update_factor, best, omega):
        local_speed_update = local_update_factor * (particle.best - particle)
        global_speed_update = global_update_factor * (best - particle)

        nspeed = (omega * particle.speed) + (local_speed_update + global_speed_update)

        return nspeed
    
    def up_speed_particle_if_new_optimum(self, particle, local_update_factor, global_update_factor, best_subspace_w_pcentral, omega):
        local_speed_update = local_update_factor * (particle.best - particle)
        global_speed_update = global_update_factor * (best_subspace_w_pcentral - particle)

        nspeed = (omega * particle.speed) + (local_speed_update + global_speed_update)

        return nspeed

    def update_particle(self, particle, best, best_subspace, bounds, omega, part_subspace_improved = None):
        local_update_factor = self.cognitive_update_factor * np.random.uniform(0, 1, particle.size)
        global_update_factor = self.social_update_factor * np.random.uniform(0, 1, particle.size)

        if particle.is_pcentral:
            particle.speed = self.up_speed_pcentral(particle, local_update_factor, global_update_factor, best, omega)
        
        if not particle.is_pcentral:
            if part_subspace_improved:
                particle.speed = self.up_speed_particle_if_new_optimum(particle, local_update_factor, global_update_factor, best_subspace, omega)
            else:
                particle.speed = self.up_speed_particle(particle, local_update_factor, global_update_factor, best_subspace, omega)
        
        out_bounds = False
        for i, speed in enumerate(particle.speed):
            if speed > self.max_speed:
                out_bounds = True
                particle.speed[i] = best.speed.max()
            if speed < self.min_speed:
                out_bounds = True
                particle.speed[i] = best.speed.min()
        
        if out_bounds:
            self.nout_bounds += 1

        # atualizando posição
        particle[:] = particle + particle.speed

        for i, part in enumerate(particle):
            if part > self.bounds[1]:
                particle[i] = self.bounds[1] - (self.bounds[1] * 0.1)
            if part < self.bounds[0]:
                particle[i] = self.bounds[0] - (self.bounds[0] * 0.1)
        # atualizando posição
         #+ particle_adjusted if not self.use_hypeshere_control else particle_adjusted
        
        # local_speed_update = local_update_factor * (particle.best - particle)
        # global_speed_update = global_update_factor * (best - particle)

        # particle.speed = (omega * particle.speed) + (local_speed_update + global_speed_update)

        # if self.use_hypeshere_control:
        #     particle_adjusted = self.adjust_with_hypersphere(particle, particle.speed, r = 250)
        # else:
        #     particle_adjusted = particle.speed


    def adjust_with_hypersphere(self, original_particle, particle_speed, r):
        updated_particle = original_particle + particle_speed #
        # Calcula a distância entre o ponto original e a nova posição
        distancia = np.linalg.norm(updated_particle - original_particle)

        # Verifica se a distância é maior que o raio
        if distancia > r:
            # Normaliza o vetor e multiplica pelo raio
            vetor_normalizado = (updated_particle - original_particle) / distancia
            particle_adjusted = original_particle + vetor_normalizado * r
            return particle_adjusted
        else:
            return particle_speed

    def register_to_update_particles(self):
        self.toolbox.register(alias = 'update',
                              function = self.update_particle)
    
    def generate_samples(self, min_bound: int, max_bound: int, dim: int, r_size: int, nsize: int):
        tamanho_amostra_desejada = nsize
        raio = r_size  # Ajuste o raio conforme necessário
        # Inicialize a lista para armazenar as amostras
        amostra = []

        # Função para calcular a distância entre duas amostras
        def distancia_entre_amostras(amostra1, amostra2):
            return np.linalg.norm(amostra1 - amostra2)

        # Função para verificar se uma amostra está a uma distância maior que x de todas as amostras existentes
        def e_valida(amostra_nova, amostra_existente):
            for amostra_antiga in amostra_existente:
                if distancia_entre_amostras(amostra_nova, amostra_antiga) < raio:
                    return False
            return True

        # Loop até atingir o tamanho da amostra desejada
        while len(amostra) < tamanho_amostra_desejada:
            nova_amostra = np.random.uniform(min_bound, max_bound, dim)  # Gere uma nova amostra
            if e_valida(nova_amostra, amostra):  # Verifique se a nova amostra é válida
                amostra.append(nova_amostra)  # Adicione a nova amostra à amostra

        # Converta a lista de amostras em um array NumPy
        amostra = np.array(amostra)

        return amostra
        
    def create_particle(self, 
                        pbounds, 
                        nsize,
                        subpop):
        psamples_ = self.generate_samples(min_bound = pbounds[0], 
                                          max_bound = pbounds[1], 
                                          dim = self.dimensions, 
                                          r_size = self.r_size, 
                                          nsize = nsize)
        
        particles = []
        for idx, particle in enumerate(psamples_):
            if idx == 0:
                p = creator.Particle(particle)
                p.is_pcentral = True
            else:
                p = creator.Particle(particle)
                p.is_pcentral = False
                
            p.speed = np.random.uniform(low = self.min_speed,
                                        high = self.max_speed,
                                        size = self.dimensions)
            p.subpop = subpop

            particles.append(p)

        return particles
    
    def create_subspaces_bounds(self):
        tamanho_intervalos = (self.bounds[1] - self.bounds[0]) / self.nsubspaces
        intervalos_ = []
        for i in range(self.nsubspaces):
            inicio = self.bounds[0] + i * tamanho_intervalos
            fim = inicio + tamanho_intervalos
            intervalos_.append((inicio, fim))

        return [(self.bounds[1], self.bounds[0])] * self.nsubspaces #intervalos_
    
    def distr_population(self):
        s = self.population_size//self.nsubspaces
        l_ = [s] * self.nsubspaces

        for idx, val in enumerate(l_):
            if sum(l_) + 1 <= 10:
                l_[idx] = val + 1
            else:
                continue
        
        return l_
        
    def criar_registro_geral(self, 
                             nexecucao: int, 
                             func_objetivo: str, 
                             best_particle: list,
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
            'omega': self.omega,
            'reduce_omega_linearly': self.reduce_omega_linearly,
            'range_speed': [self.min_speed, self.max_speed],
            'cognitive_factor': self.cognitive_update_factor,
            'social_factor': self.social_update_factor,
            'best_particle': best_particle,
            'best_fitness': best_fitness,
            'out_bounds': self.nout_bounds
        }

        self.salvar_registro_geral(registro = d,
                                   exp_path = exp_path)
