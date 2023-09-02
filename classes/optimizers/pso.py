import numpy as np
from classes.base.algoritmo import Algoritmo

from classes.enums_and_hints.problem_enum import ProblemFuncNames
from deap import base, creator, tools

class PSO(Algoritmo):
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
                 show_log: bool = True):
        super().__init__(dimensions, population_size, bounds)

        self.omega = omega
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.cognitive_update_factor = cognitive_update_factor
        self.social_update_factor = social_update_factor
        self.reduce_omega_linearly = reduce_omega_linearly
        self.show_log = show_log

        self.toolbox = base.Toolbox()

    def main(self, func_name: ProblemFuncNames, reset_classes: bool = False, nexecucao: int = 0, dirpath: str = './'):
        func_ = self.obter_func_objetivo(func_name = func_name)
        self.toolbox.register(alias = 'evaluate', 
                              function = func_[0])
        
        ## inicializações
        self.define_as_minimization_problem()
        self.creating_particle_class()
        self.create_particle()
        self.creating_particle_register()
        self.register_to_update_particles()

        ## criando a população
        population = self.toolbox.populationCreator(n = self.population_size)

        ## criando objeto para salvar as estatísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('min', np.min)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'evals'] + stats.fields

        ## loop
        best = None
        initial_omega = self.omega
        igeneration_stopped = 0
        nevaluations = 0
        finish_optimization = False

        best_fitness_history = []
        avg_fitness_history = []
        avg_euclidian_distance_history = []

        for idx, generation in enumerate(range(1, self.max_evaluations + 1)):
            # reduzindo omega linearmente
            if self.reduce_omega_linearly:
                self.omega = initial_omega - (idx * (initial_omega - 0.4) / (10000))
            
            # avaliar todas as partículas na população
            for particle in population:
                # calcular o valor de fitness da partícula / avaliação
                particle.fitness.values = (self.toolbox.evaluate(particle), )
                # atualizando melhor partícula global
                if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                    particle.best = creator.Particle(particle)
                    particle.best.fitness.values = particle.fitness.values
                # atualizando valor global
                if best is None or best.size == 0 or best.fitness < particle.fitness:
                    best = creator.Particle(particle)
                    best.fitness.values = particle.fitness.values
                # atualizando número de avaliações, verificando limites de tempo de otimização permitidos
                nevaluations += 1
                if abs(best.fitness.values[0] - func_[1]) < 10e-8 or nevaluations >= self.max_evaluations:
                    finish_optimization = True
                    igeneration_stopped = idx
                    break

            # atualizando velocidade e posição
            for particle in population:
                self.toolbox.update(particle, best)

            avg_euclidian_distance_history.append(self.calc_distancia_euclidiana_media_da_populacao(population))
            # salvando as estatísticas
            if generation == 1 or generation % 500 == 0:
                best_fitness_history.append(best.fitness.values)
                logbook.record(gen = generation,
                               evals = len(population),
                               **stats.compile(population))
                if self.show_log:
                    if self.reduce_omega_linearly:
                        print(logbook.stream + f" | omega = {self.omega}")
                    else:
                        print(logbook.stream)
            
            if finish_optimization:
                break

        avg_fitness_history = [logbook[i]['avg'] for i in range(len(logbook))]

        print("-- Melhor partícula = ", best)
        print("-- Melhor fitness = ", best.fitness.values[0])
        print("-- Geração que parou a otimização = ", igeneration_stopped)
        print("-- Qtd de vezes que saiu do espaço de busca = ", self.nout_bounds)

        if reset_classes:
            del creator.FitnessMin
            del creator.Particle

    def define_as_minimization_problem(self):
        creator.create(name = "FitnessMin",
                       base = base.Fitness,
                       weights = (-1., ))
        
    def creating_particle_class(self):
        creator.create(name = 'Particle',
                       base = np.ndarray,
                       fitness = creator.FitnessMin,
                       speed = None,
                       best = None)
        
    def create_particle(self):
        particle = creator.Particle(np.random.uniform(low = self.bounds[0],
                                                      high = self.bounds[1],
                                                      size = self.dimensions))
        
        particle.speed = np.random.uniform(low = self.min_speed,
                                           high = self.max_speed,
                                           size = self.dimensions)
        
        return particle

    def creating_particle_register(self):
        self.toolbox.register(alias = 'particleCreator',
                              function = self.create_particle)
                        
        self.toolbox.register('populationCreator', tools.initRepeat, list, self.toolbox.particleCreator)

    def update_particle(self, particle, best):
        local_update_factor = self.cognitive_update_factor * np.random.uniform(0, 1, particle.size)
        global_update_factor = self.social_update_factor * np.random.uniform(0, 1, particle.size)

        local_speed_update = local_update_factor * (particle.best - particle)
        global_speed_update = global_update_factor * (best - particle)

        particle.speed = (self.omega * particle.speed) + (local_speed_update + global_speed_update)
        # verificando se a nova posição sairá do espaço de busca. Se sim, ajustando para os limites.     
        out_bounds = False
        for i, speed in enumerate(particle.speed):
            if speed > self.max_speed:
                out_bounds = True
                particle.speed[i] = self.max_speed
            if speed < self.min_speed:
                out_bounds = True
                particle.speed[i] = self.min_speed
        
        if out_bounds:
            self.nout_bounds += 1

        # atualizando posição
        particle[:] = particle + particle.speed

    def register_to_update_particles(self):
        self.toolbox.register(alias = 'update',
                              function = self.update_particle)
        