import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from classes.helper.utils import Utils
from classes.enums_and_hints.problem_enum import ProblemFuncNames, ProblemFuncOptValue

import opfunu
# from opfunu.cec.cec2014.function import F1, F2, F4, F6, F7, F8, F9, F14

class Algoritmo(ABC):

    def __init__(self, 
                 dimensions: int,
                 population_size: int,
                 bounds: list
                 ):
        
        self.dimensions = dimensions
        self.population_size = population_size
        self.bounds = bounds
        
        self.max_evaluations = 10000 * dimensions
        self.nout_bounds = 0
    
    @abstractmethod
    def main(self, 
             func_name: ProblemFuncNames, 
             nexecucao: int,
             dirpath: str,
             imgs_path: str):
        pass
            
    def obter_func_objetivo(self, func_name: ProblemFuncNames):
        if func_name == ProblemFuncNames.F1_BASIC:
            raise NotImplementedError("Essa função está em processo de refatoração.")
            # selected_func = (self.f1_basic, ProblemFuncOptValue.F1_BASIC.value)
        elif func_name == ProblemFuncNames.F1:
            selected_func = (opfunu.cec_based.F12014(ndim=self.dimensions), ProblemFuncOptValue.F1.value)
        elif func_name == ProblemFuncNames.F2:
            selected_func = (opfunu.cec_based.F22014(ndim=self.dimensions), ProblemFuncOptValue.F2.value)
        elif func_name == ProblemFuncNames.F4:
            selected_func = (opfunu.cec_based.F42014(ndim=self.dimensions), ProblemFuncOptValue.F4.value)
        elif func_name == ProblemFuncNames.F6:
            selected_func = (opfunu.cec_based.F62014(ndim=self.dimensions), ProblemFuncOptValue.F6.value)
        elif func_name == ProblemFuncNames.F7:
            selected_func = (opfunu.cec_based.F72014(ndim=self.dimensions), ProblemFuncOptValue.F7.value)
        elif func_name == ProblemFuncNames.F8_BASIC:
            raise NotImplementedError("Essa função está em processo de refatoração.")
            # selected_func = (self.f8_basic, ProblemFuncOptValue.F8_BASIC.value)
        elif func_name == ProblemFuncNames.F8:
            selected_func = (opfunu.cec_based.F82014(ndim=self.dimensions), ProblemFuncOptValue.F8.value)
        elif func_name == ProblemFuncNames.F9:
            selected_func = (opfunu.cec_based.F92014(ndim=self.dimensions), ProblemFuncOptValue.F9.value)
        elif func_name == ProblemFuncNames.F14:
            selected_func = (opfunu.cec_based.F142014(ndim=self.dimensions), ProblemFuncOptValue.F14.value)
        else:
            raise ValueError("Função selecionada não está implementada ainda.")

        return selected_func
    
    def f1_basic(self, particle):
        dim = len(particle)
        assert dim > 0, "A dimensão deve ser maior que zero."

        particle_flattened = np.array(particle).ravel()
        ndim = len(particle_flattened)
        idx = np.arange(0, ndim)

        return np.sum((10 ** 6) ** (idx / (ndim - 1)) * particle_flattened ** 2)    
        
    def f8_basic(self, particle):
        dim = len(particle)
        assert dim > 0, "A dimensão deve ser maior que zero."

        v = 0.0
        for i in range(dim):
            v += pow(particle[i], 2.0) - (10.0 * math.cos(2.0 * math.pi * particle[i])) + 10.0

        return v
    
    def calc_distancia_euclidiana_media_da_populacao(self, population):
        # Inicializando uma matriz para armazenar as distâncias
        num_vectors = len(population)
        distances = np.zeros((num_vectors, num_vectors))

        # Calcula as distâncias euclidianas entre os vetores
        for i in range(num_vectors):
            for j in range(i, num_vectors):
                distance = np.sqrt(np.sum((population[i] - population[j])**2))
                distances[i, j] = distance
                distances[j, i] = distance

        # Calcula a média das distâncias
        average_distance = np.mean(distances)

        return average_distance
    
    def salvar_registro_geral(self,
                              registro: dict,
                              exp_path: str):
        
        df_registro = pd.DataFrame([registro])
        Utils.save_experiment_as_csv(base_dir = exp_path, dataframe = df_registro, filename = 'opt_history')

    def criar_grafico_evolucao_fitness(self, 
                                       hist_best_fitness: list, 
                                       hist_avg_fitness: list, 
                                       imgs_path: str, 
                                       img_name: str,
                                       use_log: bool = False) -> None:
        
        plt.figure()
        xticks_ajustado = [v * 500 for v in range(len(hist_best_fitness))]
        
        plt.plot(xticks_ajustado, hist_best_fitness, color = 'green')
        plt.plot(xticks_ajustado, hist_avg_fitness, color = 'red')
        
        plt.title('Best e Avg fitness através das gerações')
        plt.xlabel('Gerações')
        plt.ylabel('Fitness')
        plt.legend(['Best', 'Avg'])
        
        if use_log:
            plt.yscale('log')

        filename = f'{imgs_path}/{img_name}.jpg' 
        plt.savefig(filename)

    def criar_grafico_evolucao_distancia_media_pontos(self, 
                                                      hist_dist_pontos: list,
                                                      imgs_path: str, 
                                                      img_name: str,
                                                      use_log: bool = False) -> None:
        
        plt.figure()
        
        plt.plot(hist_dist_pontos, color = 'green')
        
        plt.title('Distância média das partículas')
        plt.xlabel('Gerações')
        plt.ylabel('Avg')
        plt.legend(['Avg'])
        
        if use_log:
            plt.yscale('log')

        filename = f'{imgs_path}/{img_name}.jpg' 
        plt.savefig(filename)