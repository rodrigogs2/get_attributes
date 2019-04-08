#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:35:46 2018

@author: rodrigo
"""

#import numpy
from deap import base, creator, tools
from random import sample
import random
import loadattribs


class Individual:
    values = []
    
    
    

# Use this arguments to set the input directory of attributes files
#attributes_dir = "/home/rodrigo/Downloads/fake_output_dir2/"
#csv_file = '/home/rodrigo/Documents/_phd/csv_files/ADNI1_Complete_All_Yr_3T.csv' 
# Getting all files

#attribs, body_planes, slice_num, slice_amounts, output_classes = load_all_data(attributes_dir, csv_file)

# Definindo classe (tipo) da função de avaliação
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Definindo classe Inviidual que será composta de dois atributos: 
# array de valores decimais e um valor de aptidao (fitness)
# Usando numpy: creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)
creator.create("Individual", list, fitness=creator.FitnessMin)

total_jogadores = 24
jogadores = []
nota_maxima = 60.0
ind_size = 12 # As duas equipes de volei (6 priimeiras posicoes definem a equipe A e as demais definem a equipe B)
pop_size = 50 # Definição do tamanho da população


# inicializando toolbox
toolbox = base.Toolbox()

# definindo alias para a função que cria lista de indices (valores que definem o individuo)
toolbox.register("possiveis_indices", sample, range(ind_size), ind_size)

# definindo alias para função que cria um objeto do tipo Individuo
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.possiveis_indices)

# definindo alias para a função que cria uma população de N indivíduos aleatórios
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def build_players_list(num_players=total_jogadores, max_score=nota_maxima):
    for i in range(num_players):
        jogadores.append(random.random() * max_score)

build_players_list()

##print("Notas dos Jogadores: ", jogadores)

def get_ability(index):
    if index > total_jogadores:
        raise ValueError("index tem valor além do permitido")
    elif index < 0:
        raise ValueError("index negativo nao é permitido")
    else:
        return jogadores[index]

# Funcao para avaliar individuo para o problema do volei    
def evaluate(individual,debug=False):
    time_a = individual[0:6]
    time_b = individual[6:12]
    
    time_a_abilities = []
    time_b_abilities = []
    
    #total_ability_a = 0.0 # Acumulador
    #total_ability_b = 0.0 # Acumulador

    for ind_a in time_a:
        time_a_abilities.append(jogadores[ind_a])
    for ind_b in time_b:
        time_b_abilities.append(jogadores[ind_b])
    
    total_ability_a = sum(time_a_abilities)
    total_ability_b = sum(time_b_abilities)
    
    fitness_value = abs(total_ability_a - total_ability_b)
    individual.fitness.values = (fitness_value,) # Fitness must be a tuple!
    
    if(debug):
        print("\nIndividuo avaliado: ", individual)
        print("\nTime A: ", time_a)    
        print("Time B: ", time_b)
        print("\nAll players abilities: ", jogadores)
        print("\nTeam A abilities:", time_a_abilities)
        print("Team B abilities:", time_b_abilities)
        print("\nTeam A Total Ability: ", total_ability_a)
        print("Team B Total Ability: ", total_ability_b)
        print("Individual Fitness: ", fitness_value)
    return fitness_value

def evaluate_population(some_population):
    for ind in some_population:
        evaluate(ind)

## criando e imprimindo populacao
pop = toolbox.population(pop_size)

def print_population_fitness(some_population=pop):
    error = False
    for ind in some_population:
        if ind.fitness.valid :
            print("Individuo:", ind, "Fitness:", ind.fitness.values[0])
        else:
            error = True
    if error :
        print("*** ERROR: There is a no evaluated individual on population at least")

# testando criação de instância do tipo Individual
#ind_teste = toolbox.individual()
#print("Teste para criação de Individuo: ", ind_teste)
#evaluate(ind_teste,debug=True)


#print("\nPopulacao de testede tamanho", pop_size, ":\n", pop)

## avaliando e imprimindo populacao
evaluate_population(pop)



print("\nPopulacao de teste já avaliada:\n", pop)
print_population_fitness(pop)


