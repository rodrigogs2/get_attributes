#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:35:46 2018

@author: rodrigo
"""

#import numpy
from deap import base, creator, tools
#from random import sample
import random, sys, os
import loadattribs
import getopt


# Creates a tuple whichs represents a slice grouping composed by @length
# consecutives slices
def random_slices_grouping(planes=(0,1,2),
                           length=20,
                           max_indexes=(255,255,170),
                           dbug=False):
    # number of consecutive slices will determine the maximum index for the first slice
    total_slices = random.sample(range(length),1)[0]
    
    # picking just one value from the range of planes
    plane = random.sample(planes,1)[0]
    
    # maximum index to the last slice not considering the number choosen of total slices 
    max_index = max_indexes[plane]
    
    # calculing the first slice index based on total of slices that will be used
    first_slice_index = random.sample(range(max_index - total_slices),1)[0]
    
    
    
    if dbug:
        print('Plane:',plane)
        print('Max_index for plane {0}: {1}'.format(plane, max_index))
        print('first slice index:',first_slice_index)
        print('total_slices:',total_slices)
    
    return plane, first_slice_index, total_slices


# Funcao para avaliar individuo para o problema do volei    
def evaluate(individual,debug=False):
    
    '''
    # Crop database:
    loadattribs.get_attributes_partition(attribs,
                        slice_amounts,
                        specific_body_plane,
                        start_slice,
                        total_slices)
    # Run classifier:
    
    '''
    #
    
    fitness_value = random.random()
    individual.fitness.values = (fitness_value,) # Fitness must be a tuple!
    
    if(debug):
        print("\nIndividual evaluated: ", individual)
        print("Individual Fitness: ", fitness_value)
    return fitness_value


def evaluate_population(some_population):
    for ind in some_population:
        evaluate(ind)

def print_population_fitness(some_population):
    error = False
    for ind in some_population:
        if ind.fitness.valid :
            print("Individuo:", ind, "Fitness:", ind.fitness.values[0])
        else:
            error = True
    if error :
        print("*** ERROR: There is a no evaluated individual on population at least")


def run_deap(all_attribs, 
             all_slice_amounts,
             all_output_classes,
             possibles_bplanes=[0,1,2], # usully means the list: (0,1,2)
             max_consecutive_slices=20, # length of slices range
             number_of_groupings=1, # controls how many slices ranges there will be used
             target_fitness=0.0, # sets the target fitness that will stop evolution if it was achieved
             number_of_experiments=1, # means how many experiments must run
             verbose=False):
    
    # Use this arguments to set the input directory of attributes files
    # attributes_dir = "/home/rodrigo/Downloads/fake_output_dir2/"
    # csv_file = '/home/rodrigo/Documents/_phd/csv_files/ADNI1_Complete_All_Yr_3T.csv' 
    # Getting all files
    
    # Getting attributes, slice_amounts and other stuff
    #attribs, body_planes, slice_num, slice_amounts, output_classes = loadattribs.load_all_data(attribs_dir, csv_file)
    
    # Definindo classe (tipo) da função de avaliação
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
    
    # Para o problema de otimizacao para a escolha dos dados que devem ser usados
    # na classificacao da Doenca de Alzheimer, um individuo representa uma selecao 
    # de fatias especificas extraidas de um exame de MRI. A ideia e' descobrir o
    # conjunto otimo de fatias capaz de promover alta acurracia na classificao dos
    # exames de imagem entre as classes CN, MCI ou AD. 
    
    # Dessa forma, a aptidao (Fitness) do individuo corresponde a acurracia dessa 
    # classificao para o arranjo de fatias definido pelo individuo.
    
    # Definindo classe Inviidual que será composta de dois atributos: 
    #   i.  array de valores inteiros e 
    #   ii. um valor de aptidao (fitness)
    
    # Usando numpy: 
    #creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)
    # Usando listas:
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Individuos sao compostos de agrupamentos e cada agrupamento e' corresponde a
    # uma quantidade especifica de fatias consecutivas extraidas de um dado plano
    # do corpo humano. Assim, cada um desses agrupamentos e' 
    # definido como uma tupla de tres numeros inteiros que correspondem a:
    #   i.  O plano corporal que da origem as fatias
    #   ii. O indice da primeira fatia do agrupamento 
    #   iii.O total de fatias conscutivas existente no agrupamento
    
    # Quantidade de possiveis planos do corpo
    num_bplanes = len(possibles_bplanes)
    
    # Quantidade maxima de fatias consecutivas de cada agrupamento
    max_consecutive_slice_amount = max_consecutive_slices
    if verbose:
        print('* max_consecutive_slice_amount:',max_consecutive_slice_amount)
    
    
    # Valores maximos dos indices para um fatia existente em cada um dos planos 
    # do corpo humano
    slice_limits = loadattribs.get_slices_limits(all_slice_amounts)
    
    max_slice_values = []
    for p in range(num_bplanes):
        max_slice_values.append(slice_limits[p] - max_consecutive_slice_amount)
    
    if verbose:
        print('* max_slice_values:',max_slice_values)
    
    
    # Numero total de agrupamentos (tuplas) usados para definir o individuo
    # Note: evaluate values between 1 and 6)
    total_groupings = number_of_groupings
    # Note que o comprimento total do individuo e' definido pelo valor acima 
    # multiplicado pelo tamanho (3) de cada tupla que representa cada agrupamento
    if verbose:
        print('* total_groupings',total_groupings)
    
    # Values used to another data representation
    #min_slice_num = 0
    #max_slice_num = 680 # 255 (axial) + 255 (coronario) + 170 (sargital)
    
    pop_size = 50 # Definição do tamanho da população
    if verbose:
        print('* pop_size:',pop_size )
    
    # inicializando toolbox
    toolbox = base.Toolbox()
    
    #def get_random_first_slice_index(body_plane, max_indexes):
    #    max_index = max_indexes[body_plane]
    #    return random.sample(range(max_index + 1),1)
    
    #toolbox.register("plane", random.sample, bplanes, 1)
    #toolbox.register("first_slice",get_random_first_slice_index, max_slice_values)
    #toolbox.register("length",)
    
    # Definindo funcao que cria a tupla de valores que define um agrupamento de fatias
    toolbox.register("get_slice_grouping", 
                     random_slices_grouping, 
                     possibles_bplanes, # @planes argument
                     max_consecutive_slice_amount, # @length argument
                     max_slice_values) # @max_indexes argument
    
    # definindo alias para função que cria um objeto do tipo Individuo
    toolbox.register("individual",
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.get_slice_grouping,
                     n=total_groupings)
    
    # Testing
    #random_slices_grouping(bplanes,max_consecutive_slice_amount,max_slice_values)
    
    # definindo alias para a função que cria uma população de N indivíduos aleatórios
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    

    ## criando e imprimindo populacao
    if verbose:
        print('* initializing population with {0} individuals...'.format(pop_size))
    pop = toolbox.population(pop_size)
    if verbose:
        print('Done!')
    
    
    # testando criação de instância do tipo Individual
    #ind_teste = toolbox.individual()
    #print("Teste para criação de Individuo: ", ind_teste)
    
    #evaluate(ind_teste,debug=True)
    
    ## avaliando e imprimindo populacao
    if verbose:
        print('* Initializing population evaluation...')
    evaluate_population(pop)
    if verbose:
        print('Done!')

    if verbose:
        print_population_fitness(pop)

# REFAZER!!
def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]

    print ('Usage:\n    ', script_name, '[Options] -i <inputfile> -o <outputdir> ')
    print ('  Options:')
    print('\t-m, --multicpu\tset on computation over all cores (default: multicore is off)')
    print('\t-v, --verbose\tenables verbose mode (default: disabled)')
    print('\t-r, --resume\tresume extraction: output files are not overwritten (default: resume is off)')



# FUNCAO MAIN
def main(argv):
    csv_file = ''
    attribs_dir = ''
    csv_file_ok = False
    attribs_dir_ok = False
    verbose_ok = False
    multi_cpu_ok = False
    number_of_experiments = 1
    
    try:
        opts, args = getopt.getopt(argv[1:],"hc:d:vn:",["csv=","dir=","verbose","multicpu","number_of_experiments="]) 
    except getopt.GetoptError:
        display_help()
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            display_help()
            sys.exit(0);
        elif opt in ("-c", "--csv"):
            csv_file = arg
            csv_file_ok = True
        elif opt in ("-d", "--dir"):
            attribs_dir = arg
            attribs_dir_ok = True
        elif opt in ("-v", "--verbose"):
            verbose_ok = True
        elif opt in ("-m", "--multicpu"):
            multi_cpu_ok = True
        elif opt in ("-n", "--number_of_experiments"):
            number_of_experiments = arg
    
    if csv_file_ok and attribs_dir_ok:
        print('* Loading data...')
        print ('\t* Attribs directory is: {0}\n'.format(attribs_dir))
        print ('\t* Input CSV file is: {0}'.format(csv_file))
       
        if verbose_ok:
            import time
            start = time.time()
            
        
        all_attribs, all_body_planes, all_slice_num, all_slice_amounts, all_output_classes = loadattribs.load_all_data(attribs_dir, csv_file)
        print('Done!')

        if verbose_ok:
            end = time.time()
            print('* total used time to load all attributes:',end - start,' seconds')

        # REMOVE THIS LATER
        should_run = True
       
        if should_run :
            print('Running experiments...')
            for experiment in range(number_of_experiments):
                if verbose_ok:
                    print('* Running experiment {0}'.format(experiment))
                
                possibles_bplanes = [0,1,2]
                max_consecutive_slices = 20
                number_of_groupings = 1
                target_fitness = 0.0

                
                run_deap(all_attribs,
                         all_slice_amounts,
                         all_output_classes,
                         possibles_bplanes, # usully means the list: (0,1,2)
                         max_consecutive_slices, # length of slices range
                         number_of_groupings, # controls how many slices ranges there will be used
                         target_fitness, # sets the target fitness that will stop evolution if it was achieved
                         number_of_experiments, # means how many experiments must run
                         verbose=verbose_ok)
                
                
    
                
            print('* All experiments have finished\nGood bye!')
    else:
        display_help()
    

if __name__ == "__main__":    
    main(sys.argv)
    