#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:35:46 2018

@author: rodrigo
"""

#import numpy
from deap import base, creator, tools, algorithms
#from random import sample
import random, sys, os
import loadattribs 
import knn_alzheimer
import getopt

# Runtime Parameters
__DEFAULT_DEBUG = False
__DEFAULT_MULTI_CPU_USAGE = False

# Default Evolutionary Parameters
__TOURNEAMENT_SIZE = 3
__MUTATE_INDP = 0.05
__NUMBER_OF_GENERATIONS = 5
__POPULATION_SIZE = 10
__DEFAULT_TARGET_FITNESS = 0.0

# Default MRI Slicing Parameters
__DEFAULT_MAX_CONSEC_SLICES = 20
__DEFAULT_MAX_SLICES_VALUES = (255,255,170)
__DEFAULT_BPLANES = (0,1,2)
__DEFAULT_NUMBER_OF_GROUPINGS = 1


# Creates a tuple whichs represents a slice grouping composed by @length
# consecutives slices
def getRandomSliceGrouping(planes = __DEFAULT_BPLANES,
                           length = __DEFAULT_MAX_CONSEC_SLICES,
                           max_indexes = __DEFAULT_MAX_SLICES_VALUES,
                           dbug=__DEFAULT_DEBUG):
    # number of consecutive slices will determine the maximum index for the first slice
    total_slices = random.sample(range(length),1)[0]
    
    # picking just one value from the range of planes
    plane = random.sample(planes,1)[0]
    
    # maximum index to the last slice not considering the number choosen of total slices 
    max_index = max_indexes[plane]
    
    # calculing the first slice index based on total of slices that will be used
    first_slice_index = random.sample(range(max_index - total_slices),1)[0]
    
    if dbug:
        print('Choosen plane:',plane)
        print('Max indexes:',max_indexes)
        print('Max_index for plane {0}: {1}'.format(plane, max_index))
        print('first slice index:',first_slice_index)
        print('total slices:',total_slices)
    
    return plane, first_slice_index, total_slices


# Evaluates Slices Groupings represented by a individual instance
def evaluateSlicesGroupings(slices_groupings,
                            all_attribs,
                            all_slice_amounts,
                            output_classes,
                            k_value=5,
                            debug=False):
    

    
    all_groupings_partitions = []
    
    accuracy = 0.0
    conf_matrix = [[1,0,0],[0,1,0],[0,0,1]]
    
    
    if len(slices_groupings) % 3 == 0:
        
        # Getting and Merging data from all slices groupings
        for g in range(slices_groupings):
            if g % 3 == 0:
                plane = slices_groupings[g]
                first_slice = slices_groupings[g+1]
                total_slices = slices_groupings[g+2]
                
                # Debugging
                grouping_index = g // 3
                print('{0}th grouping: plane={1},first_slice={2},total_slices'.format(grouping_index,plane,first_slice,total_slices))
                
                
                partition = loadattribs.get_attributes_partition(all_attribs,all_slice_amounts,plane,first_slice,total_slices)
                all_groupings_partitions.append(partition)
                
        
        # TEMPORARIO!!
        # ATENCAO!!
        accuracy, conf_matrix = knn_alzheimer.runKNN(all_groupings_partitions[0], output_classes)
        # ATENCAO!!

    #fitness_value = random.random()
    fitness_value = accuracy
    
    return fitness_value, conf_matrix


#def evaluate_population(some_population):
#    for ind in some_population:
#        evaluate(ind)

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
             #possibles_bplanes=__DEFAULT_BPLANES, # usually means the list: (0,1,2)
             max_consecutive_slices=__DEFAULT_MAX_CONSEC_SLICES, # max length of the each slices range
             number_of_groupings=__DEFAULT_NUMBER_OF_GROUPINGS, # controls how many slices ranges there will be used
             #target_fitness=__DEFAULT_TARGET_FITNESS, # sets the target fitness that will stop evolution if it was achieved
             multi_cpu_ok=__DEFAULT_MULTI_CPU_USAGE,
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
    creator.create("Individual", list, fitness=creator.FitnessMax, confusion_matrix=None)
    
    # Individuos sao compostos de agrupamentos e cada agrupamento e' corresponde a
    # uma quantidade especifica de fatias consecutivas extraidas de um dado plano
    # do corpo humano. Assim, cada um desses agrupamentos e' 
    # definido como uma tupla de tres numeros inteiros que correspondem a:
    #   i.  O plano corporal que da origem as fatias
    #   ii. O indice da primeira fatia do agrupamento 
    #   iii.O total de fatias conscutivas existente no agrupamento
    
    # Quantidade de possiveis planos do corpo
    possibles_bplanes = loadattribs.getBplanes(all_slice_amounts)
    #num_bplanes = len(possibles_bplanes)
    
    
    # Numero total de agrupamentos (tuplas) usados para definir o individuo
    # Note: evaluate values between 1 and 6)
    # total_groupings = number_of_groupings
    
    # Quantidade maxima de fatias consecutivas de cada agrupamento
    #max_consecutive_slice_amount = max_consecutive_slices
#    if verbose:
#        print('* max_consecutive_slices:',max_consecutive_slices)
#    
    
    # Valores maximos dos indices para um fatia existente em cada um dos planos 
    # do corpo humano
    slice_limits = loadattribs.getSliceLimits(all_slice_amounts)
    
    
    # Note que o comprimento total do individuo e' definido pelo valor acima 
    # multiplicado pelo tamanho (3) de cada tupla que representa cada agrupamento
    if verbose:
        print('* number_of_groupings', number_of_groupings)
    
    # Values used to another data representation
    #min_slice_num = 0
    #max_slice_num = 680 # 255 (axial) + 255 (coronario) + 170 (sargital)
    
#    pop_size = __POPULATION_SIZE # Definição do tamanho da população
#    if verbose:
#        print('* pop_size:',pop_size )

    # inicializando toolbox
    toolbox = base.Toolbox()
    
    # registrando funcao que cria uma tupla de valores que define um agrupamento de fatias
    toolbox.register('slice_range', 
                     getRandomSliceGrouping, 
                     possibles_bplanes, # @planes argument
                     max_consecutive_slices, # @length argument
                     slice_limits) # @max_indexes argument
    
    # registrando alias para a função que cria um objeto do tipo Individuo atraves
    # da chamada consecutiva por n vezes da funcao ja registrada 'slice_range' 
    toolbox.register('individual',
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.slice_range,
                     n=number_of_groupings)
    
    
    toolbox.register('evaluate',
                     evaluateSlicesGroupings)
    
#                     individual,
#                     all_attribs,
#                     all_slice_amounts,
#                     output_classes,
#                     k_value=5,
#                     debug=False)
#    
            
    # population bag registry
    toolbox.register('population_bag',
                     tools.initRepeat,
                     list,
                     toolbox.individual
                     )
    
    pop = toolbox.population_bag(n=100)
    
    
                
    #    __MAX_CONSEC_SLICES = 20
#    __MAX_SLICES_VALUES = (255,255,170)
#    __BPLANES = (0,1,2)
    
    

    def checkBounds(max_slices_indexes = __DEFAULT_MAX_SLICES_VALUES, 
                    max_slices_amount = __DEFAULT_MAX_CONSEC_SLICES):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        max_value = 1
                        min_value = 0
                        if i % 3 == 0:
                            # child[i] value is a body plane number (0,1 or 2)
                            max_value = 2
                            min_value = 0
                        elif i % 3 == 1:
                            # child[i] value is the first slice index (depends on bplane)
                            bplane = child[i-1]
                            max_value = max_slices_indexes[bplane] - max_slices_amount
                            min_value = 0
                        elif i % 3 == 2:
                            # child[i] value is the total of consecutive slices
                            max_value = max_slices_amount
                            min_value = 1 
                        
                        child[i] = (child[i] % (max_value - min_value)) + min_value
                        # se apos mutacao == 24, 
                        # valor corrigido = (24 % (20 - 1)) + 1
                        #                   = (5) + 1 = 6
                        
                return offspring
            return wrapper
        return decorator
    
    toolbox.register("mate", tools.cxOnePoint) # crossing
    toolbox.register("mutate", tools.mutUniformInt, indpb=0.05) # mutation
    toolbox.register("select", tools.selTournament, tournsize=10) # selection
    
    toolbox.decorate('mutate',checkBounds())

    
    # Testing
    #random_slices_grouping(bplanes,max_consecutive_slice_amount,max_slice_values)
    
    # definindo alias para a função que cria uma população de N indivíduos aleatórios
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    pop = toolbox.population(n=__POPULATION_SIZE)
    
    # Evaluating initial population individuals
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    
    ## criando e imprimindo populacao
    if verbose:
        print('* Initial population with {0} individuals:'.format(len(pop)))
        for ind in pop: print ('Individual={0} Fitness={1}'.format(ind,ind.fitness.values))
    if verbose:
        print('Done!')
    
    number_of_generations = __NUMBER_OF_GENERATIONS
    
    for gen in range(number_of_generations):
        offspring = algorithms.varAnd(pop, 
                                      toolbox, 
                                      cxpb=0.5, 
                                      mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate,offspring)
    
    
    # testando criação de instância do tipo Individual
    #ind_teste = toolbox.individual()
    #print("Teste para criação de Individuo: ", ind_teste)
    
    #evaluate(ind_teste,debug=True)
    
    ## avaliando e imprimindo populacao
    
#    if verbose:
#        print('* Initializing population evaluation...')
#    evaluate_population(pop)
#    if verbose:
#        print('Done!')
#
#    if verbose:
#        print_population_fitness(pop)


# REVISAR!!
def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]

    print ('Usage:\n    ', script_name, '[Options] -c <csv_file> -d <attributes_dir> ')
    print ('  Options:')
    #print('\t-c, --csv\tADNI csv file which contains images metadata')
    #print('\t-d, --dir\tdirectory which contains all attributes files extracted by get_attribs.py tool')
    print('\t-m, --multicpu\tset on computation over all cores (default: multicore is off)')
    print('\t-v, --verbose\tenables verbose mode (default: disabled)')
    print('\t-n, --resume\tnumber_of_experiments: number of experiments to run with deap (default: 1)')
    print('\t-h, --help\tdisplays this help screen')



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
        if opt in ('-h','--help'):
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
            try:    
                number_of_experiments = int(arg)
            except ValueError:
                print('Error: argument {0} must be a integer!'.format(arg))
                display_help()
                sys.exit(0);
            except Exception as err:
                print('Error: An exception has rised on try of integer conversion of the argument {0}.\n\tCause: '.format(arg,err.__cause__))
                display_help()
                sys.exit(0);
    
    if csv_file_ok and attribs_dir_ok:
        print('* Loading data...')
        print ('\t* Attribs directory is: {0}\n'.format(attribs_dir))
        print ('\t* Input CSV file is: {0}'.format(csv_file))
       
        if verbose_ok:
            import time
            start = time.time()
            
        
        # Loading all data just once
        all_attribs, all_body_planes, all_slice_num, all_slice_amounts, all_output_classes = loadattribs.load_all_data(attribs_dir, csv_file)
        print('Done!')

        if verbose_ok:
            end = time.time()
            print('* total used time to load all attributes:',end - start,' seconds')

        print('Running experiments...')
        for experiment in range(number_of_experiments):
            if verbose_ok:
                print('* Running experiment {0}'.format(experiment))
            
            possibles_bplanes = loadattribs.getBplanes(all_slice_amounts)
            max_consecutive_slices = loadattribs.getSliceLimits(all_slice_amounts)
            
            number_of_groupings = __DEFAULT_NUMBER_OF_GROUPINGS
            target_fitness = __DEFAULT_TARGET_FITNESS

            
            run_deap(all_attribs,
                     all_slice_amounts,
                     all_output_classes,
                     #possibles_bplanes, # usully means the list: (0,1,2)
                     max_consecutive_slices, # length of slices range
                     number_of_groupings, # controls how many slices ranges there will be used
                     #target_fitness, # sets the target fitness that will stop evolution if it was achieved
                     multi_cpu_ok, # enables use of multicpu to individuals evaluation
                     verbose_ok)
                
            print('* All experiments have finished\nGood bye!')
    else:
        display_help()
    

if __name__ == "__main__":    
    main(sys.argv)
    