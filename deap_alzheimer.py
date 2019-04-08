#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:35:46 2018

@author: rodrigo
"""

#import numpy
from deap import base, creator, tools, algorithms
#from random import samplei
import random, sys, os
import loadattribs 
import knn_alzheimer
import getopt
import time

# Globla Slicing Arguments
#global __ALL_ATTRIBS, __ALL_OUTPUT_VALUES, __BODY_PLANES, __MAX_SLICES_VALUES, __DEFAULT_MAX_CONSEC_SLICES, __DEFAULT_NUMBER_OF_GROUPINGS

# Global Alzheimer Classification Problem Arguments
#global __GENES_LOW_LIMITS, __GENES_UP_LIMITS, __DEFAULT_KNN_K_VALUE, __VERBOSE

# Global Runtime Parameters
#global __MULTI_CPU_USAGE

# Global Evolutionary Parameters
#global __TOURNEAMENT_SIZE, __MUTATE_INDP, __CROSSOVER_INDP, __NUMBER_OF_GENERATIONS, __POPULATION_SIZE, __DEFAULT_TARGET_FITNESS, __DEFAULT_WORST_FITNESS

# Slicing Arguments
__ALL_ATTRIBS = []
__ALL_OUTPUT_VALUES = []
__BODY_PLANES = []
__MAX_SLICES_VALUES = []
__MIN_SLICES_VALUES = []
__DEFAULT_MAX_CONSEC_SLICES = 20
__DEFAULT_NUMBER_OF_GROUPINGS = 1

# Evolutionary arguments
__GENES_LOW_LIMITS = [0,0,1]
__GENES_UP_LIMITS = [2,160,20]
__DEFAULT_KNN_K_VALUE = 7
__VERBOSE = False

# Runtime Parameters
__MULTI_CPU_USAGE = False

# Default Evolutionary Parameters
__TOURNEAMENT_SIZE = 4
__MUTATE_INDP = 0.10
__CROSSOVER_INDP = 0.4
__NUMBER_OF_GENERATIONS = 5
__POPULATION_SIZE = 20
__DEFAULT_TARGET_FITNESS = 0.0
__DEFAULT_WORST_FITNESS = -1.0



def getRandomPlane(planes = __BODY_PLANES):
    plane = random.sample(planes,1)[0]
    return plane



def getRandomTotalSlices(length = __DEFAULT_MAX_CONSEC_SLICES):
    possibles_total_slices = list(range(length))
    total_slices = random.sample(possibles_total_slices, 1)[0]
    return total_slices



#def getRandomFirstSlice(plane,
#                        total_slices,
#                        sampled_max_index = __MAX_SLICES_VALUES, # Maximum value for the first slice index 
#                        dbug= __VERBOSE):
#    
#    range_first_slice = list(range(abs(max_index - total_slices)))
#    
#    first_slice_index = random.sample(range_first_slice,1)[0]
#    
#    return 1

# Creates a tuple whichs represents a slice grouping composed by @length
# consecutives slices
def buildRandomSliceGrouping(planes, length, max_indexes, dbug):
    
    # choosing slice grouping size
    total_slices = getRandomTotalSlices(length-1) + 1
    #print('sampled total_slices=',total_slices)
    
    #picking a plane
    plane = getRandomPlane(planes)
    #print('sampled plane=',plane)
    
    # maximum index to the last slice not considering the number choosen of total slices 
    max_index = max_indexes[plane]
    #print ('max_index of plane ({0})={1}'.format(plane,max_index))
        
    # calculing the first slice index based on total of slices that will be used
    possibles_indexes_to_first_slice = list(range(abs(max_index - total_slices)))
    #print ('range_first_slice=',range_first_slice)
    
    first_slice_index = random.sample(possibles_indexes_to_first_slice ,1)[0]
    #print('sampled first_slice_index=',first_slice_index)
    
    return [plane, first_slice_index, total_slices]


def initIndividual(planes = __BODY_PLANES,
                   length = __DEFAULT_MAX_CONSEC_SLICES,
                   max_indexes = __MIN_SLICES_VALUES, # Maximum value for the first slice index 
                   groupings = __DEFAULT_NUMBER_OF_GROUPINGS,
                   dbug= __VERBOSE):
   
    data = []
    for i in list(range(groupings)):
        slice_grouping = buildRandomSliceGrouping(planes, length, max_indexes, dbug)
        data = data + slice_grouping
        
    return data

# Evaluates a individual instance represented by Slices Groupings 
def evaluateSlicesGroupingsKNN(individual, # list of integers
                               all_attribs,
                               all_slice_amounts,
                               output_classes,
                               k_value=5,
                               debug=__VERBOSE):
    
    all_groupings_partitions = []
    
    accuracy = 0.0
    conf_matrix = [[1,0,0],[0,1,0],[0,0,1]] # typical confusion matrix for the Alzheimer classification problem     
    
    ind_size = len(individual)
    
    if ind_size % 3 == 0:    
        # Getting and Merging data from all slices groupings
        #if debug: print('evaluating individual=',individual)
        for g in list(range(ind_size)):
            if g % 3 == 0:
                plane = individual[g]
                first_slice = individual[g+1]
                total_slices = individual[g+2]
                
                # Debugging
                #grouping_index = g // 3
                #if debug: print('{0}th grouping: plane={1},first_slice={2},total_slices={3}'.format(grouping_index,plane,first_slice,total_slices))
                
                partition = loadattribs.getAttribsPartitionFromSingleSlicesGrouping(all_attribs,all_slice_amounts,plane,first_slice,total_slices)
                all_groupings_partitions.append(partition)
                
                g = g + 3
    else:
        raise ValueError('Bad format individual: slices grouping length ({0}) must be a multiple of three.\nIndividual = {1}'.format(ind_size,individual))
        
    
        
    # TEMPORARIO!!
    # ATENCAO!!
    accuracy = random.random()
    accuracy, conf_matrix = knn_alzheimer.runKNN(all_groupings_partitions[0], output_classes)
    # ATENCAO!!

    #fitness_value = random.random()
    fitness_value = accuracy
    
    return fitness_value, conf_matrix


def print_population_fitness(some_population):
    error = False
    for ind in some_population:
        if ind.fitness.valid :
            print("Individuo:", ind, "Fitness:", ind.fitness.values[0])
        else:
            error = True
    if error :
        print("*** ERROR: There is a no evaluated individual on population at least")

def updateLowAndUpLimits():
    return 0

def updateGeneBounds(bplanes,
                     slices_limits,
                     max_consec_slices,
                     number_of_slices_groupings,
                     dbug=__VERBOSE):
        
        low_limits_per_slice_grouping = [0,0,1]
        up_limits_per_slice_grouping = [len(bplanes),
                                        min(slices_limits),
                                        max_consec_slices]
        
        all_up_limits = []
        all_low_limits = []
        for n in range(number_of_slices_groupings):
            all_up_limits = all_up_limits + up_limits_per_slice_grouping
            all_low_limits = all_low_limits + low_limits_per_slice_grouping
            
        global __GENES_LOW_LIMITS
        global __GENES_UP_LIMITS
        
        __GENES_LOW_LIMITS = all_low_limits
        __GENES_UP_LIMITS = all_up_limits
        
        if dbug:
            print('updating global MUTATE Low Limits...')
            print('__GENES_LOW_LIMITS: ',__GENES_LOW_LIMITS)
            print('__GENES_UP_LIMITS: ',__GENES_UP_LIMITS)
   


def run_deap(all_attribs, 
             all_slice_amounts,
             all_output_classes,
             max_consecutive_slices=__DEFAULT_MAX_CONSEC_SLICES, # max length of the each slices range
             number_of_groupings=__DEFAULT_NUMBER_OF_GROUPINGS # controls how many slices ranges there will be used
             ):
    

    
    # Global Variables
    global __BODY_PLANES
    global __MAX_SLICES_VALUES
    global __MIN_SLICES_VALUES
    global __VERBOSE
    global __DEFAULT_MAX_CONSEC_SLICES
    
    # Updating global variables
    __BODY_PLANES = loadattribs.getBplanes(all_slice_amounts)
    __MIN_SLICES_VALUES,__MAX_SLICES_VALUES = loadattribs.getSliceLimits(all_slice_amounts)
    #__MAX_SLICES_VALUES = loadattribs.getSliceLimits(all_slice_amounts)[0]
    
    updateGeneBounds(__BODY_PLANES, __MIN_SLICES_VALUES, __DEFAULT_NUMBER_OF_GROUPINGS, __VERBOSE)
    
    print('* Running deap with:')
    print('\t* max_consecutive_slices=',max_consecutive_slices)
    print('\t* number_of_groupings=',number_of_groupings)
    print('\t* __BODY_PLANES=',__BODY_PLANES)
    print('\t* __MAX_SLICES_VALUES=',__MAX_SLICES_VALUES)
    print('\t* __MIN_SLICES_VALUES=',__MIN_SLICES_VALUES)
    print('\t* __VERBOSE=',__VERBOSE)
    print('\t* __DEFAULT_MAX_CONSEC_SLICES=',__DEFAULT_MAX_CONSEC_SLICES)
    print('\t* __POPULATION_SIZE=', __POPULATION_SIZE)
    
    print('\t* __TOURNEAMENT_SIZE =',__TOURNEAMENT_SIZE)
    print('\t* __MUTATE_INDP =',__MUTATE_INDP)
    print('\t* __CROSSOVER_INDP =',__CROSSOVER_INDP )
    print('\t* __NUMBER_OF_GENERATIONS =',__NUMBER_OF_GENERATIONS)
    print('\t* __POPULATION_SIZE =',__POPULATION_SIZE )
    print('\t* __DEFAULT_TARGET_FITNESS =',__DEFAULT_TARGET_FITNESS)
    print('\t* __DEFAULT_WORST_FITNESS =',__DEFAULT_WORST_FITNESS)
    
    # Evolutionary arguments,)
    print('\t* __GENES_LOW_LIMITS = ',__GENES_LOW_LIMITS)
    print('\t* __GENES_UP_LIMITS = ',__GENES_UP_LIMITS)
    print('\t* __DEFAULT_KNN_K_VALUE = ',__DEFAULT_KNN_K_VALUE)
    print('\t* __VERBOSE = ',__VERBOSE)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax, confusion_matrix=None)

    # inicializando toolbox
    toolbox = base.Toolbox()

    # registrando funcao que cria uma tupla de valores que define um agrupamento de fatias
    toolbox.register('init_grouping',
                     initIndividual,
                     planes = __BODY_PLANES, 
                     length = __DEFAULT_MAX_CONSEC_SLICES, 
                     max_indexes = __MIN_SLICES_VALUES, # Maximum value for the first slice index 
                     groupings = __DEFAULT_NUMBER_OF_GROUPINGS,
                     dbug= __VERBOSE)
    
    
    # registrando alias para a função que cria um objeto do tipo Individuo atraves
    # da chamada consecutiva por n vezes da funcao ja registrada 'init_grouping' 

    toolbox.register('individual',
                     tools.initIterate,
                     creator.Individual,
                     toolbox.init_grouping)
    
    toolbox.register('evaluate',
                     evaluateSlicesGroupingsKNN,
                     all_attribs = all_attribs,
                     all_slice_amounts = all_slice_amounts,
                     output_classes = all_output_classes,
                     k_value = __DEFAULT_KNN_K_VALUE,
                     debug=__VERBOSE)
#    
            
    # defining population as a plane list
    toolbox.register('population',
                     tools.initRepeat,
                     list,
                     toolbox.individual)    
    
    # creating initial individuals
    pop = toolbox.population(n=__POPULATION_SIZE)
    
    # TEMPORARY
    global best_ind
    global best_fit
    best_fit = -1.0
    
    # Evaluate initial population
    fits_and_matrices = list(map(toolbox.evaluate, pop))
    for ind, fit_and_matrix_tuple in zip(pop, fits_and_matrices):
        ind.fitness.values = (fit_and_matrix_tuple[0],)
        ind.confusion_matrix = fit_and_matrix_tuple[1]
        
        # Temporary
        if ind.fitness.values[0] > best_fit:
            best_ind = ind
        
    
    # evaluating initial individuals
    
#    import queue
#    queue_size = 5
#    best_inds = queue.Queue(maxsize=queue_size ) # where best individuals are saved in
#    
#    for ind in pop:
#        best_fit = -1 # maximization problem
#        fitness, conf_matrix = toolbox.evaluate(ind,
#                                                all_attribs=all_attribs,
#                                                all_slice_amounts=all_slice_amounts,
#                                                output_classes=all_output_classes,
#                                                debug=__VERBOSE)
#        ind.fitness.values = (fitness,)
#        ind.confusion_matrix = conf_matrix
#        
#        if fitness > best_fit:
#            best_fit = fitness
#
#            if best_inds.full:
#                ind_out = best_inds.get() # pop from left
#                
#            best_inds.put(ind)
#            
#            print('New best individual was found!\n best_inds now is:\n',best_inds)
    
    toolbox.register("mate", tools.cxUniform, indpb=__CROSSOVER_INDP) # crossing
    toolbox.register("mutate", tools.mutUniformInt, low=__GENES_LOW_LIMITS, up=__GENES_UP_LIMITS, indpb=__MUTATE_INDP) # mutation
    toolbox.register("select", tools.selTournament, tournsize=__TOURNEAMENT_SIZE) # selection        
        
    
    def checkBounds(bplanes,
                          slice_limits = __MIN_SLICES_VALUES,
                          max_consec_slices = __DEFAULT_MAX_CONSEC_SLICES,
                          amount_of_groupings = __DEFAULT_NUMBER_OF_GROUPINGS,
                          dbug = __VERBOSE):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                
                # COMENTAR ESTE FOR
                for child in offspring:
                    for i in range(len(child)):
                        max_value = __GENES_UP_LIMITS[i]
                        min_value = __GENES_LOW_LIMITS[i]

                        if i % 3 == 0: # it's a body plane value!
                            # child[i] value is a body plane number (0,1 or 2)
                            max_value = len(bplanes) - 1 # Last plane
                            min_value = 0 # first plane
                            #print('i % 3 == 0, so it\'s a body plane value which is:')
                        
                        elif i % 3 == 1: # it's a first slice value!
                            # child[i] value is the first slice index (depends on bplane)
                            bplane = child[i-1] # getting actual bplane
                            max_value = slice_limits[bplane] - max_consec_slices
                            min_value = 0
                            #print('i % 3 == 1, so it\'s the first slice index value which is:')
                        
                        elif i % 3 == 2: # it's a total_slices value!
                            # child[i] value is the total of consecutive slices
                            max_value = max_consec_slices
                            min_value = 1
                            #print('i % 3 == 1, so it\'s the consecutive slices amount value which is:')
                        
                        child[i] = (child[i] % (max_value - min_value)) + min_value
                        
                        #print('child[{0}] is {1}'.format(i,child[i]))
                        
                        # Ex.: Se apos mutacao o valor child[i]==24, max==20 e min==1 
                        # valor corrigido = (24 % (20 - 1)) + 1
                        #                   = (5) + 1 = 6
                        
                        
                return offspring
            return wrapper
        return decorator
    
    
    # Building lower and upper bounds to each gene of a individual
    
    # Updating global limits for genes
    #updateGeneBounds(__BODY_PLANES,__MAX_SLICES_VALUES,max_consecutive_slices,number_of_groupings,__VERBOSE)
    
    
    
    #toolbox.decorate('mate',checkMateBounds(slice_limits, max_consecutive_slices ))#max_slice_values, max_consecutive_slices))
    toolbox.decorate('mutate',checkBounds(__BODY_PLANES,
                                          __MIN_SLICES_VALUES, 
                                          max_consecutive_slices, 
                                          number_of_groupings, 
                                          __VERBOSE))

    toolbox.decorate('mate',checkBounds(__BODY_PLANES,
                                          __MIN_SLICES_VALUES, 
                                          max_consecutive_slices, 
                                          number_of_groupings, 
                                          __VERBOSE))
    
    
    ## criando e imprimindo populacao
    if __VERBOSE:
        print('* Initial population with {0} individuals:'.format(len(pop)))
        for ind in pop: print ('Individual={0}\t Fitness={1}'.format(ind,ind.fitness.values[0]))
    if __VERBOSE:
        print('Done!')
    
    
    number_of_generations = __NUMBER_OF_GENERATIONS
    
    
    print('\n* Initializing evolution along to {0} generations'.format(number_of_generations))
    
    for gen in range(number_of_generations):
        print('\n* Initializing {0}th generation...'.format(gen))
        
        print('\t* Running variation operators...')
        offspring = algorithms.varAnd(pop, 
                                      toolbox, 
                                      __CROSSOVER_INDP, 
                                      __MUTATE_INDP)
        print('\t Done!')
        
        print('\n\t* Evaluating offspring...')
        fits_and_matrices = list(map(toolbox.evaluate,offspring)) # list of (fitness,conf_matrix) tuples
        print('\t Done!')
        
        
    
        #for child in offspring:
        #child.fitness.values = fits_and_matrices[0][0]
            #child.confusion_matrix = fits_and_matrices[]
        
        
        print('\n\t* Updating fitness and confusion matrix of offspring...')
        for i in range(len(offspring)):
            # fitness should be a one element tuple
            fit = fits_and_matrices[i][0]
            offspring[i].fitness.values = fit, # first tuple element
            
            conf_matrix = fits_and_matrices[i][1]
            offspring[i].confusion_matrix =  conf_matrix # second tuple element
            
            
            # Temporary
            if fit > best_fit:
                print('\t New BEST Individual {0} with Fitness={1} was found!'.format(offspring[i],fit))
                best_fit = fit
                best_ind = offspring[i]
            
        print('\t Done!')
    
    print('Evolution process has finished')
    
    print('Best Individual Found is: ', best_ind)
    print('\t* Fitness: ', best_fit)
    print('\t* Confusion Matrix:\n', best_ind.confusion_matrix)
    
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
            start = time.time()
            global __VERBOSE
            __VERBOSE = True
            
        if multi_cpu_ok:
            global __MULTI_CPU_USAGE
            __MULTI_CPU_USAGE = True
            
        # Loading all data just once
        all_attribs, all_body_planes, all_slice_num, all_slice_amounts, all_output_classes = loadattribs.load_all_data(attribs_dir, csv_file)
        print('Done!')
        
        #max_slice_values = loadattribs.getSliceLimits(all_slice_num)
        max_consecutive_slices = __DEFAULT_MAX_CONSEC_SLICES

        if __VERBOSE:
            end = time.time()
            print('* total used time to load all attributes:',end - start,' seconds')

        print('Running experiments...')
        for experiment in range(number_of_experiments):
            if __VERBOSE:
                print('* Running experiment {0}'.format(experiment))
            
            #possibles_bplanes = loadattribs.getBplanes(all_slice_amounts)
            #max_consecutive_slices = loadattribs.getSliceLimits(all_slice_amounts)

            number_of_groupings = __DEFAULT_NUMBER_OF_GROUPINGS
            #target_fitness = __DEFAULT_TARGET_FITNESS

#def run_deap(all_attribs, 
#             all_slice_amounts,
#             all_output_classes,
#             max_consecutive_slices=__DEFAULT_MAX_CONSEC_SLICES, # max length of the each slices range
#             number_of_groupings=__DEFAULT_NUMBER_OF_GROUPINGS # controls how many slices ranges there will be used
#             ):
            
            run_deap(all_attribs,
                     all_slice_amounts,
                     all_output_classes,
                     #possibles_bplanes, # usully means the list: (0,1,2)
                     max_consecutive_slices, # length of slices range
                     number_of_groupings) # controls how many slices ranges there will be used
                     #max_slice_values, # maximum valid slice index value for each body plane
                     #target_fitness, # sets the target fitness that will stop evolution if it was achieved
                     #multi_cpu_ok, # enables use of multicpu to individuals evaluation
                     #__VERBOSE)
                
            print('* All experiments have finished\nGood bye!')
    else:
        display_help()
    

if __name__ == "__main__":    
    main(sys.argv)
    