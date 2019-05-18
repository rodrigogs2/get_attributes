

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
import datetime
import multiprocessing
from multiprocessing import Pool


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

# Classifier parameters
__DEFAULT_KNN_K_VALUE = 5

# Runtime Parameters
__DEAP_RUN_ID = ''
__OUTPUT_DIRECTORY = '../'
__MULTI_CPU_USAGE = False
__VERBOSE = False

# Default Evolutionary Parameters
__TOURNEAMENT_SIZE = 10
__MUTATE_INDP = 0.10
__CROSSOVER_INDP = 0.40
__POPULATION_SIZE = 300
__NUMBER_OF_GENERATIONS = 200
__MAX_GENERATIONS_WITHOUT_IMPROVEMENTS = 40
__DEFAULT_TARGET_FITNESS = 0.0
__DEFAULT_WORST_FITNESS = -1.0
__GENES_LOW_LIMITS = [0,0,1]
__GENES_UP_LIMITS = [2,160,20]
__SEEDS_FILE = ''


# Alarm Variables
__ALARM = True
__DURATION = 1 #seconds
__FREQ = 440 # Hz



def getRandomPlane(planes = __BODY_PLANES):
    plane = random.sample(planes,1)[0]
    return plane



def getRandomTotalSlices(length = __DEFAULT_MAX_CONSEC_SLICES):
    possibles_total_slices = list(range(length))
    total_slices = random.sample(possibles_total_slices, 1)[0]
    return total_slices


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
    
    all_groupings_partitions_list = [] 
    accuracy = 0.0
    conf_matrix = [[1,0,0],[0,1,0],[0,0,1]] # typical confusion matrix for the Alzheimer classification problem     
    ind_size = len(individual)
    
    if ind_size % 3 == 0:    
        # Getting data from a slices grouping
        for g in list(range(ind_size)):
            if g % 3 == 0:
                plane = individual[g]
                first_slice = individual[g+1]
                total_slices = individual[g+2]
                                
                partition = loadattribs.getAttribsPartitionFromSingleSlicesGrouping(all_attribs,all_slice_amounts,plane,first_slice,total_slices)
                
                # append data to a list which will be merged later
                all_groupings_partitions_list.append(partition)
                
                g = g + 3
    else:
        raise ValueError('Bad formatted individual: slices grouping length ({0}) must be a multiple of three.\nIndividual = {1}'.format(ind_size,individual))
        
    # Merging all partitions data
    all_partitions_merged = all_groupings_partitions_list[0]
    for i in range(1,len(all_groupings_partitions_list)):
        all_partitions_merged = all_partitions_merged + all_groupings_partitions_list[i]
    
    # Classifying merged data
    accuracy, conf_matrix = knn_alzheimer.runKNN(all_partitions_merged, output_classes, knn_debug=debug)

    
    return accuracy, conf_matrix


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


def build_parameters_string(max_consecutive_slices,number_of_groupings):
    strPool = []

    #strPool.append('* Running deap with:\n')
    
    global __DEAP_RUN_ID
    strPool.append(' __DEAP_RUN_ID ={0}\n'.format(__DEAP_RUN_ID))
    
    # Classifier parameters
    global __DEFAULT_KNN_K_VALUE
    strPool.append(' __DEFAULT_KNN_K_VALUE ={0}\n'.format(__DEFAULT_KNN_K_VALUE))

    # Slicing parameters
    global __BODY_PLANES, __MAX_SLICES_VALUES, __MIN_SLICES_VALUES, __DEFAULT_MAX_CONSEC_SLICES
    strPool.append(' max_consecutive_slices={0}\n'.format(max_consecutive_slices))
    strPool.append(' number_of_groupings={0}\n'.format(number_of_groupings))
    strPool.append(' __BODY_PLANES={0}\n'.format(__BODY_PLANES))
    strPool.append(' __MAX_SLICES_VALUES={0}\n'.format(__MAX_SLICES_VALUES))
    strPool.append(' __MIN_SLICES_VALUES={0}\n'.format(__MIN_SLICES_VALUES))
    strPool.append(' __DEFAULT_MAX_CONSEC_SLICES={0}\n'.format(__DEFAULT_MAX_CONSEC_SLICES))
    
    
    # Evolutionary arguments
    global __TOURNEAMENT_SIZE, __MUTATE_INDP, __CROSSOVER_INDP, __POPULATION_SIZE, __NUMBER_OF_GENERATIONS 
    global __MAX_GENERATIONS_WITHOUT_IMPROVEMENTS, __DEFAULT_TARGET_FITNESS, __DEFAULT_WORST_FITNESS
    global  __GENES_LOW_LIMITS, __GENES_UP_LIMITS
    strPool.append(' __TOURNEAMENT_SIZE ={0}\n'.format(__TOURNEAMENT_SIZE))
    strPool.append(' __MUTATE_INDP ={0}\n'.format(__MUTATE_INDP))
    strPool.append(' __CROSSOVER_INDP ={0}\n'.format(__CROSSOVER_INDP ))
    strPool.append(' __POPULATION_SIZE ={0}\n'.format(__POPULATION_SIZE ))
    strPool.append(' __NUMBER_OF_GENERATIONS ={0}\n'.format(__NUMBER_OF_GENERATIONS))
    strPool.append(' __MAX_GENERATIONS_WITHOUT_IMPROVEMENTS  ={0}\n'.format(__MAX_GENERATIONS_WITHOUT_IMPROVEMENTS ))
    strPool.append(' __DEFAULT_TARGET_FITNESS ={0}\n'.format(__DEFAULT_TARGET_FITNESS))
    strPool.append(' __DEFAULT_WORST_FITNESS ={0}\n'.format(__DEFAULT_WORST_FITNESS))
    strPool.append(' __GENES_LOW_LIMITS ={0}\n'.format(__GENES_LOW_LIMITS))
    strPool.append(' __GENES_UP_LIMITS ={0}\n'.format(__GENES_UP_LIMITS))

    # Runtime parameters
    global __MULTI_CPU_USAGE, __OUTPUT_DIRECTORY
    strPool.append(' __MULTI_CPU_USAGE  ={0}\n'.format(__MULTI_CPU_USAGE ))
    strPool.append(' __OUTPUT_DIRECTORY ={0}\n'.format(__OUTPUT_DIRECTORY))
    
    return strPool
    

def setRunID():
    global __DEAP_RUN_ID
    __DEAP_RUN_ID = str(datetime.date.today()) + '_' + str(int(round(time.time())))
    return __DEAP_RUN_ID

def saveParametersFile(max_consec_slices,num_groupings):
    global __OUTPUT_DIRECTORY, __DEAP_RUN_ID
    output_dir_path = os.path.join(__OUTPUT_DIRECTORY, build_experiment_output_dir_name())
    
    
    filename = 'run_{0}.txt'.format(__DEAP_RUN_ID)
    param_full_filename = os.path.join(output_dir_path, filename)
    
    append_mode = "a"
    blank_file = False
    
    # checking output file
    if not os.path.exists(param_full_filename):
        blank_file = True
        #param_file_dir,param_filename = os.path.split(param_full_filename)
        
        # creates output dir when path doesnt exist
        if output_dir_path != '' and not os.path.exists(output_dir_path):
            try:
                os.makedirs(output_dir_path)
            except os.error:
                print ('*** ERROR: Output directory (%s) can not be created\n' % output_dir_path)
                sys.exit(1)
        
    # Writting to output file
    try :
        
        file_handler = open(param_full_filename, append_mode)
        
        if blank_file:
            head = 'Global Parameters'
            file_handler.write(head)
        
        lines = build_parameters_string(max_consec_slices,num_groupings)
        for line in lines:
            file_handler.write(line)
        
        file_handler.close()
            
    except os.error:
        print("\n*** ERROR: file %s can not be written" % param_full_filename)
        exit(1)
    
    

def build_experiment_output_dir_name():
    
    global __OUTPUT_DIRECTORY, _DEAP_RUN_ID
    global __DEFAULT_MAX_CONSEC_SLICES, __DEFAULT_KNN_K_VALUE
    global __MUTATE_INDP, __CROSSOVER_INDP, __POPULATION_SIZE
    
    parameters = 'knn_{0}-pm_{1}-pc_{2}-slices_{3}-'.format(__DEFAULT_KNN_K_VALUE,__MUTATE_INDP, __CROSSOVER_INDP,__DEFAULT_MAX_CONSEC_SLICES)
    run_str = 'run_{0}'.format(__DEAP_RUN_ID)
    
    parent_dir = __OUTPUT_DIRECTORY # usually './' or '../'
    
    full_dir_path = parent_dir + parameters + run_str
    return full_dir_path
    #return run_str

def build_experiment_output_filename(exp_num, best_accuracy):
    acc_value = '{0:.2f}'.format(best_accuracy)
    
    full_dir_path = build_experiment_output_dir_name()
    filename = 'acc_{0}-run_{1}-exp_{2:03d}.txt'.format(acc_value, __DEAP_RUN_ID, exp_num)
    output_full_filename = os.path.join(full_dir_path, filename)
    return output_full_filename, __DEAP_RUN_ID

def append_experiment_data_to_output_file(
        exp_num, 
        gen_num, 
        exp_output_full_filename, #built by build_experiment_output_filename inside saveExperimentsData function
        best_ind,
        dbug =__VERBOSE):
    
    bfit = best_ind.fitness.values[0]
        
    append_mode = "a"
    blank_file = False
    
    # checking output file
    if not os.path.exists(exp_output_full_filename):
        blank_file = True
        output_file_dir,output_filename = os.path.split(exp_output_full_filename)
        
        # creates output dir when path doesnt exist
        if output_file_dir != '' and not os.path.exists(output_file_dir):
            try:
                os.makedirs(output_file_dir)
            except os.error:
                print ('*** ERROR: Output directory (%s) can not be created\n' % output_file_dir)
                sys.exit(1)
        
    # Writting to output file
    try :
        output_file = open(exp_output_full_filename, append_mode)
        if blank_file:
            head = 'generation, best_fit, best_individual, confusion_matrix\n'
            output_file.write(head)
        
        line = '{0:3d},{1:.8f},{2}\n'.format(gen_num, bfit, best_ind, str(best_ind.confusion_matrix))
        output_file.write(line)
        output_file.close()
    except os.error:
        output_file.close()
        print(" *** ERROR: file %s can not be written" % exp_output_full_filename)
        exit(1)
    
    return exp_output_full_filename


def saveExperimentsDataToFile(exp_num, best_ind, bestIndividuals, generationsWithImprovements, runID):
    best_accuracy = best_ind.fitness.values[0]
    ofile = build_experiment_output_filename(exp_num,best_accuracy)[0]
    print('Saving output file {0}'.format(ofile))
    
    if len(bestIndividuals) == len(generationsWithImprovements):
    
        for ind, gnum in zip(bestIndividuals, generationsWithImprovements):
            #cmat = ind.confusion_matrix
            append_experiment_data_to_output_file(exp_num,gnum,ofile,ind)
        
    else:
        print('Problem with results format:\n')
        print('len(bestIndividuals)={0}, len(generationsWithImprovements)={1}'.format(len(bestIndividuals), len(generationsWithImprovements)))
        sys.exit(1)


def printExperimentsResults(bestIndividuals):
    for i in range(len(bestIndividuals)):
        print('Best Individual (id={0}): {1}, with Fitness: {2}'.format(bestIndividuals[i]).format)
        print('', end='', flush=True)

    


def run_deap(all_attribs, 
             all_slice_amounts,
             all_output_classes,
             max_consecutive_slices=__DEFAULT_MAX_CONSEC_SLICES, # max length of the each slices range
             number_of_groupings=__DEFAULT_NUMBER_OF_GROUPINGS, # controls how many slices ranges there will be used
             current_experiment=1,
             ):
 
    generationsWithImprovements = []
    
    bestIndividuals = []
    best_ind = None
    lastGenWithImprovements = 0
    
    # Global Variables
    global __DEAP_RUN_ID
    global __BODY_PLANES
    global __MAX_SLICES_VALUES
    global __MIN_SLICES_VALUES
    global __VERBOSE
    global __OUTPUT_DIRECTORY
    global __DEFAULT_MAX_CONSEC_SLICES
    
    if __VERBOSE:
        print('* Starting experiment {0}'.format(current_experiment))
    
    # Updating global variables
    __BODY_PLANES = loadattribs.getBplanes(all_slice_amounts)
    __MIN_SLICES_VALUES,__MAX_SLICES_VALUES = loadattribs.getSliceLimits(all_slice_amounts)
    
    updateGeneBounds(__BODY_PLANES, __MIN_SLICES_VALUES, __DEFAULT_NUMBER_OF_GROUPINGS, __VERBOSE)
    
    #if 'FitnessMax' not in globals() and current_experiment == 1:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    #if 'Individual' not in globals() and current_experiment == 1:
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
                     debug=False)
#    
            
    # defining population as a plane list
    toolbox.register('population',
                     tools.initRepeat,
                     list,
                     toolbox.individual)    
    
    # creating initial individuals
    pop = toolbox.population(n=__POPULATION_SIZE)
    
   
    # Initializing variables
    best_ind = pop[0]            
    current_generation = 0
    #new_best_found = False
    
    # Evaluate initial population
    fits_and_matrices = list(map(toolbox.evaluate, pop))
    for ind, fit_and_cmatrix_tuple in zip(pop, fits_and_matrices):
        ind.fitness.values = fit_and_cmatrix_tuple[0],
        ind.confusion_matrix = fit_and_cmatrix_tuple[1]
        
        # tracking initial best individual
        if fit_and_cmatrix_tuple[0] > best_ind.fitness.values[0]:
            best_ind = ind
        
    
    # Saving improvements
    bestIndividuals.append(best_ind)
    generationsWithImprovements.append(current_generation)
            
       
    if __VERBOSE:
        print('\n* Experiment {0:3d}: first best individual={0} with fitness={1}'.format(current_experiment, best_ind, best_ind.fitness.values[0]))
    
    toolbox.register("mate", tools.cxUniform, indpb=__CROSSOVER_INDP) # crossing
    toolbox.register("mutate", tools.mutUniformInt, low=__GENES_LOW_LIMITS, up=__GENES_UP_LIMITS, indpb=__MUTATE_INDP) # mutation
    toolbox.register("select", tools.selTournament, tournsize=__TOURNEAMENT_SIZE) # selection        
        
    # Fuction which checks and resolves individual integrity issues
    def checkBounds(bplanes,
                          slice_limits = __MIN_SLICES_VALUES,
                          max_consec_slices = __DEFAULT_MAX_CONSEC_SLICES,
                          amount_of_groupings = __DEFAULT_NUMBER_OF_GROUPINGS,
                          dbug = __VERBOSE):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                
                for child in offspring:
                    for i in range(len(child)):
                        max_value = __GENES_UP_LIMITS[i]
                        min_value = __GENES_LOW_LIMITS[i]

                        if i % 3 == 0: # it's a body plane value!
                            # child[i] value is a body plane number (0,1 or 2)
                            max_value = bplanes[len(bplanes)-1] # Last bplane value is the maximum
                            min_value = 0 # first plane
                        
                        elif i % 3 == 1: # it's a first slice value!
                            # child[i] value is the first slice index (depends on bplane)
                            bplane = child[i-1] # getting actual bplane
                            max_value = slice_limits[bplane] - max_consec_slices
                            min_value = 0
                        
                        elif i % 3 == 2: # it's a total_slices value!
                            # child[i] value is the total of consecutive slices
                            max_value = max_consec_slices
                            min_value = 1
                        child[i] = (child[i] % (max_value - min_value)) + min_value
                        
                return offspring
            return wrapper
        return decorator
    
    # Registering individual integrity checking after mutation operation
    toolbox.decorate('mutate',checkBounds(__BODY_PLANES,
                                          __MIN_SLICES_VALUES, 
                                          max_consecutive_slices, 
                                          number_of_groupings, 
                                          __VERBOSE))
    # Registering individual integrity checking after crossover operation
    toolbox.decorate('mate',checkBounds(__BODY_PLANES,
                                        __MIN_SLICES_VALUES, 
                                          max_consecutive_slices, 
                                        number_of_groupings, 
                                        __VERBOSE))
   
    
    # Initial Population Resume (remove later)
    if __VERBOSE:
        print('* Initial population with {0} individuals:'.format(len(pop)))
        for ind in pop: print ('Individual={0}\t Fitness={1}'.format(ind,ind.fitness.values[0]))
    if __VERBOSE:
        print('Done!')
    
    
    number_of_generations = __NUMBER_OF_GENERATIONS
    
    
    print('\n* Initializing evolution along to {0} generations'.format(number_of_generations))
    
    generations = list(range(1,number_of_generations + 1))
    
    for current_generation in generations:
        if __VERBOSE:
            print('\n* Experiment {0:3d}: Initializing {1:3}th generation...'.format(current_experiment,current_generation))
            print('\tCurrent BEST fitness = {0:.6f}'.format(best_ind.fitness.values[0]))
                  
            print('\t* Running variation operators...',end='')
        
        offspring = algorithms.varAnd(pop, 
                                          toolbox, 
                                      __CROSSOVER_INDP, 
                                      __MUTATE_INDP)
        if __VERBOSE:
            print('\t Done!')
            print('\t* Evaluating offspring...',end='')
        fits_and_matrices = list(map(toolbox.evaluate,offspring)) # list of (fitness,conf_matrix) tuples
        
        if __VERBOSE:
            print('\t Done!')
            print('\t* Updating fitness and confusion matrix of offspring...',end='')
            
        new_best_found = False
        for i in range(len(offspring)):
            # fitness should be a one element tuple
            fit = fits_and_matrices[i][0]
            offspring[i].fitness.values = fit, # updating offspring fitness
            
            conf_matrix = fits_and_matrices[i][1]
            offspring[i].confusion_matrix =  conf_matrix # second tuple element
            
            
            # tracking new best individual
            if fit > best_ind.fitness.values[0]:
                new_best_found = True
                best_ind = offspring[i]
                lastGenWithImprovements = current_generation
        
        if new_best_found:
            # saving improvements
            new_best_found = False
            bestIndividuals.append(best_ind)
            generationsWithImprovements.append(current_generation)
            print('\n\t\t*** Best Individuals found in the {0}th experiment: {1} (fitness={2})'.format(current_experiment,ind,ind.fitness.values[0]))
        
        if __VERBOSE:
            print(' Done!')
                
        if not new_best_found:
            print('\t *** WARNING: Consecutive generations without improvements = {0}'.format(
                current_generation - lastGenWithImprovements))
        
        if (current_generation - lastGenWithImprovements) >= __MAX_GENERATIONS_WITHOUT_IMPROVEMENTS:
            if __VERBOSE:
                print('Evolution was interrupted: More than {0} generations without improvements!'.format(
                        __MAX_GENERATIONS_WITHOUT_IMPROVEMENTS))
            break # stops evolution!!

                
            
        
    if __VERBOSE:
        print('* Best Individual found in {0}th experiment :'.format(current_experiment))
        for ind, current_generation in zip(bestIndividuals, generationsWithImprovements):
            print('\tExperiment={3:2d}  Best Fiteness={0:.6f}  Best Individual={1} at Generation={2}'.format(
                    ind.fitness.values[0],ind,current_generation,current_experiment))
            
    
        print('\n\t*Evolution process has finished')
        
        print('\n\t*Saving experiment data to output file...')
    
    #exp_output_filename = build_experiment_output_filename(current_experiment,best_fit)
    
    #saving data to output file"
    
    saveExperimentsDataToFile(current_experiment, best_ind, bestIndividuals, generationsWithImprovements, __DEAP_RUN_ID)

    
    print('Best Individual Found is: ', best_ind)
    print('\t* Fitness: ', best_ind.fitness.values[0])
    print('\t* Confusion Matrix:\n', best_ind.confusion_matrix)
    



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
    print('\t-n, --number_of_experiments\tnumber of experiments to run with deap (default: 1)')
    print('\t-h, --help\tdisplays this help screen')



# FUNCAO MAIN
def main(argv):
    csv_file = ''
    attribs_dir = ''
    out_dir = './'
    seeds_file = ''
    csv_file_ok = False
    attribs_dir_ok = False
    out_dir_ok = False
    seeds_file_ok = False
    verbose_ok = False
    multi_cpu_ok = False
    number_of_experiments = 1
    
    try:
        opts, args = getopt.getopt(argv[1:],"hc:a:o:s:vmn:",["csv=","attributes_dir=","output_dir=","seeds_file=","verbose","multicpu","number_of_experiments="]) 
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
        elif opt in ("-a", "--attributes_dir"):
            attribs_dir = arg
            attribs_dir_ok = True
        elif opt in ("-o", "--output_dir"):
            out_dir = arg
            out_dir_ok = True
        elif opt in ("-s", "--seeds_file"):
            seeds_file = arg
            seeds_file_ok = True            
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
        print ('\t* Attribs directory is: {0}'.format(attribs_dir))
        print ('\t* Input CSV file is: {0}'.format(csv_file))

        if out_dir_ok:
            global __OUTPUT_DIR
            __OUTPUT_DIR = out_dir
            print ('\t* Output dir is: {0}'.format(__OUTPUT_DIR))

        if seeds_file_ok:
            global SEEDS_FILE
            __SEEDS_FILE = seeds_file
            print ('\t* Seeds file: {0}'.format(__SEEDS_FILE))
       
        if verbose_ok:
            start = time.time()
            global __VERBOSE
            __VERBOSE = True
            
        if multi_cpu_ok:
            global __MULTI_CPU_USAGE
            __MULTI_CPU_USAGE = True
            
        setRunID()
        global __ALARM, __FREQ, __DURATION
        number_of_groupings = __DEFAULT_NUMBER_OF_GROUPINGS
        
        # Loading all data just once
        all_attribs, all_body_planes, all_slice_num, all_slice_amounts, all_output_classes = loadattribs.load_all_data(attribs_dir, csv_file)
        
        #max_slice_values = loadattribs.getSliceLimits(all_slice_num)
        max_consecutive_slices = __DEFAULT_MAX_CONSEC_SLICES

        if __VERBOSE:
            end = time.time()
            print('* Time to load all attributes:',end - start,' seconds')

        
        
        
        print('Running experiments...')
        
        print('Saving parameters list to file... ',end='')
        saveParametersFile(max_consecutive_slices,number_of_groupings)
        print('Done!')
        
        all_experiments = list(range(1,number_of_experiments + 1))
        
        
        if not __MULTI_CPU_USAGE:
            for experiment in all_experiments:
                if __VERBOSE:
                    print('* Running experiment {0}'.format(experiment))
                
                run_deap(all_attribs,
                         all_slice_amounts,
                         all_output_classes,
                         max_consecutive_slices, # length of slices range
                         number_of_groupings,
                         experiment) # controls how many slices ranges there will be used

        else:
            cores_num = multiprocessing.cpu_count()
            with Pool(cores_num) as p:
                from functools import partial
                p.map( 
                    partial(run_deap,
                            all_attribs,
                            all_slice_amounts,
                            all_output_classes,
                            max_consecutive_slices, # length of slices range
                            number_of_groupings),
                    all_experiments) 
        
        if __ALARM:
            os.system('play -nq -t alsa synth {} sine {}'.format(__DURATION, __FREQ))
    else:
        display_help()
    

if __name__ == "__main__":    
    main(sys.argv)
    
