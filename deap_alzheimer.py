

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

# Evolutionary arguments
__GENES_LOW_LIMITS = [0,0,1]
__GENES_UP_LIMITS = [2,160,20]
__DEFAULT_KNN_K_VALUE = 5
__VERBOSE = False

# Runtime Parameters
__MULTI_CPU_USAGE = False

# Default Evolutionary Parameters
__TOURNEAMENT_SIZE = 10
__MUTATE_INDP = 0.30
__CROSSOVER_INDP = 0.40
__NUMBER_OF_GENERATIONS = 100
__POPULATION_SIZE = 200
__DEFAULT_TARGET_FITNESS = 0.0
__DEFAULT_WORST_FITNESS = -1.0
#__BEST_INDIVIDUALS_SIZE = 5

# Results Variables
#queueSize = 10
#bestFitnesses = asyncio.Queue(maxsize=queueSize)
#bestIndividuals = asyncio.Queue(maxsize=queueSize)
#generationsWihImprovements = asyncio.Queue(maxsize=queueSize)
#bestConfMatrices = asyncio.Queue(maxsize=queueSize)
#bestFitnesses = []
bestIndividuals = []
generationsWihImprovements = []
#bestConfMatrices = []
__OUTPUT_DIRECTORY = './'

# Alarm Variables
__ALARM = True
__DURATION = 1 #seconds
__FREQ = 440 # Hz


def getBestFit():
    global best_ind
    if best_ind != None:
        return best_ind.fitness.values[0]
    else:
        return __DEFAULT_WORST_FITNESS

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


def build_experiment_output_filename(exp_num, best_accuracy):
    acc = 'acc_{0:.2f}_'.format(best_accuracy)
    global __OUTPUT_DIRECTORY
    dirname = __OUTPUT_DIRECTORY
    filename = '{3}_experiment_{0:03d}_{1}_{2}.txt'.format(exp_num,datetime.date.today(),int(round(time.time())),acc)
    output_full_filename = os.path.join(dirname, filename)
    return output_full_filename

def append_experiment_data_to_output_file(
        exp_num, 
        gen_num, 
        exp_output_full_filename,
        dbug =__VERBOSE):
    
    #global best_fit
    global best_ind 
    best_fit = getBestFit()
        
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
            head = 'gen,best_fit,individual\n'
            output_file.write(head)
        
        line = '{0:3d},{1:.8f},{2}\n'.format(gen_num, best_fit, best_ind)
        output_file.write(line)
        output_file.close()
    except os.error:
        output_file.close()
        print(" *** ERROR: file %s can not be written" % exp_output_full_filename)
        exit(1)
    
    return exp_output_full_filename


def saveExperimentDataToFile(exp_num):
    best_accuracy = getBestFit()
    ofile = build_experiment_output_filename(exp_num,best_accuracy)
    print('Saving output file {0}'.format(ofile))
    
    #global bestFitnesses, bestIndividuals, generationsWihImprovements, bestConfMatrices
    global bestIndividuals, generationsWihImprovements
    
    if len(bestIndividuals) == len(generationsWihImprovements):
    #if len(bestFitnesses) == len(bestIndividuals) and len(bestIndividuals) == len(generationsWihImprovements) and len(generationsWihImprovements) == len(bestConfMatrices):
#        for fit,gnum,ind,cmat in zip(bestFitnesses,
#                                     generationsWihImprovements,
#                                     bestIndividuals,
#                                     bestConfMatrices):
#            append_experiment_data_to_output_file(exp_num,gnum,str(cmat),ofile)
        for ind, gnum in zip(bestIndividuals, generationsWihImprovements):
            #cmat = ind.confusion_matrix
            append_experiment_data_to_output_file(exp_num,gnum,ofile)
        
    else:
        print('Problem with results format:\n')
        print('len(bestIndividuals)={0}, len(generationsWihImprovements)={1}'.format(len(bestIndividuals), len(generationsWihImprovements)))
        sys.exit(1)


def printExperimentsResults():
    #global queueSize
    global bestIndividuals
    for i in range(len(bestIndividuals)):
        print('Best Individual (id={0}): {1}, with Fitness: {2}'.format(bestIndividuals[i]).format)
        print('', end='', flush=True)
#def saveExperimentData(currentGeneration, 
#                       experiment_number, 
#                       experiment_output_full_filename):
#    global best_fit
#    global best_ind
#    line = str(experiment_number) + ',' + str(currentGeneration) + ',' + str(best_fit) + ',' + str(best_ind)  
#    print('New Best Individual: ' + line)
#    # append_data_to_experiment_output_file(line, experiment_output_file)
    

def updateBestIndividual(newBestIndividual, improvementGeneration):
    
    if getBestFit() < newBestIndividual.fitness.values[0]:
        global best_ind
        best_ind = newBestIndividual # updating best individual
        #best_fit = newBestIndividual.fitness.values[0] # updating best fitness
        #global best_cmatrix 
        #best_cmatrix = newBestIndividual.confusion_matrix# updating best fitness
        global bestIndividuals
        bestIndividuals.append(newBestIndividual)
        global generationsWihImprovements
        generationsWihImprovements.append(improvementGeneration)
        #global bestFitnesses
        #bestFitnesses.append(best_fit)
        #global bestConfMatrices
        #bestConfMatrices.append(best_cmatrix)


def run_deap(all_attribs, 
             all_slice_amounts,
             all_output_classes,
             max_consecutive_slices=__DEFAULT_MAX_CONSEC_SLICES, # max length of the each slices range
             number_of_groupings=__DEFAULT_NUMBER_OF_GROUPINGS, # controls how many slices ranges there will be used
             current_experiment=1
             ):
 
    # Global Variables
    global __BODY_PLANES
    global __MAX_SLICES_VALUES
    global __MIN_SLICES_VALUES
    global __VERBOSE
    global __OUTPUT_DIRECTORY
    global __DEFAULT_MAX_CONSEC_SLICES
    
    if __VERBOSE:
        print('* Running experiment {0}'.format(current_experiment))
    
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
    print('\t* __OUTPUT_DIRECTORY = ',__OUTPUT_DIRECTORY)

    if 'FitnessMax' not in globals():
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if 'Individual' not in globals():
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
    
    # best indivudal tracking
    global best_ind
    #global best_fit
    #global best_conf_matrix
    #best_fit = -1.0
    best_ind = pop[0]
    
    # Queue (FIFO) where best individuals are saved in
    #global bestFitnesses
    global bestIndividuals
    #global bestConfMatrices
    global generationsWihImprovements
    
    
    
    # Evaluate initial population
    fits_and_matrices = list(map(toolbox.evaluate, pop))
    for ind, fit_and_matrix_tuple in zip(pop, fits_and_matrices):
        ind.fitness.values = (fit_and_matrix_tuple[0],)
        ind.confusion_matrix = fit_and_matrix_tuple[1]
        
        # Tracking new best individuals
        if ind.fitness.values[0] > getBestFit(): # It's a maximization problem
            best_ind = ind
            
            
    current_generation = 0
    updateBestIndividual(best_ind, current_generation) # New individual has found on the initial population
        
    print('Initial best individual: {0} (fitness={1})'.format(best_ind,best_ind.fitness.values[0]))
    
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
    
    for gen in list(range(1,number_of_generations + 1)):
        print('\n* Initializing {0}th generation of experiment={2}(current best fitness={1})...'.format(gen,best_ind.fitness.values[0],current_experiment))
        
        print('\t* Running variation operators...')
        offspring = algorithms.varAnd(pop, 
                                          toolbox, 
                                      __CROSSOVER_INDP, 
                                      __MUTATE_INDP)
        print('\t Done!')
        
        print('\n\t* Evaluating offspring...')
        fits_and_matrices = list(map(toolbox.evaluate,offspring)) # list of (fitness,conf_matrix) tuples
        print('\t Done!')
       
        
        print('\n\t* Updating fitness and confusion matrix of offspring...')
        for i in range(len(offspring)):
            # fitness should be a one element tuple
            fit = fits_and_matrices[i][0]
            offspring[i].fitness.values = fit, # first tuple element
            
            conf_matrix = fits_and_matrices[i][1]
            offspring[i].confusion_matrix =  conf_matrix # second tuple element
            
            
            # tracking new best individual
            if fit > best_ind.fitness.values[0]:
                print('\t New BEST Individual {0} with Fitness={1} was found!'.format(offspring[i],fit))
                updateBestIndividual(offspring[i],gen)
                
            
        
    if __VERBOSE:
        print('\t* Best Individuals Found:')
        for ind, gen in zip(bestIndividuals, generationsWihImprovements):
            print('\tbest_fit={0:.6f}\t best_ind={1}\t at generation={2}'.format(ind.fitness.values[0],ind,gen))
            
    
        print('\n\t*Evolution process has finished')
        
        print('\n\t*Saving experiment data to output file...')
    
    #exp_output_filename = build_experiment_output_filename(current_experiment,best_fit)
    
    #saving data to output file"
    saveExperimentDataToFile(current_experiment)
    
    print('Best Individual Found is: ', best_ind)
    print('\t* Fitness: ', getBestFit())
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
    csv_file_ok = False
    attribs_dir_ok = False
    verbose_ok = False
    multi_cpu_ok = False
    number_of_experiments = 1
    
    try:
        opts, args = getopt.getopt(argv[1:],"hc:d:vmn:",["csv=","dir=","verbose","multicpu","number_of_experiments="]) 
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

        
        global __ALARM, __FREQ, __DURATION
        number_of_groupings = __DEFAULT_NUMBER_OF_GROUPINGS
        
        print('Running experiments...')
        
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
    
