

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:35:46 2018

@author: rodrigo
"""


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split
from sklearn import metrics

import numpy as np
from deap import base, creator, tools, algorithms

import random, sys, os
import loadattribs 
#import knn_alzheimer_crossvalidate
#import evaluating_classifiers as ec
import getopt
import time
import datetime
import multiprocessing
from multiprocessing import Pool
import copy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt



def build_models_names_list():
    models_names = []
    models_names.append('KNN')
    models_names.append('LDA')
    models_names.append('CART')
    models_names.append('NB')
    models_names.append('SVM')
    models_names.append('RF')
    models_names.append('LR')
    return models_names

def build_models_dictionary(knn_k_value=3,lr_solver='sag',lr_multiclass='ovr'):
    models_dic = {}
    
    models_names = build_models_names_list()
    models_constructors = []
    
    models_constructors.append(KNeighborsClassifier(n_neighbors=knn_k_value))
    models_constructors.append(LinearDiscriminantAnalysis())
    models_constructors.append(DecisionTreeClassifier())
    models_constructors.append(GaussianNB())
    models_constructors.append(SVC())
    models_constructors.append(RandomForestClassifier())
    models_constructors.append(LogisticRegression(solver=lr_solver, multi_class=lr_multiclass))
    #models_constructors.append()
    
    for name, const in zip(models_names, models_constructors):
        models_dic[name]=const
        
    return models_dic


# Globla Slicing Arguments
#global __ALL_ATTRIBS, __ALL_OUTPUT_VALUES, __BODY_PLANES, __MAX_SLICES_VALUES, __DEFAULT_MAX_CONSEC_SLICES, __DEFAULT_NUMBER_OF_GROUPINGS

# Global Alzheimer Classification Problem Arguments
#global __GENES_LOW_LIMITS, __GENES_UP_LIMITS, __DEFAULT_KNN_K_VALUE, __VERBOSE

# Global Runtime Parameters
#global __MULTI_CPU_USAGE

# Global Evolutionary Parameters
#global __TOURNEAMENT_SIZE, __MUTATE_INDP, __CROSSOVER_INDP, __NUMBER_OF_GENERATIONS, __POPULATION_SIZE, __DEFAULT_TARGET_FITNESS, __DEFAULT_WORST_FITNESS

# Data Cohort
__VALID_GENDERS = ['M','F']
__MIN_AGE = 0.0
__MAX_AGE = 200.0


# Image Black List
#BLACK_LIST_ID = []
__BLACK_LIST_ID = ['I288905','I288906','I120446','I120441','I120426','I120436','I120423','I120416']

# Slicing Arguments
__ALL_ATTRIBS = []
__ALL_OUTPUT_VALUES = []
__BODY_PLANES = []
__MAX_SLICES_VALUES = []
__MIN_SLICES_VALUES = []
__DEFAULT_MAX_CONSEC_SLICES = 20 #20
__DEFAULT_NUMBER_OF_GROUPINGS = 1

# Classifier parameters
__MODEL_NAME = 'KNN'
#    models = []
#    models.append(('KNN', KNeighborsClassifier(n_neighbors=knn_k_value)))
#    models.append(('LDA', LinearDiscriminantAnalysis()))
#    models.append(('CART', DecisionTreeClassifier()))
#    models.append(('NB', GaussianNB()))
#    models.append(('SVM', SVC()))
#    models.append(('RF',RandomForestClassifier()))
#    models.append(('LR', LogisticRegression(solver=lr_solver, multi_class=lr_multiclass)))


# Default Specific Parameters for each Classifier
__DEFAULT_KNN_K_VALUE = 3
__DEFAULT_LR_SOLVER = 'sag'
__DEFAULT_LR_MULTICLASS = 'ovr'


__MODELS = build_models_dictionary(
        knn_k_value=-__DEFAULT_KNN_K_VALUE,
        lr_solver=__DEFAULT_LR_SOLVER,
        lr_multiclass=__DEFAULT_LR_MULTICLASS)
__MODEL_CONSTRUCTOR = __MODELS[__MODEL_NAME]


__USE_RESCALING = True
__USE_SMOTE = True
__USE_PCA = True
__CV_TYPE = 'kcv'
__CV_MULTI_THREAD = False
__CV_SHUFFLE = True
__KCV_FOLDS = 10
__USE_STRATIFIED_KFOLD = True
__MAXIMIZATION_PROBLEM = True

# Runtime Parameters
__DEAP_RUN_ID = ''
__OUTPUT_DIRECTORY = './'
__MULTI_CPU_USAGE = False
__VERBOSE = False
__CORES_NUM = 1

# Default Evolutionary Parameters
__MUTATE_INDP = 0.10
__CROSSOVER_INDP = 0.40
__POPULATION_SIZE = 200
__NUMBER_OF_GENERATIONS = 100
__MAX_GENERATIONS_WITHOUT_IMPROVEMENTS = 10
__TOURNEAMENT_SIZE_IS_DYNAMIC = False
__TOURNEAMENT_UPDATE_LIMIT = 10
__TOURNEAMENT_LAST_UPDATE = 0
#__TOURNEAMENT_SIZE_INCREMENT_VALUE = int(__POPULATION_SIZE * 0.02)
__TOURNEAMENT_SIZE_INCREMENT_VALUE = 4
#__INITIAL_TOURNEAMENT_SIZE = int(__POPULATION_SIZE * 0.02)
__INITIAL_TOURNEAMENT_SIZE = 10
#__MAX_TORNEAMENT_SIZE = int(__POPULATION_SIZE * 0.20)
__MAX_TORNEAMENT_SIZE = 24
__TOURNEAMENT_SIZE = __INITIAL_TOURNEAMENT_SIZE

__DEFAULT_TARGET_FITNESS = 0.0
__DEFAULT_WORST_FITNESS = -1.0
__GENES_LOW_LIMITS = [0,0,1]
__GENES_UP_LIMITS = [2,160,20]
__SEEDS_FILE = ''

# Alarm Variables
__ALARM = True
__DURATION = 1 #seconds
__FREQ = 440 # Hz


def model_name_is_valid(model_name):
    names = build_models_names_list()
    try:
        names.index(model_name)
        return True
    except ValueError:
        return False
    


def build_gender_to_num_dic():
    return {'M':1, 'F':-1}

def num_genders_list(all_genders):
    dic = build_gender_to_num_dic()
    num_genders_list = []
    for gender in all_genders:
        num_genders_list.append(dic[gender])
    return num_genders_list

def update_tourneament_size(current_gen, last_improvement_gen, toolbox):
    global __TOURNEAMENT_SIZE, __TOURNEAMENT_SIZE_IS_DYNAMIC, __TOURNEAMENT_LAST_UPDATE
    global __VERBOSE, __TOURNEAMENT_SIZE_INCREMENT_VALUE, __INITIAL_TOURNEAMENT_SIZE, __TOURNEAMENT_INCREMENTS 
    
    if __TOURNEAMENT_SIZE_IS_DYNAMIC:
        gens_without_improvements = current_gen - last_improvement_gen
        last_update_was_recent = (current_gen - __TOURNEAMENT_LAST_UPDATE) <= __TOURNEAMENT_UPDATE_LIMIT
        still_can_grow = __MAX_TORNEAMENT_SIZE >= __TOURNEAMENT_SIZE + __TOURNEAMENT_SIZE_INCREMENT_VALUE
        
        if gens_without_improvements > __TOURNEAMENT_UPDATE_LIMIT and still_can_grow and not last_update_was_recent:
            __TOURNEAMENT_LAST_UPDATE = current_gen
            __TOURNEAMENT_SIZE = __TOURNEAMENT_SIZE + __TOURNEAMENT_SIZE_INCREMENT_VALUE
            toolbox.unregister('select')
            toolbox.register("select", tools.selTournament, tournsize=__TOURNEAMENT_SIZE) # selection
            if __VERBOSE:
                print('\t* Generation {1:3d}: New Torneament Size={0:2d}'.format(__TOURNEAMENT_SIZE,current_gen))
    return __TOURNEAMENT_SIZE
            

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

#####################################################################################
def get_data_partition(individual, all_attribs, all_slice_amounts, all_genders, all_ages, debug=__VERBOSE):
    all_groupings_partitions_list = []
    
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
                
                g = g + 3 # (step to another slice grouping)
    else:
        raise ValueError('Bad formatted individual: slices grouping length ({0}) must be a multiple of three.\nIndividual = {1}'.format(ind_size,individual))
        
    # Merging all partitions data
    all_partitions_merged = all_groupings_partitions_list[0] # getting first
    for i in range(1,len(all_groupings_partitions_list)):
        all_partitions_merged = all_partitions_merged + all_groupings_partitions_list[i]
    
    return all_partitions_merged

def buildDataFrames(X_data, y_data, all_genders, all_ages, debug=False):
    # Data preparation
    try:
        new_dimensions = (X_data.shape[0],
                          X_data.shape[1]*X_data.shape[2])
    except IndexError:
        print('** IndexValue exception')
        print('\tX_data.shape=',X_data.shape)
        print('\ty_data.shape=',y_data.shape)
        print('\t')
        sys.exit(-1)
    
    if __VERBOSE and debug: 
        print('* Reshaping for this data partition with these dimensions:',new_dimensions)

    new_partition = np.reshape(X_data, new_dimensions)
        
    if __VERBOSE and debug: 
        print('...done')
        print('* The shape of a line data retrived from the new partition=', new_partition[0].shape)
    
    import pandas as pd
    X_pandas = pd.DataFrame(data=new_partition)
    y_pandas = pd.DataFrame(data=y_data)
    
    # Reconding genders to numeric format
    num_genders = num_genders_list(all_genders)
    #num_genders = all_genders
    
    # numeric Gender and Age columns inclusion
    genders_series = pd.Series(num_genders,name='gender')
    age_series = pd.Series(all_ages,name='age')
    
    # Concatanation
    X_pandas = pd.concat([X_pandas, genders_series, age_series],axis=1,ignore_index=True)
    
    # Formating all data as float
    X_pandas = X_pandas.astype(dtype=np.float64)
    
    return X_pandas, y_pandas

# Evaluates a individual instance represented by Slices Groupings 
def evaluateSlicesGroupingsNOVO(individual, # list of integers
                               all_attribs,
                               all_slice_amounts,
                               output_classes,
                               all_genders,
                               all_ages,
                               model_name=__MODEL_NAME,
                               debug=__VERBOSE):
    
    all_groupings_partitions_list = [] 
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
    
    
    global __USE_RESCALING, __USE_SMOTE, __CV_TYPE, __CV_MULTI_THREAD, __KCV_FOLDS, __MODEL_CONSTRUCTOR
    
    # Formating data as Pandas DataFrame
    X_pandas, y_pandas = buildDataFrames(all_partitions_merged, output_classes, all_genders, all_ages)

    dicionary_results = evaluate_model(X_pandas, y_pandas , model_name,
                                       folds=__KCV_FOLDS, cv_seed=7, cv_shuffle=__CV_SHUFFLE,
                                       smote=__USE_SMOTE, rescaling=__USE_RESCALING, cores_num=1, 
                                       maximization=__MAXIMIZATION_PROBLEM, stratified_kfold=__USE_STRATIFIED_KFOLD,
                                       debug=__VERBOSE)
    accuracy = dicionary_results['mean_acc']
    conf_matrix = dicionary_results['conf_matrix']
    
    
    return accuracy, conf_matrix

def evaluateSlicesGroupings(individual, all_attribs, all_slice_amounts,
                           output_classes, all_genders, all_ages, 
                           model_name, debug=__VERBOSE):

    if individual.fitness.valid:
            
        all_partitions_merged = get_data_partition(individual, all_attribs, all_slice_amounts, all_genders, all_ages, debug=__VERBOSE)
        
        global __USE_RESCALING, __USE_SMOTE, __CV_TYPE, __CV_MULTI_THREAD, __KCV_FOLDS, __CV_SHUFFLE
        global __MAXIMIZATION_PROBLEM, __CORES_NUM, __USE_STRATIFIED_KFOLD
        
        # Formating data as Pandas DataFrame
        X_pandas, y_pandas = buildDataFrames(all_partitions_merged, output_classes)
    
        # Preparing cross-validation
        #cv = KFold(n_splits=__KCV_FOLDS, random_state=42, shuffle=True)
        #all_train_and_test_indexes = cv.split(X_pandas)
        
        dicionary_results = evaluate_model(X_pandas, y_pandas , model_name,
                                           folds=__KCV_FOLDS, cv_seed=7, cv_shuffle=__CV_SHUFFLE,
                                           smote=__USE_SMOTE, rescaling=__USE_RESCALING, cores_num=1, 
                                           maximization=__MAXIMIZATION_PROBLEM, stratified_kfold=__USE_STRATIFIED_KFOLD,
                                           pca=__USE_PCA,
                                           debug=__VERBOSE)
        accuracy = dicionary_results['mean_acc']
        conf_matrix = dicionary_results['confusion_matrix']

    
    return accuracy, conf_matrix


def evaluate_model(X_data, y_data, model_name,
                   folds, cv_seed=7, cv_shuffle=True,
                   smote=True, rescaling=True, cores_num=1, 
                   maximization=True, stratified_kfold=True,
                   pca=False, debug=False):
    
    all_acc = []
    #all_cmat = []
    
    start_time = time.time()
    
    # STEP 1: perform rescaling if required
    if rescaling:
        
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        X_fixed = scaler.fit_transform(X_data) # Fit your data on the scaler object
    else:
        X_fixed = X_data
        
    # Added to solve column-vector issue
    y_fixed = np.ravel(y_data)
     
    # Validation setup

    from sklearn import model_selection 
#    cv = model_selection.KFold(n_splits=folds, random_state=cv_seed, shuffle=cv_shuffle)
#    both_indexes = cv.split(X_data)
    
    if stratified_kfold:
        cv = model_selection.StratifiedKFold(n_splits=folds, random_state=cv_seed, shuffle=cv_shuffle)
        both_indexes = cv.split(X_data, y_data)
    else:
        cv = model_selection.KFold(n_splits=folds, random_state=cv_seed, shuffle=cv_shuffle)
        both_indexes = cv.split(X_data)
    
    num_classes = len(np.unique(np.array(y_data))) # number of classes
    conf_matrix = np.zeros(dtype=np.int64, shape=[num_classes,num_classes])
    
    #for train_indexes, test_indexes in all_train_and_test_indexes:
    for train_indexes, test_indexes in both_indexes:
        
        # STEP 2: split data between test and train sets
        X_train = X_fixed[train_indexes]
        X_test = X_fixed[test_indexes]
        y_train = y_fixed[train_indexes]
        y_test = y_fixed[test_indexes]
        
        # OPTIONAL: Apply PCA to reduce dimensionality
        if pca:
            from sklearn.decomposition import PCA
            pca = PCA(.95)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
        
        # STEP 3: oversampling training data using SMOTE if required
        if smote:
            from imblearn.over_sampling import SMOTE
            smt = SMOTE() 
            X_train, y_train = smt.fit_sample(X_train, y_train)
    
        models_dic = build_models_dictionary()
        model = models_dic[model_name] # model receives a object from a class who implements the .fit method
        
        # STEP 4: Training (Fit) Model
        model.fit(X_train, y_train)
        
        # STEP 5: Testing Model (Making Predictions)
        y_pred = model.predict(X_test) # testing
        
        # STEP 6: Building Evaluation Metrics
        acc = metrics.accuracy_score(y_test, y_pred)
        cmat = metrics.confusion_matrix(y_test,y_pred,labels=None,sample_weight=None)
#        print('acc={0:.4}'.format(acc))
        all_acc.append(acc)
        conf_matrix = conf_matrix + np.array(cmat)
        #cv_results = model_selection.cross_val_score(model, X_data, y_data, cv=kfold, scoring=metric, n_jobs=cores_num)
    	
    # Converting to Numpy Array to use its statiscs pre-built functions
    np_all_acc = np.array(all_acc)
#    print('length of all_acc={0} and np_all_acc={1}'.format(len(all_acc),len(np_all_acc)))
    
    # Finding position of the best and the worst individual
    best_acc_pos  = np.argmax(np_all_acc) if maximization else np.argmin(np_all_acc)
    worst_acc_pos = np.argmin(np_all_acc) if maximization else np.argmax(np_all_acc)
    median_pos = folds//2
    
    best_acc = np_all_acc[best_acc_pos]
    #best_cmat = all_cmat[best_acc_pos]
    worst_acc = all_acc[worst_acc_pos]
    #worst_cmat = all_cmat[worst_acc_pos]
    
    mean_acc = np_all_acc.mean()
    
    median_acc = np_all_acc[median_pos] #np_all_acc[folds//2] if folds % 2 == 1 else (np_all_acc[(folds+1)//2] + np_all_acc[(folds-1)//2])//2
    #median_cmat = all_cmat[median_pos]
    
    std_acc = np_all_acc.std()
    
    # Calculing execution time
    end_time = time.time()
    total_time = end_time - start_time
    
    #dic = {'name':name, 'mean_acc':mean_acc, 'std_acc':std_acc, 'best_acc':best_acc, 
    #'best_cmat':best_cmat, 'worst_acc':worst_acc, 'worst_cmat':worst_cmat, 'total_time':total_time, 'all_acc':np_all_acc, 'all_cmat':all_cmat, 'median_acc':median_acc, 'median_cmat':median_cmat}
    
    #metrics_list = [model_name,mean_acc,best_acc,std_acc,best_cmat,worst_acc,worst_cmat,total_time,all_acc,all_cmat,median_acc,median_cmat]
    metrics_list = [model_name,mean_acc,best_acc,std_acc,worst_acc,conf_matrix,total_time,all_acc,median_acc]

    metrics_names = all_metrics_names()

    dic = {}
    for metric_num in range(len(metrics_names)):
        name = metrics_names[metric_num]
        dic[name] = metrics_list[metric_num]
        
#    dic['name'] = name
#    dic['mean_acc'] = mean_acc
#    dic['std_acc'] = std_acc
#    dic['best_acc'] = best_acc
#    dic['best_cmat'] = best_cmat
#    dic['worst_acc'] = worst_acc
#    dic['worst_cmat'] = worst_cmat
#    dic['total_time'] = total_time
#    dic['all_acc'] = all_acc
#    dic['all_cmat'] = all_cmat
#    dic['median_acc'] = median_acc
#    dic['median_cmat'] = median_cmat
    
    return dic    
#####################################################################################

# Evaluates a individual instance represented by Slices Groupings 
#def evaluateSlicesGroupingsKNN(individual, # list of integers
#                               all_attribs,
#                               all_slice_amounts,
#                               output_classes,
#                               k_value=__DEFAULT_KNN_K_VALUE,
#                               debug=__VERBOSE):
#    
#    all_groupings_partitions_list = [] 
#    accuracy = 0.0
#    conf_matrix = [[1,0,0],[0,1,0],[0,0,1]] # typical confusion matrix for the Alzheimer classification problem     
#    ind_size = len(individual)
#    
#    if ind_size % 3 == 0:    
#        # Getting data from a slices grouping
#        for g in list(range(ind_size)):
#            if g % 3 == 0:
#                plane = individual[g]
#                first_slice = individual[g+1]
#                total_slices = individual[g+2]
#                                
#                partition = loadattribs.getAttribsPartitionFromSingleSlicesGrouping(all_attribs,all_slice_amounts,plane,first_slice,total_slices)
#                
#                # append data to a list which will be merged later
#                all_groupings_partitions_list.append(partition)
#                
#                g = g + 3
#    else:
#        raise ValueError('Bad formatted individual: slices grouping length ({0}) must be a multiple of three.\nIndividual = {1}'.format(ind_size,individual))
#        
#    # Merging all partitions data
#    all_partitions_merged = all_groupings_partitions_list[0]
#    for i in range(1,len(all_groupings_partitions_list)):
#        all_partitions_merged = all_partitions_merged + all_groupings_partitions_list[i]
#    
#    
#    global __USE_RESCALING, __USE_SMOTE, __CV_TYPE, __CV_MULTI_THREAD, __KCV_FOLDS, __MODEL_CONSTRUCTOR
#    
#    # Classifying merged data
#    accuracy, conf_matrix = knn_alzheimer_crossvalidate.runMODEL(
#            all_partitions_merged,
#            output_classes,
#            k_value,
#            knn_debug=debug,
#            use_smote=__USE_SMOTE,
#            use_rescaling=__USE_RESCALING,
#            cv_type=__CV_TYPE,
#            kcv_value=__KCV_FOLDS,
#            use_Pool=__CV_MULTI_THREAD,
#            model=__MODEL_CONSTRUCTOR)
#    
#    
#    return accuracy, conf_matrix


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
    #print('up_limits_per_slice_grouping=',up_limits_per_slice_grouping)
    
    all_up_limits = []
    all_low_limits = []
    for n in range(number_of_slices_groupings):
        all_up_limits = all_up_limits + up_limits_per_slice_grouping
        all_low_limits = all_low_limits + low_limits_per_slice_grouping
        #print('all_up_limits=',all_up_limits)
        
    global __GENES_LOW_LIMITS
    global __GENES_UP_LIMITS
    
    __GENES_LOW_LIMITS = all_low_limits
    __GENES_UP_LIMITS = all_up_limits
    


def build_parameters_string(max_consecutive_slices,number_of_groupings):
    strPool = []

    # MODEL CHOOSE
    global __MODEL_NAME,__MODEL_CONSTRUCTOR
    strPool.append('\n# Model Selection parameters\n')
    strPool.append('__MODEL_NAME = {0}\n'.format(__MODEL_NAME))
    strPool.append('__MODEL_CONSTRUCTOR = {0}\n'.format(__MODEL_CONSTRUCTOR))
    
    # DATA COHORT
    strPool.append('\n# Data Cohort parameters\n')
    strPool.append('__VALID_GENDERS = {0}\n'.format(__VALID_GENDERS))
    strPool.append('__MIN_AGE = {0}\n'.format(__MIN_AGE))
    strPool.append('__MAX_AGE = {0}\n'.format(__MAX_AGE))
    
    # BLACKLIST
    global __BLACK_LIST_ID
    strPool.append('\n# Images Black List\n')
    strPool.append('__BLACK_LIST_ID = {0}\n'.format(__BLACK_LIST_ID))
    
    # Runtime parameters    
    global __DEAP_RUN_ID, __MULTI_CPU_USAGE, __OUTPUT_DIRECTORY, __VERBOSE, __CORES_NUM
    strPool.append('\n# Runtime parameters\n')
    strPool.append(' __DEAP_RUN_ID ={0}\n'.format(__DEAP_RUN_ID))
    strPool.append(' __MULTI_CPU_USAGE  ={0}\n'.format(__MULTI_CPU_USAGE ))
    strPool.append(' __OUTPUT_DIRECTORY ={0}\n'.format(__OUTPUT_DIRECTORY))
    strPool.append('__VERBOSE = {0}\n'.format(__VERBOSE ))
    strPool.append('__CORES_NUM = {0}\n'.format(__CORES_NUM ))
    
    # Classifier pipeline parameters
    global __USE_RESCALING, __USE_SMOTE, __CV_TYPE, __CV_MULTI_THREAD, __CV_SHUFFLE
    global __KCV_FOLDS, __MAXIMIZATION_PROBLEM, __USE_PCA, __USE_STRATIFIED_KFOLD
    strPool.append('\n# Classifier pipeline parameters\n')
    strPool.append('__USE_RESCALING = {0}\n'.format(__USE_RESCALING))
    strPool.append('__USE_SMOTE = {0}\n'.format(__USE_SMOTE))
    strPool.append('__USE_PCA = {0}\n'.format(__USE_PCA))
    
    strPool.append('__CV_TYPE = {0}\n'.format(__CV_TYPE))
    strPool.append('__CV_MULTI_THREAD = {0}\n'.format(__CV_MULTI_THREAD))
    strPool.append('__CV_SHUFFLE = {0}\n'.format(__CV_SHUFFLE))
    strPool.append('__KCV_FOLDS = {0}\n'.format(__KCV_FOLDS))
    strPool.append('__USE_STRATIFIED_KFOLD = {0}\n'.format(__USE_STRATIFIED_KFOLD))
    strPool.append('__MAXIMIZATION_PROBLEM = {0}\n'.format(__MAXIMIZATION_PROBLEM))
    
    
    # Specific models parameters
    global __DEFAULT_KNN_K_VALUE, __DEFAULT_LR_SOLVER, __DEFAULT_LR_MULTICLASS
    strPool.append(' __DEFAULT_KNN_K_VALUE ={0}\n'.format(__DEFAULT_KNN_K_VALUE))
    strPool.append(' __DEFAULT_LR_SOLVER ={0}\n'.format(__DEFAULT_LR_SOLVER))
    strPool.append(' __DEFAULT_LR_MULTICLASS ={0}\n'.format(__DEFAULT_LR_MULTICLASS))

    # Slicing parameters
    global __BODY_PLANES, __MAX_SLICES_VALUES, __MIN_SLICES_VALUES, __DEFAULT_MAX_CONSEC_SLICES
    strPool.append('\n# Slicing parameters\n')
    strPool.append(' max_consecutive_slices = {0}\n'.format(max_consecutive_slices))
    strPool.append(' number_of_groupings = {0}\n'.format(number_of_groupings))
    strPool.append(' __BODY_PLANES = {0}\n'.format(__BODY_PLANES))
    strPool.append(' __MAX_SLICES_VALUES = {0}\n'.format(__MAX_SLICES_VALUES))
    strPool.append(' __MIN_SLICES_VALUES = {0}\n'.format(__MIN_SLICES_VALUES))
    #strPool.append(' __DEFAULT_MAX_CONSEC_SLICES={0}\n'.format(__DEFAULT_MAX_CONSEC_SLICES))
    
    
    # Evolutionary arguments
    global __TOURNEAMENT_SIZE, __MUTATE_INDP, __CROSSOVER_INDP, __POPULATION_SIZE, __NUMBER_OF_GENERATIONS 
    global __MAX_GENERATIONS_WITHOUT_IMPROVEMENTS, __DEFAULT_TARGET_FITNESS, __DEFAULT_WORST_FITNESS
    global  __GENES_LOW_LIMITS, __GENES_UP_LIMITS, __TOURNEAMENT_SIZE_IS_DYNAMIC
    strPool.append('\n# Evolutonary parameters\n')
    strPool.append(' __TOURNEAMENT_SIZE_IS_DYNAMIC = {0}\n'.format(__TOURNEAMENT_SIZE_IS_DYNAMIC))
    strPool.append(' __TOURNEAMENT_SIZE = {0}\n'.format(__TOURNEAMENT_SIZE))
    strPool.append(' __MUTATE_INDP = {0}\n'.format(__MUTATE_INDP))
    strPool.append(' __CROSSOVER_INDP = {0}\n'.format(__CROSSOVER_INDP ))
    strPool.append(' __POPULATION_SIZE = {0}\n'.format(__POPULATION_SIZE ))
    strPool.append(' __NUMBER_OF_GENERATIONS = {0}\n'.format(__NUMBER_OF_GENERATIONS))
    strPool.append(' __MAX_GENERATIONS_WITHOUT_IMPROVEMENTS = {0}\n'.format(__MAX_GENERATIONS_WITHOUT_IMPROVEMENTS ))
    #strPool.append(' __DEFAULT_TARGET_FITNESS = {0}\n'.format(__DEFAULT_TARGET_FITNESS))
    #strPool.append(' __DEFAULT_WORST_FITNESS = {0}\n'.format(__DEFAULT_WORST_FITNESS))
    strPool.append(' __GENES_LOW_LIMITS = {0}\n'.format(__GENES_LOW_LIMITS))
    strPool.append(' __GENES_UP_LIMITS = {0}\n'.format(__GENES_UP_LIMITS))


    return strPool
    

def setRunID():
    global __DEAP_RUN_ID
    __DEAP_RUN_ID = str(datetime.date.today()) + '_' + str(int(round(time.time())))
    return __DEAP_RUN_ID

def saveParametersFile(max_consec_slices,num_groupings):
    global __OUTPUT_DIRECTORY, __DEAP_RUN_ID
    #output_dir_path = os.path.join(__OUTPUT_DIRECTORY, build_experiments_output_dir_name())
    output_dir_path = build_experiments_output_dir_name()
    
    
    filename = 'parameters_{0}.txt'.format(__DEAP_RUN_ID)
    param_full_filename = os.path.join(output_dir_path, filename)
    
    append_mode = "a"
    blank_file = False
    
    # checking parameters file
    if not os.path.exists(param_full_filename):
        blank_file = True
        
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
    
    return param_full_filename


def save_final_result_boxplot(best_inds,labels,title='Title',debug=__VERBOSE,bckp=None):
    best_fits = []
    for ind in best_inds:
        best_fits.append(ind.fitness.values[0])
    
    #best_fits = bckp
    
    global __MODEL_NAME, __DEAP_RUN_ID
    fig,ax = plt.subplots()
    plt.title(title)
    ax.boxplot(best_fits,labels=labels)
    
    best_of_bests_position = np.argmax(np.array(best_fits))
    best_of_bests = best_fits[best_of_bests_position]
    img_filename = 'final_result-acc_{0}-model_{1}-run_{2}.png'.format(best_of_bests, __MODEL_NAME,__DEAP_RUN_ID)
    
    output_dir_path = build_experiments_output_dir_name()
    
    
    img_full_filename = os.path.join(output_dir_path, img_filename)
    if __VERBOSE: 
        print('\n* Saving final result as bloxplot imagem in: {0}'.format(img_full_filename))
    
    fig.savefig(img_full_filename)
    
    if __VERBOSE: 
        print('\t Done')
    



def saveResultsCSVFile(all_experiments_best_ind):
    global __OUTPUT_DIRECTORY, __DEAP_RUN_ID
    output_dir_path = build_experiments_output_dir_name()
    
    
    filename = 'results_{0}_{1}.csv'.format(__DEAP_RUN_ID, __MODEL_NAME)
    results_full_filename = os.path.join(output_dir_path, filename)
    
    append_mode = "a"
    blank_file = False
    
    # checking parameters file
    if not os.path.exists(results_full_filename):
        blank_file = True
        
        # creates output dir when path doesnt exist
        if output_dir_path != '' and not os.path.exists(output_dir_path):
            try:
                os.makedirs(output_dir_path)
            except os.error:
                print ('*** ERROR: Output directory (%s) can not be created\n' % output_dir_path)
                sys.exit(1)
        
    if __VERBOSE: 
        print('\n* Saving best results of each experiment in: {0}'.format(results_full_filename))
    
    # Writting to output file
    try :
        
        file_handler = open(results_full_filename, append_mode)
        if blank_file:
            head = 'Experiment,Best_Individual,Best_Fit,Best_ConfMatrix\n'
            file_handler.write(head)
        
        all_best_fit = []
        for exp_num in range(1, len(all_experiments_best_ind)+1):
            ind = all_experiments_best_ind[exp_num-1]
            best_fit = ind.fitness.values[0]
            ind_without_commas = ' '.join(map(str, np.array(ind)))
            
            flatten_cmat =  ' '.join(map(str,np.array(ind.confusion_matrix).flatten()))
            line = '{0},{1},{2},{3}\n'.format(exp_num, ind_without_commas, ind.fitness.values[0], flatten_cmat)
            file_handler.write(line)
            
            all_best_fit.append(best_fit)
        file_handler.close()
            
    except os.error:
        print("\n*** ERROR: file %s can not be written" % results_full_filename)
        exit(1)
        

    if __VERBOSE: 
        print('\t Done')        
    
    return results_full_filename

def saveDetailedResultsCSVFile(all_experiments_best_ind):
    global __OUTPUT_DIRECTORY, __DEAP_RUN_ID
    output_dir_path = build_experiments_output_dir_name()
    
    
    filename = 'detailed_results_{0}_{1}.csv'.format(__DEAP_RUN_ID, __MODEL_NAME)
    results_full_filename = os.path.join(output_dir_path, filename)
    
    append_mode = "a"
    blank_file = False
    
    # checking parameters file
    if not os.path.exists(results_full_filename):
        blank_file = True
        
        # creates output dir when path doesnt exist
        if output_dir_path != '' and not os.path.exists(output_dir_path):
            try:
                os.makedirs(output_dir_path)
            except os.error:
                print ('*** ERROR: Output directory (%s) can not be created\n' % output_dir_path)
                sys.exit(1)
        
    if __VERBOSE: 
        print('\n* Saving best results of each experiment in: {0}'.format(results_full_filename))
    
    # Writting to output file
    try :
        
        file_handler = open(results_full_filename, append_mode)
        if blank_file:
            conf_matrix_head = 'true_CN,CN_as_MCI,CN_as_AD,MCI_as_CN,true_MCI,MCI_as_AD,AD_as_CN,AD_as_MCI,true_AD'
            head = 'Experiment,Body_Plane,First_Slice,Slices_Amount,Best_Fit,' + conf_matrix_head + '\n'
            file_handler.write(head)
        
        all_best_fit = []
        for exp_num in range(1, len(all_experiments_best_ind)+1):
            ind = all_experiments_best_ind[exp_num-1]
            best_fit = ind.fitness.values[0]
            ind_without_commas = ','.join(map(str, np.array(ind)))
            flatten_cmat =  ','.join(map(str,np.array(ind.confusion_matrix).flatten()))
            line = '{0},{1},{2},{3}\n'.format(exp_num, ind_without_commas, ind.fitness.values[0], flatten_cmat)
            file_handler.write(line)
            
            all_best_fit.append(best_fit)
        file_handler.close()
            
    except os.error:
        print("\n*** ERROR: file %s can not be written" % results_full_filename)
        exit(1)
        

    if __VERBOSE: 
        print('\t Done')        
    
    return results_full_filename


def build_experiments_output_dir_name():
    
    global __OUTPUT_DIRECTORY, _DEAP_RUN_ID, __MODEL_NAME
    global __DEFAULT_MAX_CONSEC_SLICES, __DEFAULT_KNN_K_VALUE
    global __MUTATE_INDP, __CROSSOVER_INDP, __POPULATION_SIZE
    
    run_str = 'run_{0}_{1}'.format(__DEAP_RUN_ID, __MODEL_NAME)
    slash = '/'
    
    parent_dir = __OUTPUT_DIRECTORY # usually './' or '../'
    if parent_dir.endswith('/'):
        slash = ''
    
    full_dir_path = parent_dir + slash + run_str
    return full_dir_path

def build_experiment_output_filename(exp_num, best_accuracy):
    acc_value = '{0:.2f}'.format(best_accuracy)
    
    full_dir_path = build_experiments_output_dir_name()
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
        
        flatten_cmat = ' '.join(map(str, np.array(best_ind.confusion_matrix)))
        #flatten_cmat = ' '.join(np.array(best_ind.confusion_matrix))
        line = '{0:3d}, {1:.8f}, {2}, {3}\n'.format(gen_num, bfit, best_ind, flatten_cmat)
        output_file.write(line)
        output_file.close()
    except os.error:
        output_file.close()
        print(" *** ERROR: file %s can not be written" % exp_output_full_filename)
        exit(1)
    
    return exp_output_full_filename

def all_metrics_names():
    metrics = []
    metrics.append('name')
    metrics.append('mean_acc')
    metrics.append('std_acc')
    metrics.append('best_acc')
    #metrics.append('best_cmat')
    metrics.append('worst_acc')
    metrics.append('conf_matrix')
    metrics.append('total_time')
    metrics.append('all_acc_np')
    #metrics.append('all_cmat')
    metrics.append('median_acc')
    #metrics.append('median_cmat')
    return metrics


def saveExperimentsDataToFile(exp_num, best_ind, bestIndividuals, generationsWithImprovements, runID):
    best_accuracy = best_ind.fitness.values[0]
    ofile = build_experiment_output_filename(exp_num,best_accuracy)[0]
    print('Saving output file {0}'.format(ofile))
    
    if len(bestIndividuals) == len(generationsWithImprovements):
    
        for ind, gnum in zip(bestIndividuals, generationsWithImprovements,):
            #cmat = ind.confusion_matrix
            append_experiment_data_to_output_file(exp_num,gnum,ofile,ind)
        
    else:
        print('Problem with results format:\n')
        print('len(bestIndividuals)={0}, len(generationsWithImprovements)={1}'.format(len(bestIndividuals), len(generationsWithImprovements)))
        sys.exit(1)
    
    return ofile


def run_deap(all_attribs, 
             all_slice_amounts,
             all_output_classes,
             all_genders,
             all_ages,
             max_consecutive_slices=__DEFAULT_MAX_CONSEC_SLICES, # max length of the each slices range
             number_of_groupings=__DEFAULT_NUMBER_OF_GROUPINGS, # controls how many slices ranges there will be used
             current_experiment=1,
             ):
 
    generationsWithImprovements = []
    
    bestIndividuals = []
    best_ind = None

    lastGenWithImprovements = 0
    
    # Runtime global Variables
    global __DEAP_RUN_ID
    global __MULTI_CPU_USAGE
    global __VERBOSE
    global __OUTPUT_DIRECTORY
    global __MODEL_NAME
    global __USE_PCA
    global __MODEL_CONSTRUCTOR
    
    # Slicing global Variables
    global __BODY_PLANES
    global __MAX_SLICES_VALUES
    global __MIN_SLICES_VALUES

    global __DEFAULT_MAX_CONSEC_SLICES
    
    if __VERBOSE:
        print('* Starting experiment {0}'.format(current_experiment))
    
#    # Updating global variables
#    __BODY_PLANES = loadattribs.getBplanes(all_slice_amounts)
#    __MIN_SLICES_VALUES,__MAX_SLICES_VALUES = loadattribs.getSliceLimits(all_slice_amounts)
#    #print('__MIN_SLICES_VALUES=',__MIN_SLICES_VALUES,'\n__MIN_SLICES_VALUES=',__MIN_SLICES_VALUES)
    
    updateGeneBounds(__BODY_PLANES, __MIN_SLICES_VALUES, max_consecutive_slices, __DEFAULT_NUMBER_OF_GROUPINGS, __VERBOSE)
    
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
    
#    toolbox.register('evaluate',
#                     evaluateSlicesGroupingsKNN,
#                     all_attribs = all_attribs,
#                     all_slice_amounts = all_slice_amounts,
#                     output_classes = all_output_classes,
#                     k_value = __DEFAULT_KNN_K_VALUE,
#                     debug=False)
    

    toolbox.register('evaluate',
                     evaluateSlicesGroupingsNOVO,
                     all_attribs = all_attribs,
                     all_slice_amounts = all_slice_amounts,
                     output_classes = all_output_classes,
                     all_genders = all_genders,
                     all_ages = all_ages,
                     #k_value = __DEFAULT_KNN_K_VALUE,
                     debug=False)
            

    
    
#    toolbox.register('evaluate',
#                     evaluateSlicesGroupings,
#                     all_attribs = all_attribs,
#                     all_slice_amounts = all_slice_amounts,
#                     output_classes = all_output_classes,
#                     k_value = __DEFAULT_KNN_K_VALUE,
#                     debug=False)
#            
    # defining population as a plane list
    toolbox.register('population',
                     tools.initRepeat,
                     list,
                     toolbox.individual)    
    
    # creating initial individuals
    pop = toolbox.population(n=__POPULATION_SIZE)
    
   
    # Initializing variables

    
    # Evaluate initial population
    fits_and_matrices = list(map(toolbox.evaluate, pop))
    
    
    for ind, fit_and_cmatrix_tuple in zip(pop, fits_and_matrices):
        ind.fitness.values = fit_and_cmatrix_tuple[0],
        ind.confusion_matrix = fit_and_cmatrix_tuple[1]
        
    
    
    # tracking initial best individual
    for ind in pop:
        if best_ind == None:
            best_ind = copy.deepcopy(pop[0])
        elif ind.fitness.values[0] > best_ind.fitness.values[0]:
            best_ind = copy.deepcopy(ind)

    current_generation = 0
    #print('# Initial best_ind=',best_ind,' Exp=',current_experiment,' Gen=',current_generation)
    
    
    # Saving improvements
    bestIndividuals.append(best_ind)
    generationsWithImprovements.append(current_generation)
            
       
    if __VERBOSE:
        print('\n* Experiment {0:03d}: first best individual={1} with fitness={2:.6f}'.format(current_experiment, best_ind, best_ind.fitness.values[0]))
    
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
                global __GENES_UP_LIMITS, __GENES_LOW_LIMITS
                
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
        for ind in pop: print ('Individual={0}\t Fitness={1:.03f}'.format(ind,ind.fitness.values[0]))
        print('Done!')
        
    
    number_of_generations = __NUMBER_OF_GENERATIONS
    generations = list(range(1,number_of_generations + 1))

    for current_generation in generations:

        print('\n* Experiment {0:3d},generation={1:3}: Current best_ind={2} best_acc= {3:.6f}'.format(current_experiment,current_generation,best_ind,best_ind.fitness.values[0]))        

        if __VERBOSE:
            print('\t* Running variation operators...',end='')

        offspring = algorithms.varAnd(pop, 
                                          toolbox, 
                                      __CROSSOVER_INDP, 
                                      __MUTATE_INDP)
        if __VERBOSE: print('\t Done!')

        print('\t* Evaluating offspring...',end='')
        fits_and_matrices = list(map(toolbox.evaluate,offspring)) # list of (fitness,conf_matrix) tuples

        if __VERBOSE: print('\t Done!')

            
        if __VERBOSE: print('\t* Updating fitness and confusion matrix of offspring...',end='')
        for i in list(range(len(offspring))):
            # fitness should be a one element tuple
            offspring[i].fitness.values = fits_and_matrices[i][0], # updating offspring fitness
            offspring[i].confusion_matrix = fits_and_matrices[i][1] # second tuple element
        
        best_offspring = None
        
        # tracking new best offspring
        for ind in offspring:
            if best_offspring == None:
                best_offspring = ind
            elif ind.fitness.values[0] > best_offspring.fitness.values[0]:
                #print('old best fit={0} (ind={1}) new best fit={2} (ind={3}) at exp={4} gen={5}'.format(best_offspring.fitness.values[0], best_offspring, ind.fitness.values[0], ind, current_experiment, current_generation))
                best_offspring = ind
        
        if best_offspring.fitness.values[0] > best_ind.fitness.values[0]:
            lastGenWithImprovements = current_generation
            best_ind = copy.deepcopy(best_offspring)
            
            # saving improvements to historic
            bestIndividuals.append(best_ind)
            generationsWithImprovements.append(current_generation)

            if __VERBOSE:
                print('\n\t* Improvement at {0}th experiment: {1} with fitness={2} at generation={3}'.format(current_experiment, best_ind, best_ind.fitness.values[0], current_generation))
        
        if __VERBOSE and lastGenWithImprovements == current_generation:
            print('\t*** WARNING: Consecutive generations without improvements for {1}th experiment= {0}'.format(current_generation - lastGenWithImprovements,current_experiment))
            print(' Done!')

                
        update_tourneament_size(current_generation,lastGenWithImprovements,toolbox)
        
        
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
        
        print('\n\t*Saving experiment data to output file: ')
    
    
    ofile = saveExperimentsDataToFile(current_experiment, best_ind, bestIndividuals, generationsWithImprovements, __DEAP_RUN_ID)
    if __VERBOSE:
        print(ofile)
        print('Best Individual Found is: ', best_ind)
        print('\t* Fitness: ', best_ind.fitness.values[0])
        print('\t* Confusion Matrix:\n', best_ind.confusion_matrix)
    
    print('* Experiment {0:03d} is finished'.format(current_experiment))
    
    return best_ind #, best_ind.fitness.values[0], best_ind.confusion_matrix


# REVISAR!!
def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]
    
    print ('Usage:\n    ', script_name, '[Options] -c <csv_file> -d <attributes_dir> ')
    print ('  Options:')
    #print('\t-c, --csv\tADNI csv file which contains images metadata')
    #print('\t-d, --dir\tdirectory which contains all attributes files extracted by get_attribs.py tool')
    print('\t-o, --output_dir\t\tdefines the directory where output files will be saved')
    print('\t-m, --model\t\t\tmodel used to fit MRI data. Supported models are: ', build_models_names_list())
    #print('\t\t\t\t\t\t',build_models_names_list())
    print('\t-p, --parallel\t\t\tenable parallel computation of experiments over all cores (default: parallel is off)')
    print('\t-v, --verbose\t\t\tenables verbose mode (default: disabled)')
    print('\t-n, --number_of_experiments\tnumber of experiments to run with deap (default: 1)')
    print('\t-h, --help\t\t\tdisplays this help screen')



# FUNCAO MAIN
def main(argv):
    csv_file = ''
    attribs_dir = ''
    out_dir = './'
    model = ''
    seeds_file = ''
    csv_file_ok = False
    attribs_dir_ok = False
    out_dir_ok = False
    model_ok = False
    seeds_file_ok = False
    verbose_ok = False
    multi_cpu_ok = False
    number_of_experiments = 1
    
    try:
        opts, args = getopt.getopt(argv[1:],"hc:a:o:m:s:vpn:",["csv=","attributes_dir=","output_dir=","model=","seeds_file=","verbose","parallel","number_of_experiments="]) 
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
        elif opt in ("-p", "--parallel"):
            multi_cpu_ok = True
        elif opt in ("-n", "--number_of_experiments"):
            try:    
                number_of_experiments = int(arg)
            except ValueError:
                print('Error: argument {0} must be a integer!'.format(arg))
                display_help()
                sys.exit(0)
            except Exception as err:
                print('Error: An exception has rised on try of integer conversion of the argument {0}.\n\tCause: '.format(arg,err.__cause__))
                display_help()
                sys.exit(0)
        elif opt in ("-m", "--model"):
            if model_name_is_valid(arg):
                model_ok = True
                global __MODEL_NAME
                __MODEL_NAME = arg
            else:
                print('Error: argument {0} must be a valid model name!'.format(arg))
                display_help()
                sys.exit(0)
    
    if csv_file_ok and attribs_dir_ok and model_ok:
            
        print('* Loading data...')
        print('\t* Attribs directory is: {0}'.format(attribs_dir))
        print('\t* Input CSV file is: {0}'.format(csv_file))
        print('\t* Model: {0}'.format(__MODEL_NAME))

        if out_dir_ok:
            global __OUTPUT_DIRECTORY
            __OUTPUT_DIRECTORY = out_dir
            print ('\t* Output dir is: {0}'.format(__OUTPUT_DIRECTORY))

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
        global __VALID_GENDERS, __MAX_AGE, __MIN_AGE
        
        all_attribs, all_body_planes, all_slice_num, all_slice_amounts, all_output_classes, all_genders, all_ages, demographics_dic = loadattribs.load_all_data_using_filters(attribs_dir, csv_file, valid_genders=__VALID_GENDERS, max_age=__MAX_AGE, min_age=__MIN_AGE, black_list_id=__BLACK_LIST_ID)
        
        #max_slice_values = loadattribs.getSliceLimits(all_slice_num)
        max_consecutive_slices = __DEFAULT_MAX_CONSEC_SLICES

        if __VERBOSE:
            end = time.time()
            print('* Time to load all attributes:',end - start,' seconds')


        pfilename = saveParametersFile(max_consecutive_slices,number_of_groupings)
        print('Saving parameters list to file {0}'.format(pfilename))
        
        all_experiments = list(range(1,number_of_experiments + 1))
        print('Running experiments...')
        
        # Updating global variables
        global __BODY_PLANES, __MAX_SLICES_VALUES, __MIN_SLICES_VALUES
        __BODY_PLANES = loadattribs.getBplanes(all_slice_amounts)
        __MIN_SLICES_VALUES,__MAX_SLICES_VALUES = loadattribs.getSliceLimits(all_slice_amounts)

        all_experiments_best_ind = []

        if __MULTI_CPU_USAGE and __name__ == "__main__":

            cores_num = multiprocessing.cpu_count()
            if __VERBOSE: print('* Running Experiments using Multicore option')
            
            with Pool(cores_num) as p:
                from functools import partial
                all_experiments_best_ind = p.map(
                    partial(run_deap,
                            all_attribs,
                            all_slice_amounts,
                            all_output_classes,
                            all_genders,
                            all_ages,
                            max_consecutive_slices, # length of slices range
                            number_of_groupings),
                    all_experiments)
        else:
            for experiment in all_experiments:
                
                exp_ind = run_deap(all_attribs,
                         all_slice_amounts,
                         all_output_classes,
                         all_genders,
                         all_ages,
                         max_consecutive_slices, # length of slices range
                         number_of_groupings,
                         experiment) # controls how many slices ranges there will be used
                all_experiments_best_ind.append(exp_ind)

        if __ALARM:
            os.system('play -nq -t alsa synth {} sine {}'.format(__DURATION, __FREQ))
        
        print('* Saving blotspot using final result... ', end='')
        bplot_file = save_final_result_boxplot(all_experiments_best_ind,[__MODEL_NAME])
        print('Done. (file={0}'.format(bplot_file))
        
        print('* Saving final results... ', end='')
        #final_results_file = saveResultsCSVFile(all_experiments_best_ind)
        final_results_file = saveDetailedResultsCSVFile(all_experiments_best_ind)
        print('Done. (file={0}'.format(final_results_file))
            
        
    else:
        display_help()
    

if __name__ == "__main__":    
    main(sys.argv)
    
