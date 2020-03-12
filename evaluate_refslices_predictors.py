#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:11:58 2020

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.metrics import mean_absolute_error

import numpy as np
#from deap import base, creator, tools, algorithms

import random, sys, os, csv
import loadattribs 
#import knn_alzheimer_crossvalidate
#import evaluating_classifiers as ec
import getopt
import time
import datetime
import multiprocessing
from multiprocessing import Pool
#import copy

# Ignore warnings from scikit learn
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

#def build_models_dictionary(knn_k_value=3,lr_solver='sag',lr_multiclass='ovr',rf_estimators=100):
def build_models_dictionary(knn_k_value=3,lr_solver='sag',lr_multiclass='ovr',rf_estimators=100):
    models_dic = {}
    
    models_names = build_models_names_list()
    models_constructors = []
    
    models_constructors.append(KNeighborsClassifier(n_neighbors=knn_k_value))
    models_constructors.append(LinearDiscriminantAnalysis())
    models_constructors.append(DecisionTreeClassifier())
    models_constructors.append(GaussianNB())
    models_constructors.append(SVC())
    models_constructors.append(RandomForestClassifier(n_estimators=rf_estimators))
    models_constructors.append(LogisticRegression(solver=lr_solver, multi_class=lr_multiclass))
    #models_constructors.append()
    
    for name, const in zip(models_names, models_constructors):
        models_dic[name]=const
        
    return models_dic

#Global Variables
__VERBOSE = False

# Global Slicing Arguments
#global __ALL_ATTRIBS, __ALL_OUTPUT_VALUES, __BODY_PLANES, __MAX_SLICES_VALUES, __DEFAULT_MAX_CONSEC_SLICES, __DEFAULT_NUMBER_OF_GROUPINGS

# Global Alzheimer Classification Problem Arguments
#global __GENES_LOW_LIMITS, __GENES_UP_LIMITS, __DEFAULT_KNN_K_VALUE, __VERBOSE

# Global Classifiers Parameters
global __DEFAULT_KNN_K_VALUE, __DEFAULT_RF_NUM_ESTIMATORS, __DEFAULT_LR_SOLVER, __DEFAULT_LR_MULTICLASS


# Global Runtime Parameters
global __MULTI_CPU_USAGE

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
#__ALL_ATTRIBS = []
#__ALL_OUTPUT_VALUES = []
#__BODY_PLANES = []
#__MAX_SLICES_VALUES = []
#__MIN_SLICES_VALUES = []
#__DEFAULT_MAX_CONSEC_SLICES = 20 #20
#__DEFAULT_NUMBER_OF_GROUPINGS = 1

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
__DEFAULT_RF_NUM_ESTIMATORS = 100
__DEFAULT_LR_SOLVER = 'sag'
__DEFAULT_LR_MULTICLASS = 'ovr'


__MODELS = build_models_dictionary(
        knn_k_value=-__DEFAULT_KNN_K_VALUE,
        lr_solver=__DEFAULT_LR_SOLVER,
        lr_multiclass=__DEFAULT_LR_MULTICLASS,
        rf_estimators = __DEFAULT_RF_NUM_ESTIMATORS
        )
__MODEL_CONSTRUCTOR = __MODELS[__MODEL_NAME]


__USE_RESCALING = True
__USE_SMOTE = False
__USE_PCA = True
__CV_TYPE = 'kcv'
__CV_MULTI_THREAD = True
__CV_SHUFFLE = True
__KCV_FOLDS = 10
__USE_STRATIFIED_KFOLD = False
__MAXIMIZATION_PROBLEM = False

# Runtime Parameters
__REFSP_RUN_ID = ''
__OUTPUT_DIRECTORY = './'
__MULTI_CPU_USAGE = False
__VERBOSE = False
__CORES_NUM = 1


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



def setRunID():
    global __REFSP_RUN_ID
    __REFSP_RUN_ID = str(datetime.date.today()) + '_' + str(int(round(time.time())))
    return __REFSP_RUN_ID

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
    global __REFSP_RUN_ID, __MULTI_CPU_USAGE, __OUTPUT_DIRECTORY, __VERBOSE, __CORES_NUM
    strPool.append('\n# Runtime parameters\n')
    strPool.append(' __REFSP_RUN_ID ={0}\n'.format(__REFSP_RUN_ID))
    strPool.append(' __MULTI_CPU_USAGE  ={0}\n'.format(__MULTI_CPU_USAGE ))
    strPool.append(' __OUTPUT_DIRECTORY ={0}\n'.format(__OUTPUT_DIRECTORY))
    strPool.append('__VERBOSE = {0}\n'.format(__VERBOSE ))
    strPool.append('__CORES_NUM = {0}\n'.format(__CORES_NUM ))
    
    # Classifier pipeline parameters
    global __USE_RESCALING, __USE_SMOTE, __CV_TYPE, __CV_MULTI_THREAD, __CV_SHUFFLE
    global __KCV_FOLDS, __MAXIMIZATION_PROBLEM, __USE_PCA, __USE_STRATIFIED_KFOLD
    strPool.append('\n# Regression pipeline parameters\n')
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
    strPool.append(' __DEFAULT_RF_NUM_ESTIMATORS ={0}\n'.format(__DEFAULT_RF_NUM_ESTIMATORS))
    
    # Slicing parameters
#    global __BODY_PLANES, __MAX_SLICES_VALUES, __MIN_SLICES_VALUES, __DEFAULT_MAX_CONSEC_SLICES
#    strPool.append('\n# Slicing parameters\n')
#    strPool.append(' max_consecutive_slices = {0}\n'.format(max_consecutive_slices))
#    strPool.append(' number_of_groupings = {0}\n'.format(number_of_groupings))
#    strPool.append(' __BODY_PLANES = {0}\n'.format(__BODY_PLANES))
#    strPool.append(' __MAX_SLICES_VALUES = {0}\n'.format(__MAX_SLICES_VALUES))
#    strPool.append(' __MIN_SLICES_VALUES = {0}\n'.format(__MIN_SLICES_VALUES))
    #strPool.append(' __DEFAULT_MAX_CONSEC_SLICES={0}\n'.format(__DEFAULT_MAX_CONSEC_SLICES))
    
    
    # Evolutionary arguments
#    global __TOURNEAMENT_SIZE, __MUTATE_INDP, __CROSSOVER_INDP, __POPULATION_SIZE, __NUMBER_OF_GENERATIONS 
#    global __MAX_GENERATIONS_WITHOUT_IMPROVEMENTS, __DEFAULT_TARGET_FITNESS, __DEFAULT_WORST_FITNESS
#    global  __GENES_LOW_LIMITS, __GENES_UP_LIMITS, __TOURNEAMENT_SIZE_IS_DYNAMIC
#    strPool.append('\n# Evolutonary parameters\n')
#    strPool.append(' __TOURNEAMENT_SIZE_IS_DYNAMIC = {0}\n'.format(__TOURNEAMENT_SIZE_IS_DYNAMIC))
#    strPool.append(' __TOURNEAMENT_SIZE = {0}\n'.format(__TOURNEAMENT_SIZE))
#    strPool.append(' __MUTATE_INDP = {0}\n'.format(__MUTATE_INDP))
#    strPool.append(' __CROSSOVER_INDP = {0}\n'.format(__CROSSOVER_INDP ))
#    strPool.append(' __POPULATION_SIZE = {0}\n'.format(__POPULATION_SIZE ))
#    strPool.append(' __NUMBER_OF_GENERATIONS = {0}\n'.format(__NUMBER_OF_GENERATIONS))
#    strPool.append(' __MAX_GENERATIONS_WITHOUT_IMPROVEMENTS = {0}\n'.format(__MAX_GENERATIONS_WITHOUT_IMPROVEMENTS ))
#    #strPool.append(' __DEFAULT_TARGET_FITNESS = {0}\n'.format(__DEFAULT_TARGET_FITNESS))
#    #strPool.append(' __DEFAULT_WORST_FITNESS = {0}\n'.format(__DEFAULT_WORST_FITNESS))
#    strPool.append(' __GENES_LOW_LIMITS = {0}\n'.format(__GENES_LOW_LIMITS))
#    strPool.append(' __GENES_UP_LIMITS = {0}\n'.format(__GENES_UP_LIMITS))


    return strPool

def saveParametersFile(max_consec_slices,num_groupings):
    global __OUTPUT_DIRECTORY, __REFSP_RUN_ID
    #output_dir_path = os.path.join(__OUTPUT_DIRECTORY, build_experiments_output_dir_name())
    output_dir_path = build_experiments_output_dir_name()
    
    
    filename = 'parameters_{0}.txt'.format(__REFSP_RUN_ID)
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
    
    global __MODEL_NAME, __REFSP_RUN_ID
    fig,ax = plt.subplots()
    plt.title(title)
    ax.boxplot(best_fits,labels=labels)
    
    best_of_bests_position = np.argmax(np.array(best_fits))
    best_of_bests = best_fits[best_of_bests_position]
    img_filename = 'final_result-acc_{0}-model_{1}-run_{2}.png'.format(best_of_bests, __MODEL_NAME,__REFSP_RUN_ID)
    
    output_dir_path = build_experiments_output_dir_name()
    
    
    img_full_filename = os.path.join(output_dir_path, img_filename)
    if __VERBOSE: 
        print('\n* Saving final result as bloxplot imagem in: {0}'.format(img_full_filename))
    
    fig.savefig(img_full_filename)
    
    if __VERBOSE: 
        print('\t Done')
    



def saveResultsCSVFile(all_experiments_best_ind):
    global __OUTPUT_DIRECTORY, __REFSP_RUN_ID
    output_dir_path = build_experiments_output_dir_name()
    
    
    filename = 'results_{0}_{1}.csv'.format(__REFSP_RUN_ID, __MODEL_NAME)
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
    global __OUTPUT_DIRECTORY, __REFSP_RUN_ID
    output_dir_path = build_experiments_output_dir_name()
    
    
    filename = 'detailed_results_{0}_{1}.csv'.format(__REFSP_RUN_ID, __MODEL_NAME)
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
    global __DEFAULT_MAX_CONSEC_SLICES, __DEFAULT_KNN_K_VALUE, __DEFAULT_RF_NUM_ESTIMATORS
    global __MUTATE_INDP, __CROSSOVER_INDP, __POPULATION_SIZE
    
    run_str = 'run_{0}_{1}'.format(__REFSP_RUN_ID, __MODEL_NAME)
    slash = '/'
    
    parent_dir = __OUTPUT_DIRECTORY # usually './' or '../'
    if parent_dir.endswith('/'):
        slash = ''
    
    full_dir_path = parent_dir + slash + run_str
    return full_dir_path

def build_experiment_output_filename(exp_num, best_accuracy):
    acc_value = '{0:.2f}'.format(best_accuracy)
    
    full_dir_path = build_experiments_output_dir_name()
    filename = 'acc_{0}-run_{1}-exp_{2:03d}.txt'.format(acc_value, __REFSP_RUN_ID, exp_num)
    output_full_filename = os.path.join(full_dir_path, filename)
    return output_full_filename, __REFSP_RUN_ID

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

def all_metrics_names(metric='acc'):
    metrics = []
    
    
    if metric == 'acc':
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
    elif metric == 'mae':
    #    [model_name,mean_mse,best_mse,std_mse,worst_mse,total_time,all_mse,median_mse]
        metrics.append('name')
        metrics.append('mean_mae')
        metrics.append('std_mae')
        metrics.append('best_mae')
        #metrics.append('best_cmat')
        metrics.append('worst_mae')
        metrics.append('total_time')
        metrics.append('all_mae_np')
        #metrics.append('all_cmat')
        metrics.append('median_mae')
        #metrics.append('median_cmat')
    else:
        metrics.append('name')
        metrics.append('mean_mse')
        metrics.append('std_mse')
        metrics.append('best_mse')
        #metrics.append('best_cmat')
        metrics.append('worst_mse')
        metrics.append('total_time')
        metrics.append('all_mse_np')
        #metrics.append('all_cmat')
        metrics.append('median_mse')
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



def read_refslices_data_from_csv(csv_file):
    X_data = []
    Y_data = []
    M_data = []
    ref_class = 1
    
    #demographics_dictionary = {}
    header = ''

    if os.path.exists(csv_file):

        try:
            with open(csv_file, 'r') as file:
                #print('CSV File received: ', csv_file)
                reader = csv.reader(file)
                header = next(reader) 
                for row in reader:
                    image_id = row[0]
                    gender = row[2]
                    age = row[3]
                    alz_class = row[4]
                    all_refslice_str_attribs = row[5:]
                    
                    try:
                        age = int(age)

                    except ValueError:
                        print('* Invalid AGE({0}) entry for image ID {1}. CSV file has problems'.format(age, image_id))

                    try:
                        ref_slice = int(row[1])                        
                    except ValueError:
                        print('* Invalid REF_SLICE({0}) entry for image ID {1}. CSV file has problems'.format(ref_slice, image_id))
                    
                    try:
                        alz_class = int(alz_class)

                    except ValueError:
                        print('* Invalid ALZHEIMER CLASS({0}) entry for image ID {1}. CSV file has problems'.format(alz_class, image_id))
                    
                    M_data.append((image_id,gender,age,alz_class,ref_class))
                    
                    position = 0
                    all_refslice_attribs = []
                    for str_attrib in all_refslice_str_attribs:
                        try:
                            attrib = float(str_attrib)
                        except ValueError:
                            print('* Invalid attrib value({0}) entry for image ID {1} at {2}th atribute position. CSV file has problems'.format(str_attrib, image_id, position))
                        all_refslice_attribs.append(attrib)
                        position = position + 1
                    
                    X_data.append(all_refslice_attribs)
                    Y_data.append(ref_slice)
                    
                    
        except os.error:
            print("*** ERROR: The csv file %s can not be readed (os.error in build_classes_dictionary)" % csv_file)    

    else:
        message = str("file %s does not exist!" % csv_file)
        raise ValueError(message)
    return X_data, Y_data, M_data, header


def buildDataFrames(X_data, y_data, M_data, header='', debug=False):
    # Cheking data
    print('received header=',header)

    X_data_reshaped = X_data

    # Data preparation
#    try:
#        new_dimensions = (X_data.shape[0],
#                          X_data.shape[1]*X_data.shape[2])
#    except IndexError:
#        print('** IndexValue exception')
#        print('\tX_data.shape=',X_data.shape)
#        print('\ty_data.shape=',y_data.shape)
#        print('\tnew_dimentions=',new_dimensions)
#        print('\t')
#        sys.exit(-1)
#    
#    if __VERBOSE and debug: 
#        print('* Reshaping for this data partition with these dimensions:',new_dimensions)
#
#    X_data_reshaped = np.reshape(X_data, new_dimensions)
#        
#    if __VERBOSE and debug: 
#        print('...done')
#        print('* The shape of a line data retrived from the new partition=', X_data_reshaped[0].shape)
    
    import pandas as pd
    X_pandas = pd.DataFrame(data=X_data_reshaped)
    y_pandas = pd.DataFrame(data=y_data)
    
    if header:
        M_pandas = pd.DataFrame(data=M_data, columns=[header[0],header[2],header[3],header[4],'nao_sei'])
    else:
        M_pandas = pd.DataFrame(data=M_data)
    
#    image_ids = []
#    all_genders = []
#    all_ages = []
#    
#    for m in M_data:
#        image_id,gender,age = m
#        image_ids.append(image_id)
#        all_genders.append(gender)
#        all_ages.append(age)
        
    
    # Reconding genders to numeric format
    #num_genders = num_genders_list(all_genders)
    #num_genders = all_genders
    
    # numeric Gender and Age columns inclusion
    #image_id_series = pd.Series(image_ids,name=header[0])
    #genders_series = pd.Series(num_genders,name=header[2])
    #age_series = pd.Series(all_ages,name=header[3])
    
    # Formating all data as float
    X_pandas = X_pandas.astype(dtype=np.float64)
    
    # Concatanation
    #X_pandas = pd.concat([X_pandas, genders_series, age_series],axis=1,ignore_index=True)
    
    #X_pandas = pd.concat([X_pandas, image_id_series, genders_series, age_series],axis=1,ignore_index=True)
    
    X_final_pandas = pd.concat([M_pandas,X_pandas],axis=1,sort=False)
    

    
    return X_final_pandas, y_pandas

#def evaluate_model(X_data, y_data, model_name,
#                   folds, cur_metric='mse', cv_seed=7, cv_shuffle=True,
#                   smote=True, rescaling=True, cores_num=1, 
#                   maximization=True, stratified_kfold=True,
#                   pca=False, debug=False):
def evaluate_model(X_data, y_data, model_name,
                   folds, cv_seed=7, cv_shuffle=True,
                   smote=True, rescaling=True, cores_num=1, 
                   maximization=True, stratified_kfold=True,
                   pca=False, debug=False):

    all_acc = []
    all_cmat = []
    
    start_time = time.time()
    
    #import pandas as pd
    non_float_columns = []
    
    print('X_data:\n',X_data)
    
    for col in X_data:
        if X_data[col].dtypes != np.float64:
            non_float_columns.append(col)
        else:
            print('found a np.float64 column at label ',col)
    
    print('non_float_columns: ',non_float_columns)
    
    
    numeric_X_data = X_data.iloc[:, 4:]
    print('numeric_X_data:\n',numeric_X_data)
    #numeric_X_data = X_data
    #for col in non_float_columns:
    #    numeric_X_data = X_data.drop(col)
    #numeric_X_data = X_data.drop(non_float_columns)
            
            
    
    #to_be_dropped=pd.DataFrame(X_data.categorical).columns
    #numeric_X_data = X_data.drop(to_be_dropped,axis=1)
    #numeric_X_data = X_data.__get_numeric_data()
    
    #all_mae = []
    #all_mse = []
    

    
    # STEP 1: perform rescaling if required
    if rescaling:
        
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        X_fixed = scaler.fit_transform(numeric_X_data) # Fit your data on the scaler object
    else:
        X_fixed = numeric_X_data
        
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
        y_pred = np.round(y_pred)
        
        # STEP 6: Building Evaluation Metrics
        acc = metrics.accuracy_score(y_test, y_pred)
        cmat = metrics.confusion_matrix(y_test,y_pred,labels=None,sample_weight=None)
        #mae = metrics.mean_absolute_error(y_test, y_pred)
        #mse = metrics.mean_squared_error(y_test, y_pred)

#        print('acc={0:.4}'.format(acc))
        #all_mae.append(mae)
        #all_mse.append(mse)
        all_acc.append(acc)
        all_cmat.append(cmat)
        print('conf_matrix:\n',conf_matrix)
        print('np.array(cmat)=\n',np.array(cmat))
        conf_matrix = conf_matrix + np.array(cmat)
        #all_cmat.append(cmat)
        
        
        
        #cv_results = model_selection.cross_val_score(model, X_data, y_data, cv=kfold, scoring=metric, n_jobs=cores_num)
        
    	
    # Converting to Numpy Array to use its statiscs pre-built functions
    #np_all_mae = np.round(np.array(all_mae))
    #np_all_mse = np.round(np.array(all_mse))
    np_all_acc = np.array(all_acc)

    # OPTIONAL STEP : Showing side by side y_pred x y_true x error
    import pandas as pd
    if cur_metric == 'mse':
        all_errors = pd.DataFrame(data=np_all_mse)
    else:
        all_errors = pd.DataFrame(data=np_all_mae)
    print('y_pred=',y_pred)
    print('y_test=',y_test)
#    
#    y_pred_test = pd.concat([pd.DataFrame(data=y_pred,columns='y_pred'),pd.DataFrame(data=y_test,columns='y_true')],axis=1,sort=False)
#    y_pred_test_error = pd.concat([y_pred_test,pd.DataFrame(data=all_errors,columns='error')],axis=1,sort=False)
#    print('y_pred_test_error=\n',y_pred_test_error)
    
#    print('length of all_acc={0} and np_all_acc={1}'.format(len(all_acc),len(np_all_acc)))
    
    # Finding position of the best and the worst individual
    best_acc_pos  = np.argmax(np_all_acc) if maximization else np.argmin(np_all_acc)
    worst_acc_pos = np.argmin(np_all_acc) if maximization else np.argmax(np_all_acc)
#    best_mae_pos  = np.argmax(np_all_mae) if maximization else np.argmin(np_all_mae)
#    worst_mae_pos = np.argmin(np_all_mae) if maximization else np.argmax(np_all_mae)
#    best_mse_pos  = np.argmax(np_all_mse) if maximization else np.argmin(np_all_mse)
#    worst_mse_pos = np.argmin(np_all_mse) if maximization else np.argmax(np_all_mse)
    median_pos = folds//2
    
#    best_mae = np_all_mae[best_mae_pos]
#    best_mse = np_all_mse[best_mse_pos]
    best_acc = np_all_acc[best_acc_pos]
    #best_cmat = all_cmat[best_acc_pos]
#    worst_mae = all_mae[worst_mae_pos]
#    worst_mse = all_mse[worst_mse_pos]
    worst_acc = all_acc[worst_acc_pos]
    #worst_cmat = all_cmat[worst_acc_pos]
    
    mean_acc = np_all_acc.mean()
    #mean_mae = np_all_mae.mean()
    #mean_mse = np_all_mse.mean()
    
    #median_mae = np_all_mae[median_pos] #np_all_acc[folds//2] if folds % 2 == 1 else (np_all_acc[(folds+1)//2] + np_all_acc[(folds-1)//2])//2
    #median_mse = np_all_mse[median_pos]
    #median_cmat = all_cmat[median_pos]
    median_acc = all_acc[median_pos]
    
    std_acc = np_all_acc.std()
#    std_mae = np_all_mae.std()
#    std_mse = np_all_mse.std()
    
    
    # Calculing execution time
    end_time = time.time()
    total_time = end_time - start_time
    
    #dic = {'name':name, 'mean_acc':mean_acc, 'std_acc':std_acc, 'best_acc':best_acc, 
    #'best_cmat':best_cmat, 'worst_acc':worst_acc, 'worst_cmat':worst_cmat, 'total_time':total_time, 'all_acc':np_all_acc, 'all_cmat':all_cmat, 'median_acc':median_acc, 'median_cmat':median_cmat}
    
    #metrics_list = [model_name,mean_acc,best_acc,std_acc,best_cmat,worst_acc,worst_cmat,total_time,all_acc,all_cmat,median_acc,median_cmat]
    metrics_list = [model_name,mean_acc,best_acc,std_acc,worst_acc,conf_matrix,total_time,all_acc,median_acc]
    #metrics_list = [model_name,mean_mse,best_mse,std_mse,worst_mse,total_time,all_mse,median_mse]

    #metrics_names = all_metrics_names(cur_metric)
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


def main(argv):
    csv_filename = '../../ref-slices_attributes-axis0.csv'
    
    global __USE_RESCALING, __USE_SMOTE, __CV_TYPE, __CV_MULTI_THREAD, __KCV_FOLDS, __CV_SHUFFLE
    global __MAXIMIZATION_PROBLEM, __CORES_NUM, __USE_STRATIFIED_KFOLD    
    
    X_data, Y_data, M_data, head = read_refslices_data_from_csv(csv_filename)
    
    #print('M_data:\n',M_data)
    
    X_pandas, y_pandas = buildDataFrames(X_data, Y_data, M_data, head, debug=True)
    
    print('Retrived X_pandas from ',csv_filename,'\n',X_pandas)
    
    model_name = 'RF'
    
    dicionary_results = evaluate_model(X_pandas, y_pandas, model_name, __KCV_FOLDS, cv_seed=7, cv_shuffle=__CV_SHUFFLE,
                                       smote=__USE_SMOTE, rescaling=__USE_RESCALING, cores_num=1, 
                                       maximization=__MAXIMIZATION_PROBLEM, stratified_kfold=__USE_STRATIFIED_KFOLD,
                                       debug=__VERBOSE)
    
    
    
    #rint('X_pandas:\n',X_pandas)
    print('dicionary_results:',dicionary_results)
    #buildDataFrames()
    return 0


#
#
#
#

def refsp():
    return 0


# FUNCAO MAIN do Ref Slice Predictor
def main_refsp(argv):
    csv_file = ''
    attribs_dir = ''
    out_dir = './'
    model = 'KNN'
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
                    partial(run_refsp,
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
                
                exp_ind = run_refsp(all_attribs,
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

# FUNCAO MAIN do DeapAlzheimer
#def main(argv):
#    csv_file = ''
#    attribs_dir = ''
#    out_dir = './'
#    model = ''
#    seeds_file = ''
#    csv_file_ok = False
#    attribs_dir_ok = False
#    out_dir_ok = False
#    model_ok = False
#    seeds_file_ok = False
#    verbose_ok = False
#    multi_cpu_ok = False
#    number_of_experiments = 1
#    
#    try:
#        opts, args = getopt.getopt(argv[1:],"hc:a:o:m:s:vpn:",["csv=","attributes_dir=","output_dir=","model=","seeds_file=","verbose","parallel","number_of_experiments="]) 
#    except getopt.GetoptError:
#        display_help()
#        sys.exit(1)
#    for opt, arg in opts:
#        if opt in ('-h','--help'):
#            display_help()
#            sys.exit(0);
#        elif opt in ("-c", "--csv"):
#            csv_file = arg
#            csv_file_ok = True
#        elif opt in ("-a", "--attributes_dir"):
#            attribs_dir = arg
#            attribs_dir_ok = True
#        elif opt in ("-o", "--output_dir"):
#            out_dir = arg
#            out_dir_ok = True
#        elif opt in ("-s", "--seeds_file"):
#            seeds_file = arg
#            seeds_file_ok = True            
#        elif opt in ("-v", "--verbose"):
#            verbose_ok = True
#        elif opt in ("-p", "--parallel"):
#            multi_cpu_ok = True
#        elif opt in ("-n", "--number_of_experiments"):
#            try:    
#                number_of_experiments = int(arg)
#            except ValueError:
#                print('Error: argument {0} must be a integer!'.format(arg))
#                display_help()
#                sys.exit(0)
#            except Exception as err:
#                print('Error: An exception has rised on try of integer conversion of the argument {0}.\n\tCause: '.format(arg,err.__cause__))
#                display_help()
#                sys.exit(0)
#        elif opt in ("-m", "--model"):
#            if model_name_is_valid(arg):
#                model_ok = True
#                global __MODEL_NAME
#                __MODEL_NAME = arg
#            else:
#                print('Error: argument {0} must be a valid model name!'.format(arg))
#                display_help()
#                sys.exit(0)
#    
#    if csv_file_ok and attribs_dir_ok and model_ok:
#            
#        print('* Loading data...')
#        print('\t* Attribs directory is: {0}'.format(attribs_dir))
#        print('\t* Input CSV file is: {0}'.format(csv_file))
#        print('\t* Model: {0}'.format(__MODEL_NAME))
#
#        if out_dir_ok:
#            global __OUTPUT_DIRECTORY
#            __OUTPUT_DIRECTORY = out_dir
#            print ('\t* Output dir is: {0}'.format(__OUTPUT_DIRECTORY))
#
#        if seeds_file_ok:
#            global SEEDS_FILE
#            __SEEDS_FILE = seeds_file
#            print ('\t* Seeds file: {0}'.format(__SEEDS_FILE))
#       
#        if verbose_ok:
#            start = time.time()
#            global __VERBOSE
#            __VERBOSE = True
#            
#        if multi_cpu_ok:
#            global __MULTI_CPU_USAGE
#            __MULTI_CPU_USAGE = True
#            
#        setRunID()
#        global __ALARM, __FREQ, __DURATION
#        number_of_groupings = __DEFAULT_NUMBER_OF_GROUPINGS
#        
#        # Loading all data just once
#        global __VALID_GENDERS, __MAX_AGE, __MIN_AGE
#        
#        all_attribs, all_body_planes, all_slice_num, all_slice_amounts, all_output_classes, all_genders, all_ages, demographics_dic = loadattribs.load_all_data_using_filters(attribs_dir, csv_file, valid_genders=__VALID_GENDERS, max_age=__MAX_AGE, min_age=__MIN_AGE, black_list_id=__BLACK_LIST_ID)
#        
#        #max_slice_values = loadattribs.getSliceLimits(all_slice_num)
#        max_consecutive_slices = __DEFAULT_MAX_CONSEC_SLICES
#
#        if __VERBOSE:
#            end = time.time()
#            print('* Time to load all attributes:',end - start,' seconds')
#
#
#        pfilename = saveParametersFile(max_consecutive_slices,number_of_groupings)
#        print('Saving parameters list to file {0}'.format(pfilename))
#        
#        all_experiments = list(range(1,number_of_experiments + 1))
#        print('Running experiments...')
#        
#        # Updating global variables
#        global __BODY_PLANES, __MAX_SLICES_VALUES, __MIN_SLICES_VALUES
#        __BODY_PLANES = loadattribs.getBplanes(all_slice_amounts)
#        __MIN_SLICES_VALUES,__MAX_SLICES_VALUES = loadattribs.getSliceLimits(all_slice_amounts)
#
#        all_experiments_best_ind = []
#
#        if __MULTI_CPU_USAGE and __name__ == "__main__":
#
#            cores_num = multiprocessing.cpu_count()
#            if __VERBOSE: print('* Running Experiments using Multicore option')
#            
#            with Pool(cores_num) as p:
#                from functools import partial
#                all_experiments_best_ind = p.map(
#                    partial(run_deap,
#                            all_attribs,
#                            all_slice_amounts,
#                            all_output_classes,
#                            all_genders,
#                            all_ages,
#                            max_consecutive_slices, # length of slices range
#                            number_of_groupings),
#                    all_experiments)
#        else:
#            for experiment in all_experiments:
#                
#                exp_ind = run_deap(all_attribs,
#                         all_slice_amounts,
#                         all_output_classes,
#                         all_genders,
#                         all_ages,
#                         max_consecutive_slices, # length of slices range
#                         number_of_groupings,
#                         experiment) # controls how many slices ranges there will be used
#                all_experiments_best_ind.append(exp_ind)
#
#        if __ALARM:
#            os.system('play -nq -t alsa synth {} sine {}'.format(__DURATION, __FREQ))
#        
#        print('* Saving blotspot using final result... ', end='')
#        bplot_file = save_final_result_boxplot(all_experiments_best_ind,[__MODEL_NAME])
#        print('Done. (file={0}'.format(bplot_file))
#        
#        print('* Saving final results... ', end='')
#        #final_results_file = saveResultsCSVFile(all_experiments_best_ind)
#        final_results_file = saveDetailedResultsCSVFile(all_experiments_best_ind)
#        print('Done. (file={0}'.format(final_results_file))
#            
#        
#    else:
#        display_help()
    

if __name__ == "__main__":    
    main(sys.argv)