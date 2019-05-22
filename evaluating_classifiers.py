#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:02:30 2019

@author: rodrigo
"""

    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 08:33:13 2019

@author: rodrigo
"""

import loadattribs

from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm.classes import OneClassSVM
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
#from sklearn.mixture import DPGMM
#from sklearn.mixture import GMM 
#from sklearn.mixture import GaussianMixture
#from sklearn.mixture import VBGMM
import matplotlib.pyplot as plt


# Validation
from sklearn import model_selection
from sklearn import preprocessing
#from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split
from sklearn import metrics

# Oversampling
from imblearn.over_sampling import SMOTE

# Data handling
import pandas as pd
import numpy as np

# System 
import sys
import time
import random
import multiprocessing
from multiprocessing import Pool

def get_boxplot(results,labels): # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(labels)
    plt.show()

def build_models_list(knn_k_value=3,lr_solver='sag',lr_multiclass='ovr'):
    models = []
    models.append(('KNN', KNeighborsClassifier(n_neighbors=knn_k_value)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RF',RandomForestClassifier()))
    models.append(('LR', LogisticRegression(solver=lr_solver, multi_class=lr_multiclass)))
    return models

def evaluate_model_using_smote_and_rescaling(all_train_and_test_indexes, 
                                             X_data, 
                                             y_data,
                                             model_tuple,
                                             folds,
                                             cv_seed=7,
                                             cv_shuffle=True,
                                             smote=True, 
                                             rescaling=True, 
                                             cores_num=1, 
                                             maximization=True, 
                                             debug=False):
    
    all_acc = []
    all_cmat = []
    
    start_time = time.time()
    
    # STEP 1: perform rescaling if required
    if rescaling:
        
        scaler = preprocessing.StandardScaler()
        X_fixed = scaler.fit_transform(X_data) # Fit your data on the scaler object
    else:
        X_fixed = X_data
        
    # Added to solve column-vector issue
    y_fixed = np.ravel(y_data)
     
    # Validation setup

    
    cv = model_selection.KFold(n_splits=folds, random_state=cv_seed, shuffle=cv_shuffle)
    both_indexes = cv.split(X_data)
    #both_indexes = all_train_and_test_indexes
    
    #for train_indexes, test_indexes in all_train_and_test_indexes:
    for train_indexes, test_indexes in both_indexes:
        
        # STEP 2: split data between test and train sets
        X_train = X_fixed[train_indexes]
        X_test = X_fixed[test_indexes]
        y_train = y_fixed[train_indexes]
        y_test = y_fixed[test_indexes]
        
        # STEP 3: oversampling training data using SMOTE if required
        if smote:
            smt = SMOTE() 
            X_train, y_train = smt.fit_sample(X_train, y_train)
    
        name = model_tuple[0] # model name (string)
        model = model_tuple[1] # model class (implements fit method)
        
        # STEP 4: Training (Fit) Model
        model.fit(X_train, y_train)
        
        # STEP 5: Testing Model (Making Predictions)
        y_pred = model.predict(X_test) # testing
        
        # STEP 6: Building Evaluation Metrics
        acc = metrics.accuracy_score(y_test, y_pred)
        cmat = metrics.confusion_matrix(y_test,y_pred,labels=None,sample_weight=None)
#        print('acc={0:.4}'.format(acc))
        all_acc.append(acc)
        all_cmat.append(cmat)
        #cv_results = model_selection.cross_val_score(model, X_data, y_data, cv=kfold, scoring=metric, n_jobs=cores_num)
    	
    # Converting to Numpy Array to use its statiscs pre-built functions
    np_all_acc = np.array(all_acc)
#    print('length of all_acc={0} and np_all_acc={1}'.format(len(all_acc),len(np_all_acc)))
    
    # Finding position of the best and the worst individual
    best_acc_pos  = np_all_acc.argmax() if maximization else np_all_acc.argmin()
    worst_acc_pos = np_all_acc.argmin() if maximization else np_all_acc.argmax()
    best_acc = np_all_acc[best_acc_pos]
    best_cmat = all_cmat[best_acc_pos]
    worst_acc = all_acc[worst_acc_pos]
    worst_cmat = all_cmat[worst_acc_pos]
    mean_acc = np_all_acc.mean()
    median_acc = np_all_acc[folds//2] if folds % 2 == 1 else (np_all_acc[(folds+1)//2] + np_all_acc[(folds-1)//2])//2
    median_cmat = all_cmat[folds//2] if folds % 2 == 1 else None        
    median_acc = np_all_acc[folds//2]
    std_acc = np_all_acc.std()
    
    # Calculing execution time
    end_time = time.time()
    total_time = end_time - start_time
    
    #dic = {'name':name, 'mean_acc':mean_acc, 'std_acc':std_acc, 'best_acc':best_acc, 
    #'best_cmat':best_cmat, 'worst_acc':worst_acc, 'worst_cmat':worst_cmat, 'total_time':total_time, 'all_acc':np_all_acc, 'all_cmat':all_cmat, 'median_acc':median_acc, 'median_cmat':median_cmat}
    
    dic = {}
    dic['name'] = name
    dic['mean_acc'] = mean_acc
    dic['std_acc'] = std_acc
    dic['best_acc'] = best_acc
    dic['best_cmat'] = best_cmat
    dic['worst_acc'] = worst_acc
    dic['worst_cmat'] = worst_cmat
    dic['total_time'] = total_time
    dic['all_acc'] = all_acc
    dic['all_cmat'] = all_cmat
    dic['median_acc'] = median_acc
    dic['median_cmat'] = median_cmat
    
    return dic
	

def evaluate_all(X_pandas, y_pandas, knn_k_value,lr_solver,lr_multiclass,
                 use_multiprocess=True, use_smote=True, use_rescaling=True, 
                 cv_shuffle = True, kcv_folds=11, cv_seed=7, 
                 ):
    
    
    models = build_models_list(knn_k_value,lr_solver,lr_multiclass)

#    if len(models_names) != len(models):
#        raise ValueError('models_name length ({0}) != models length ({1})'.format(len(models_names),len(models)))
#        exit(1)
#    
    cores_num = multiprocessing.cpu_count() # used by n_jobs

    # Validation setup
    cv = model_selection.KFold(n_splits=kcv_folds, random_state=cv_seed, shuffle=cv_shuffle)
    both_indexes = cv.split(X_pandas)

    # results variables
    all_results_tuples = []
    #all_models_names = []
    #all_scores = []
    #all_std_scores = []
    #all_mean_scores = []
    #execution_times = []
 

    cores_num = multiprocessing.cpu_count() # used by n_jobs

    
    
    if use_multiprocess:
        print('X_pandas class is: ', [base.__name__ for base in X_pandas.__class__.__bases__])
        with Pool(cores_num) as p:
            from functools import partial
            all_results_tuples = p.map(
                    partial(evaluate_model_using_smote_and_rescaling, 
                            all_train_and_test_indexes=both_indexes,
                            X_data = X_pandas,
                            y_data = y_pandas,
                            folds = kcv_folds,
                            smote=use_smote,
                            rescaling=use_rescaling,
                            cores_num=cores_num,
                            maximization=True),
                    models)
#        pool = Pool(processes=cores_num)
#        for model in models:
#            print('* Evaluation {0} model...'.format(model[0]))
#            #arguments = (both_indexes, X_pandas, y_pandas, model, cv, smote=use_smote, rescaling=use_rescaling, cores_num=cores_num)
#            arguments = (both_indexes, X_pandas, y_pandas, model, cv, use_smote, use_rescaling, cores_num)
#            result = pool.apply(evaluate_model_using_smote_and_rescaling, arguments)
#            all_results_tuples.append(result)
        
        
        
    else:
        for model in models:
            print('* Evaluation {0} model...'.format(model[0]))
            dic_result = evaluate_model_using_smote_and_rescaling(both_indexes, X_pandas, y_pandas,
                                                              model, kcv_folds, smote=use_smote,
                                                              rescaling=use_rescaling, cores_num=cores_num)
            # List of tuples with shape:
            # (name, mean_acc, std_acc, best_acc, best_cmat, worst_acc, worst_cmat, total_time)
            all_results_tuples.append(dic_result)
            
            
    return all_results_tuples
        
    



#########################################






def main(argv):
    
    # runtime parameters
    __TOTAL_RUNNINGS = 1
    __MULTIPROCESS = False
    __USE_SAMPLE_DATA_DIR = True # Use this arguments to set the input directory of attributes files
    USE_FIXED_SLICE_GROUPING = True # Seeds test 
    
    # Models Parameters
    knn_k_value=3
    lr_solver='sag'
    lr_multiclass='ovr'
    kcv_folds = 11

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)    
    warnings.filterwarnings("ignore", category=FutureWarning)    
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    
    FIXED_SLICE_GROUPING = [0, 80, 1]
    
    
    # Use this arguments to set the input directory of attributes files
    __SAMPLE_DATA_DIR = "../../attributes_amostra"
    __FULL_DATA_DIR = "../../attributes2"
    attributes_dir = __FULL_DATA_DIR
    csv_file = './ADNI1_Complete_All_Yr_3T.csv'

    if __USE_SAMPLE_DATA_DIR:
        attributes_dir = __SAMPLE_DATA_DIR
    
    # Getting all data
    
    #start_time = time.time()
    #print('Loading all atributes data... ', end='')
    attribs, body_planes, slice_num, slice_amounts, output_classes = loadattribs.load_all_data(attributes_dir, csv_file)
    #end_time = time.time()
    #total_time = end_time - start_time
    #print('done (total time to load: {0})'.format(total_time))
    
    import deap_alzheimer
    min_slices_values = loadattribs.getSliceLimits(slice_amounts)[0]
    valid_bplanes = loadattribs.getBplanes(slice_amounts)

    #print('Slice Limits:',min_slices_values)
    #print('valid_bplanes=',valid_bplanes)
       
    if USE_FIXED_SLICE_GROUPING:
        #print('* Using a specific known good slice grouping... ', end='')
        
        bplane, start_slice, total_slices = FIXED_SLICE_GROUPING
    else:
        #print('* Building a random valid slice grouping... ', end='')
        
        bplane, start_slice, total_slices = deap_alzheimer.buildRandomSliceGrouping(
                planes = valid_bplanes,
                length = int(random.random()*20),
                max_indexes = min_slices_values,
                dbug=False)
    
    #print('Done!\n* Slice grouping created: [{0}, {1}, {2}]'.format(bplane,start_slice,total_slices))
    
   
    #start_time = time.time()
     
    # Getting some data partition 
    #print('* Getting the specific data partition using this slice grouping {0}... '.format([bplane,start_slice,total_slices]),end='')
    data_partition = loadattribs.getAttribsPartitionFromSingleSlicesGrouping(
            attribs,
            slice_amounts,
            bplane,
            start_slice,
            total_slices)
    #end_time = time.time()
    #total_time = end_time - start_time
    #print('done.\n* Total time to get the data partition= {0}'.format(total_time,))
    
    
    # Preparing data to use with PANDAS
    # Data preparation
    #print('* Current Data Partition\'s shape= ',data_partition.shape)
    try:
        new_dimensions = (data_partition.shape[0], data_partition.shape[1]*data_partition.shape[2])
    except IndexError:
        print('** IndexValue exception: data_partition.shape={0} output_classes.shape={1}'.format(data_partition.shape,output_classes.shape))
        sys.exit(-1)
    
    # Reshapping X_data
    X_reshaped = np.reshape(data_partition, new_dimensions)
    #print('* New Data Partition\'s shape= ',X_reshaped.shape)
    #print('* New dimensions (must be equal to X_reshaped.shape): ',new_dimensions)
    
    X_pandas = pd.DataFrame(data=X_reshaped)
    y_pandas = pd.DataFrame(data=output_classes)
    
    all_mean_acc = []
    #all_std_acc = []
    #all_max_acc = []
    #all_min_acc = []
    all_median_acc = []
    all_median_cmat = []
    all_time = []
    
    for n in list(range(__TOTAL_RUNNINGS)):
        ## all_experiments_results is a LIST of LISTS of Dictionaries which have as keys:
        #all_experiments_results.append(evaluate_all(X_pandas, y_pandas, use_multiprocess=__MULTIPROCESS))
        
        n_experiment_results = evaluate_all(X_pandas, y_pandas,
                                            knn_k_value,lr_solver,lr_multiclass,
                                            kcv_folds=kcv_folds,
                                            use_multiprocess=__MULTIPROCESS)
        
        model_mean_acc = []
        #model_std_acc = []
        #model_max_acc = []
        #model_min_acc = []
        model_median_acc = []
        model_median_cmat = []
        model_time = []
        
        for model_results in n_experiment_results:
            # 'model_results' is a dicionary
            
            model_mean_acc.append(model_results['mean_acc'])
#            model_std_acc.append(model_results['std_acc'])
#            model_max_acc.append(model_results['best_acc'])
#            model_min_acc.append(model_results['worst_acc'])
            model_median_acc.append(model_results['median_acc'])
            model_median_cmat.append(model_results['median_cmat'])
            model_time.append(model_results['total_time'])
            
            #dic = {'name':name, 'mean_acc':mean_acc, 'std_acc':std_acc, 'best_acc':best_acc, 
            #'best_cmat':best_cmat, 'worst_acc':worst_acc, 'worst_cmat':worst_cmat, 'total_time':total_time,
            #'all_acc':np_all_acc, 'all_cmat':all_cmat, 'median_acc':median_acc, 'median_cmat':median_cmat}
            
        np_mean_acc = np.array(model_mean_acc)
        #max_pos = np_mean_acc.argmax()
        #min_pos = np_mean_acc.argmin()
        np_time = np.array(model_time)
        
        all_mean_acc.append(np_mean_acc.mean())
        #all_std_acc.append(np_mean_acc.std())
        #all_max_acc.append(np_mean_acc[max_pos])
        #all_min_acc.append(np_mean_acc[min_pos])
        all_median_acc.append(model_median_acc[len(model_median_acc)//2])
        all_median_cmat.append(model_median_cmat[len(model_median_cmat)//2])
        all_time.append(np_time.mean)
        #all_mean_acc.append(.mean())
        #all_std_acc.append(np.array(model_mean_acc).std())
        #all_max_acc.append(model_max_acc[np.array(model_max_acc).argmax()])
        #all_min_acc.append(np.array(model_min_acc).mean())
        #all_median_acc.append(model_median_acc)
        #all_median_cmat.append(model_median_cmat)
        #all_time.append(model_time)
        
#    for results_for_only_one_exp in all_experiments_results:
#            this_model_mean_acc_list = []
#            this_model_std_acc_list = []
#            this_model_max_acc_list = []
#            this_model_min_acc_list = []
#            this_model_cmat_list = []
#            this_model_time_list = []
#            this_model_all_acc_list = []
#            this_model_all_cmat_list = []
#            this_model_median_acc_list = []
#            this_model_median_cmat_list = []
#        
#        for dic in results_for_only_one_exp:
#            # dic is the results from only one classifier for one experiment
#            print('{0}:\tacc={1:.4f} std={2:.4f} total_time={3:.4f}s)'.format(dic['name'], 
#                 dic['mean_acc'], dic['std_acc'], dic['total_time']))
#            this_model_mean_acc_list.append(dic['mean_acc'])
#            this_model_std_acc_list.append(dic['std_acc'])
#            this_model_max_acc_list.append(dic['best_acc']) 
#            this_model_min_acc_list.append(dic['worst_acc'])
#            this_model_cmat_list.append(dic[''])
#            this_model_time_list.append(dic['total_time'])
#            this_model_all_acc_list.append(dic['all_acc'])
#            this_model_all_cmat_list.append(dic['all_cmat'])
#            this_model_median_acc_list.append(dic['median_acc'])
#            this_model_median_cmat_list.append(dic['median_cmat'])
    
    
    
    models = build_models_list()        
        
    print('* All {0} experiments results:'.format(__TOTAL_RUNNINGS))
    for exp in range(__TOTAL_RUNNINGS):
        name = models[exp][0]
        print('name=',name)
        print('all_mean_acc[0].__class__.__name__=',all_mean_acc[0].__class__.__name__)
        np_all_mean = np.array(all_mean_acc[exp])
        print(np_all_mean)
        mean_acc = np_all_mean.mean()
        std_acc = np_all_mean.std()
        max_acc = np_all_mean[np_all_mean.argmax()]
        min_acc = np_all_mean[np_all_mean.argmin()]
        
        median_pos = kcv_folds // 2 if kcv_folds % 2 == 1 else None
        median_acc = all_median_acc[exp][median_pos] if kcv_folds % 2 == 1 else None
        median_cmat = all_median_cmat[exp][median_pos] if kcv_folds % 2 == 1 else None
        
        np_all_time = np.array(all_time[exp])
        mean_time = np_all_time.mean()
        
        print('{0}:\tmean_acc={1:.4f} mean_std={2:.4f} mean_time={3:.4f}s'.format(name, mean_acc, std_acc, mean_time))
        
#        print('{0}:\tacc={1:.4f} std={2:.4f} total_time={3:.4f}s)'.format(dic['name'], 
#                 dic['mean_acc'], dic['std_acc'], dic['total_time']))
    
    
    # (name, mean_acc, std_acc, best_acc, best_cmat, worst_acc, worst_cmat, total_time)
    
    #all_acc = np.array(all_acc)
    
    #best_model_acc = all_acc.max()
    #best_acc_pos = all_acc.argmax()
    #best_acc_cmat = all_cmat[best_acc_pos]
    #worst_acc_cmat = all_cmat[all_acc.argmin()]
    
    #print('\n* Results afer {0} runnings:\n{0}'.format(__TOTAL_RUNNINGS))
    #print('\ttime to run classifier={0}'.format(total_time))
    #print('\tclassifier={0}'.format(__CLASSIFIER))
    #print('\tmean={0}'.format(np.mean(all_acc)))
    #print('\tvariance={0}'.format(all_acc.var()))
    #print('\tstd={0}'.format(all_acc.std()))
    #print('\tmax={0}'.format(best_acc))
    #print('\tmin={0}'.format(all_acc.min()))
    #print('\tConfusion matrix of the best result:\n', best_acc_cmat)
    #print('\tConfusion matrix of the worst result:\n', worst_acc_cmat)
    #print('all_classifiers_results=\n',all_classifiers_results)
    return 0
    
    

if __name__ == "__main__":    
    main(sys.argv)



#from imblearn.over_sampling import SMOTE, ADASYN
#from collections import Counter



#print('unique(y)=',np.unique(y))




#X, y = SMOTE('auto').fit_resample(new_partition, output_classes)

#print(sorted(Counter(y_resampled).items()))
#clf_smote = LinearSVC().fit(X_resampled, y_resampled)
#
#X, y = ADASYN().fit_resample(X_pandas, y_pandas)
#print(sorted(Counter(y_resampled).items()))
#clf_adasyn = LinearSVC().fit(X_resampled, y_resampled)

#X = new_partition
#y = output_classes


# SMOTE Class Balancing
#print('* Input data shape=',X.shape)
#print('* Output values shape=',y.shape)


# STEP 1: split X and y into training and testing sets
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=12)
