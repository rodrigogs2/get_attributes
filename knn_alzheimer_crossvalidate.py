#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 08:33:13 2019

@author: rodrigo
"""

import loadattribs

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys
import time
import random
import multiprocessing
from multiprocessing import Pool 

'''
#import matplotlib.pyplot as plt
import numpy as np
#from sklearn import datasets
#import tensorflow as tf
#import os, cvs, sys
#import list_dir
import loadattribs


# Getting all files names
attribs_files = list_dir.list_files(attributes_dir,".txt")

# Checking how many files were found
print("\n* %s attribs files were loaded." % len(attribs_files))

# Variable used to check memory usage to load all attributes from all files
total_memory_usage = 0

# Use these to control how many files will be loaded
first_file = 0
total_printed_files = len(attribs_files)

# Loop which loads attributes, checks their shape and gets memory usage
print("* Loaded Files:")
for n in range(first_file, total_printed_files):
    file = attribs_files[n]
    memory_size = 0
    attribs,body_plane,slice_num,slice_amount = loadattribs.load_attribs_and_metadata(file)
    memory_size = attribs.size * attribs.itemsize + body_plane.size * body_plane.itemsize + slice_num.size * slice_num.itemsize
    total_memory_usage += memory_size
    print('\t-Attributes from the {0}th data file :\n{1}'.format(n,attribs))
    print('\t-Body planes from the {0}th data file :\n{1}'.format(n,body_plane))
    print('\t-Slice numbers from the {0}th data file :\n{1}'.format(n,slice_num))
    print('\t-Dimensions of each vector: attribs({0}), body_plane({1}) slices_num({2}) and slice_amount({3})'.format(attribs.ndim, body_plane.ndim, slice_num.ndim, slice_amount.ndim))
    print('\t-Slices amounts of {0}th data file: {1}'.format(n,slice_amount))
    print('\t-Memory size usage to load the {0}th data file: {1} bytes'.format(n,memory_size))
    

print('-Total memory usage to load all the {0} data files is:\n\t\t{1} bytes'.format(total_printed_files, total_memory_usage))

#attribs,body_plane,slice_num = loadattribs.load_attribs_and_metadata(attribs_files[0])
'''
def KFoldCrossValidation(train_and_test_indexes, X_data_frame, y_data_frame, k_value=3, kcv_value=10, smote=True, debug=False):
    train_indexes = train_and_test_indexes[0]
    #print('Train Indexes:',train_indexes)
    test_indexes = train_and_test_indexes[1]
    #print('Test Indexes:',test_indexes)
    
    knn = KNeighborsClassifier(n_neighbors=k_value)
    
    #if debug:
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index, "\n")
    
    
    # STEP 1: split data between test and train sets
    if debug:
        print('* Starting train and test sets splitting... ',end='')
    
    y_data = np.ravel(y_data_frame) # Added to solve column-vector issue
    
    X_train, X_test, y_train, y_test = X_data_frame[train_indexes], X_data_frame[test_indexes], y_data[train_indexes], y_data[test_indexes]
    #print('y_data[test_indexes]:',y_data[test_indexes])
    if debug:
        print('Done!')
        
    # print the shapes of the new X objects
    if debug:
        print('* Display X and y objects\'s shape:')
        print('\t X_train.shape: ', X_train.shape)
        print('\t X_test.shape: ', X_test.shape)
        print('\t y_train.shape: ', y_train.shape)
        print('\t y_test.shape: ', y_test.shape)
    
    # SMOTE HERE
    
    if smote:
        # Oversampling training data using SMOTE
        if debug: 
            print('* Starting to oversample training data using SMOTE...')
            print('\t -Number of instances inside TRAIN set from each class BEFORE to apply SMOTE=',(sum(y_train==0),sum(y_train==1),sum(y_train==2)))
            print('\t -Number of instances inside TEST set from each class BEFORE to apply SMOTE=',(sum(y_test==0),sum(y_test==1),sum(y_test==2)))
            print('\t -Number of instances inside TRAIN set from each class BEFORE to apply SMOTE=',(sum(y_train==0),sum(y_train==1),sum(y_train==2)))
            print('\t -Number of instances inside TEST set  from each class BEFORE to apply SMOTE=',(sum(y_test==0),sum(y_test==1),sum(y_test==2)))

        from imblearn.over_sampling import SMOTE
        smt = SMOTE()
        X_train, y_train = smt.fit_sample(X_train, y_train)

        if debug: 
            print('\t -Instances amount from each class AFTER to apply SMOTE=',(sum(y_train==0),sum(y_train==1),sum(y_train==2)))
    
    #print('y_train:',y_train)    
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    #print('y_test:',y_data[test_indexes])
    #print('y_pred=',y_pred)
    
    # comparing actual response values (y_test) with predicted response values (y_pred)
    this_accuracy = metrics.accuracy_score(y_test, y_pred)
    this_confusion_matrix = metrics.confusion_matrix(y_test,y_pred,labels=None,sample_weight=None)

    return this_accuracy,this_confusion_matrix
    



def runKNN(X_data,
        y_data,
        k_value=3, 
        knn_debug=False, 
        use_smote=True, 
        use_rescaling=True,
        cv_type='kcv',
        kcv_value=10,
        use_Pool=False
        ):
    
    # Main variables
    knn = KNeighborsClassifier(n_neighbors=k_value)
    accuracy = 0
    confusion_matrix = []
    
    if knn_debug: 
        print('* Checking arguments formating...')
        print('\t-X_data.shape=',X_data.shape)
        print('\t-y_data.shape=',y_data.shape)
    
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
    
    if knn_debug: 
        print('* Reshaping for this data partition with these dimensions:',new_dimensions)
    
    new_partition = np.reshape(X_data, new_dimensions)
        
    if knn_debug: 
        print('...done')
    
    
    if knn_debug:
        print('* The shape of a line data retrived from the new partition=', new_partition[0].shape)

    ## KNN preparation
    
    # Preparing data to use with PANDAS
    X_pandas = pd.DataFrame(data=new_partition)
    y_pandas = pd.DataFrame(data=y_data)
    
    if knn_debug:
        print('* Preparing data to use with Pandas...')
        print('X_pandas=\n',X_pandas)
        print('y_pandas=\n',y_pandas)

    # Rescalling data
    if use_rescaling:
        if knn_debug: 
            print('* Rescalling data... ',end='')
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        X_pandas = scaler.fit_transform(X_pandas) # Fit your data on the scaler object
        if knn_debug: 
            print('done')
            print('Rescaled X_pandas=\n',X_pandas)
    
    
    #######################################################    
    if cv_type == 'kcv': # K-FOLD CROSS VALIDATION
        scores = []
        matrices = []
        all_results = []
        
        cv = KFold(n_splits=kcv_value, random_state=42, shuffle=True)
        both_indexes = cv.split(X_pandas)
        
        #-----------------------------------
        if not use_Pool: # Single Thread KCrossValidation
            
            
            # Single Thread KCrossValidation
            for indexes in both_indexes:
                
                result = KFoldCrossValidation(indexes, X_pandas,
                                              y_pandas, k_value,
                                              kcv_value, use_smote,
                                              knn_debug)
                all_results.append(result)
            
            for acc_with_cmat in all_results:
                acc = acc_with_cmat[0]
                cmat = acc_with_cmat[1]
                scores.append(acc)
                matrices.append(cmat)
                
            np_scores = np.array(scores)
            best_pos = np_scores.argmax()
            
            accuracy = np.mean(np_scores)
            confusion_matrix = matrices[best_pos]
            
        #-----------------------------------
        else: 
            # Multi Thread KCrossValidation
            # NOT WORKING YET!!
            cores_num = multiprocessing.cpu_count()
            with Pool(processes=cores_num) as p:
                from functools import partial

                all_results = p.map(
                    partial(KFoldCrossValidation,
                            X_data_frame=X_pandas,
                            y_data_frame=y_pandas,
                            k_value=k_value,
                            kcv_value=kcv_value,
                            smote=use_smote,
                            debug=knn_debug),
                    both_indexes)

            for acc_with_cmat in all_results:
                acc = acc_with_cmat[0]
                cmat = acc_with_cmat[1]
                scores.append(acc)
                matrices.append(cmat)
                
            np_scores = np.array(scores)
            best_pos = np_scores.argmax()
            
            accuracy = np.mean(np_scores)
            confusion_matrix = matrices[best_pos]

#########################
    else:
    
        # validation with simple split data between test and train sets
        if knn_debug:
            print('* Starting train and test sets splitting... ',end='')
        X_train, X_test, y_train, y_test = train_test_split(
                X_pandas, 
                np.ravel(y_pandas), 
                test_size=0.3, 
                random_state=12)
        if knn_debug:
            print('done')
    
        # print the shapes of the new X objects
        if knn_debug:
            print('* Display X and y objects\'s shape:')
            print('\t X_train.shape: ', X_train.shape)
            print('\t X_test.shape: ', X_test.shape)
            print('\t y_train.shape: ', y_train.shape)
            print('\t y_test.shape: ', y_test.shape)
    
        
        if use_smote:
            # Oversampling training data using SMOTE
            if knn_debug: 
                print('* Starting to oversample training data using SMOTE...')
                print('\t -Number of instances inside TRAIN set from each class BEFORE to apply SMOTE=',(sum(y_train==0),sum(y_train==1),sum(y_train==2)))
                print('\t -Number of instances inside TEST set from each class BEFORE to apply SMOTE=',(sum(y_test==0),sum(y_test==1),sum(y_test==2)))
                print('\t -Number of instances inside TRAIN set from each class BEFORE to apply SMOTE=',(sum(y_train==0),sum(y_train==1),sum(y_train==2)))
                print('\t -Number of instances inside TEST set  from each class BEFORE to apply SMOTE=',(sum(y_test==0),sum(y_test==1),sum(y_test==2)))
    
            from imblearn.over_sampling import SMOTE
            smt = SMOTE()
            X_train, y_train = smt.fit_sample(X_train, y_train)
        
            if knn_debug: 
                print('\t -Instances amount from each class AFTER to apply SMOTE=',(sum(y_train==0),sum(y_train==1),sum(y_train==2)))
        
        # print the shapes of the new X and y objects
        if knn_debug:
            print('* Display X and y objects\'s shape after apply SMOTE to this sets:')
            print('\t X_train.shape: ', X_train.shape)
            print('\t X_test.shape (should be the same): ', X_test.shape)
            print('\t y_train.shape: ', y_train.shape)
            print('\t y_test.shape (should be the same): ', y_test.shape)
            
        # STEP 2: train the model on the training set
        #knn = KNeighborsClassifier(n_neighbors=k_value)
        knn.fit(X_train, y_train)
        
        # STEP 3: make predictions on the testing set
        y_pred = knn.predict(X_test)
        #if knn_debug: 
        #    print('y_pred=\n',y_pred)
        #    print('y_pred.shape:',y_pred.shape)
        
        # compare actual response values (y_test) with predicted response values (y_pred)
        accuracy = metrics.accuracy_score(y_test, y_pred) 
        confusion_matrix = metrics.confusion_matrix(y_test,y_pred,labels=None,sample_weight=None)


    return accuracy, confusion_matrix

def main(argv):
    
    __TOTAL_RUNNINGS = 50
    
    # KNN Parameters
    K_VALUE = 3
    KNN_DEBUG = False
    pool = False
    cv_type = 'kcv'
    #cv_type = ''
    
    # Seeds test
    USE_KNOWN_GOOD_SLICE_GROUPING = True
    
    # Use this arguments to set the input directory of attributes files
    __USE_SAMPLE_DATA_DIR = True
    __SAMPLE_DATA_DIR = "../../attributes_amostra"
    __FULL_DATA_DIR = "../../attributes2"
    
    attributes_dir = __FULL_DATA_DIR
    csv_file = './ADNI1_Complete_All_Yr_3T.csv'

    if __USE_SAMPLE_DATA_DIR:
        attributes_dir = __SAMPLE_DATA_DIR
    
    # Getting all data
    
    start_time = time.time()
    print('Loading all atributes data... ', end='')
    attribs, body_planes, slice_num, slice_amounts, output_classes = loadattribs.load_all_data(attributes_dir, csv_file)
    end_time = time.time()
    total_time = end_time - start_time
    print('done (total time to load: {0})'.format(total_time))
    
    import deap_alzheimer
    min_slices_values = loadattribs.getSliceLimits(slice_amounts)[0]
    valid_bplanes = loadattribs.getBplanes(slice_amounts)

    print('Slice Limits:',min_slices_values)
    print('valid_bplanes=',valid_bplanes)
    
    
#    def getRandomSliceGrouping(all_slice_amounts,best of mozart
#                           planes = __DEFAULT_BPLANES,
#                           max_length = __DEFAULT_MAX_CONSEC_SLICES,
#                           max_indexes = __DEFAULT_MAX_SLICES_VALUES,    # Maximum value for the first slice index 
#                           dbug=__DEFAULT_DEBUG):
    
    if USE_KNOWN_GOOD_SLICE_GROUPING:
        print('\n* Using a specific known good slice grouping... ', end='')
        
        bplane, start_slice, total_slices = [2, 120, 20]
    else:
        print('\n* Building a random valid slice grouping... ', end='')
        
        bplane, start_slice, total_slices = deap_alzheimer.buildRandomSliceGrouping(
                planes = valid_bplanes,
                length = int(random.random()*20),
                max_indexes = min_slices_values,
                dbug=False)
    
    print('\nDone! Slice grouping: [{0}, {1}, {2}]'.format(bplane,start_slice,total_slices))
    
   
    start_time = time.time()
     
    # Getting some data partition 
    print('\n* Getting the specific data partition using this slice grouping {0}... '.format([bplane,start_slice,total_slices]),end='')
    data_partition = loadattribs.getAttribsPartitionFromSingleSlicesGrouping(
            attribs,
            slice_amounts,
            bplane,
            start_slice,
            total_slices)
    end_time = time.time()
    total_time = end_time - start_time
    print('done.\n\tTotal time to get the data partition= {0}'.format(total_time,))
    print('\n* Data Partition\'s shape= ',data_partition.shape)
    
    

    start_time = time.time()
    print('\n* Starting to run knn classifier {0} times to evaluate this data partition...'.format(__TOTAL_RUNNINGS))
    all_acc = []
    all_cmat = []
    
    for r in list(range(__TOTAL_RUNNINGS)):
        accuracy, conf_matrix = runKNN(data_partition, 
                                       output_classes, 
                                       K_VALUE,
                                       knn_debug=KNN_DEBUG,
                                       use_smote=True,
                                       use_rescaling=True,
                                       cv_type=cv_type,
                                       use_Pool=pool)
        all_acc.append(accuracy)
        all_cmat.append(conf_matrix)
    
    
    end_time = time.time()
    total_time = end_time - start_time
    print('done. Total time to run classifier= {0}'.format(total_time))
    
    
    all_acc = np.array(all_acc)
    
    best_acc = all_acc.max()
    best_acc_pos = all_acc.argmax()
    best_acc_cmat = all_cmat[best_acc_pos]
    worst_acc_cmat = all_cmat[all_acc.argmin()]
    
    print ('\n* Results using K={0} a long {2} runnings:\n{1}'.format(K_VALUE, all_acc, __TOTAL_RUNNINGS))
    
    print ('\tmean={0}: '.format(np.mean(all_acc)))
    print('\tvariance={0}'.format(all_acc.var()))
    print('\tstd={0}'.format(all_acc.std()))
    print ('\tmax={0}'.format(best_acc))
    print ('\tmin={0}'.format(all_acc.min()))
    print('\tConfusion matrix of the best result:\n', best_acc_cmat)
    print('\tConfusion matrix of the worst result:\n', worst_acc_cmat)
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



'''
# Iris dataset load
iris = datasets.load_iris()
x_vals = np.array([x[0:4] for x in iris.data])
y_vals = np.array(iris.target)


# Bullshit
print('iris database (only x elements) shape=',x_vals.shape)
print('iris output set (y) shape=',y_vals.shape)


# changing values to alzheimer imagning data
x_vals = new_partition
y_vals = output_classes


# Esta linha cria uma matriz com 1 na diagonal principal (funcao eye) 
# one hot encoding
y_vals = np.eye(len(set(y_vals)))[y_vals]

# normalize
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

# train-test split
np.random.seed(59)
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices =np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


#KNN

feature_number = len(x_vals_train[0])

k = 15

x_data_train = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)
y_data_train = tf.placeholder(shape=[None, len(y_vals[0])], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)

# manhattan distance
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), axis=2)

# nearest k points
_, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
top_k_label = tf.gather(y_data_train, top_k_indices)

sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
prediction = tf.argmax(sum_up_predictions, axis=1)

# Training

sess = tf.Session()
prediction_outcome = sess.run(prediction, feed_dict={x_data_train: x_vals_train,
                               x_data_test: x_vals_test,
                               y_data_train: y_vals_train})

# evaluation
accuracy = 0
for pred, actual in zip(prediction_outcome, y_vals_test):
    if pred == np.argmax(actual):
        accuracy += 1

print("final accuracy:", accuracy / len(prediction_outcome))
'''