#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 08:33:13 2019

@author: rodrigo
"""

import loadattribs

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys
import time

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

def runKNN(
        data_partition, 
        output_classes, 
        k_value=3, 
        knn_debug=False, 
        use_smote=True, 
        use_rescaling=True
        ):
    if knn_debug: 
        print('* Checking arguments formating...')
        print('\t-data_partition.shape=',data_partition.shape)
        print('\t-output_classes.shape=',output_classes.shape)
    
    # Data preparation
    try:
        new_dimensions = (data_partition.shape[0],
                          data_partition.shape[1]*data_partition.shape[2])
    except IndexError:
        print('** IndexValue exception')
        print('\tdata_partition.shape=',data_partition.shape)
        print('\toutput_classes.shape=',output_classes.shape)
        print('\t')
        sys.exit(-1)
    
    if knn_debug: 
        print('* Reshaping for this data partition with these dimensions:',new_dimensions)
    
    new_partition = np.reshape(data_partition, new_dimensions)
        
    if knn_debug: 
        print('...done')
    
    
    if knn_debug:
        print('* The shape of a line data retrived from the new partition=', new_partition[0].shape)

    ## KNN preparation
    
    # Preparing data to use with PANDAS
    X_pandas = pd.DataFrame(data=new_partition)
    y_pandas = pd.DataFrame(data=output_classes)
    
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
        
        
    # STEP 1: split data between test and train sets
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
    knn = KNeighborsClassifier(n_neighbors=k_value)
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
    
    # KNN Parameters
    K_VALUE = 5
    
    # Seeds test
    USE_KNOWN_GOOD_SLICE_GROUPING = True
    
    # Use this arguments to set the input directory of attributes files
    __USE_SAMPLE_DATA_DIR = False
    __SAMPLE_DATA_DIR = "../attributes"
    __FULL_DATA_DIR = "../attributes"
    
    attributes_dir = __FULL_DATA_DIR
    csv_file = './ADNI1_Complete_All_Yr_3T.csv'

    if __USE_SAMPLE_DATA_DIR:
        attributes_dir = __SAMPLE_DATA_DIR
    
    # Getting all data
    
    start_time = time.time()
    print('Loading all atributes data... ', end='')
    all_genders = []
    array_all_ages = []
    image_id_dictionary = {}
    attribs, body_planes, slice_num, slice_amounts, output_classes,all_genders, array_all_ages, image_id_dictionary = loadattribs.load_all_data(attributes_dir, csv_file)
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
        print('* Using a specific known good slice grouping... ', end='')
        
        bplane, start_slice, total_slices = [2, 114, 15]
    else:
        print('* Building a random valid slice grouping... ', end='')
        
        bplane, start_slice, total_slices = deap_alzheimer.buildRandomSliceGrouping(
                planes = valid_bplanes,
                length = 30,
                max_indexes = min_slices_values,
                dbug=False)
    
    print('done. Slice grouping: [{0}, {1}, {2}]'.format(bplane,start_slice,total_slices))
    
   
    start_time = time.time()
     
    # Getting some data partition 
    print('* Getting a random data partition using this slice grouping {0}... '.format([bplane,start_slice,total_slices]),end='')
    data_partition = loadattribs.getAttribsPartitionFromSingleSlicesGrouping(
            attribs,
            slice_amounts,
            bplane,
            start_slice,
            total_slices)
    end_time = time.time()
    total_time = end_time - start_time
    print('done.\n\tTotal time to get the data partition= {0}'.format(total_time,))
    print('* Data Partition\'s shape= ',data_partition.shape)
    
    

    start_time = time.time()
    print('* Starting to run knn classifier to evaluate this data partition...')
    accuracy, conf_matrix = runKNN(
            data_partition, 
            output_classes, 
            K_VALUE,
            knn_debug=True,
            use_smote=True,
            use_rescaling=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    print('done. Total time to run classifier= {0}'.format(total_time))
    
    print('\n* Confusion matrix was:\n', conf_matrix)
    print ('* KNN Acurracy with K={0} was: {1}'.format(K_VALUE, accuracy))
    
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
