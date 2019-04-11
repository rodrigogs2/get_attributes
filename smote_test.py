#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 08:47:44 2019

@author: rodrigo
"""

import loadattribs

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np
import sys
import time
import knn_alzheimer



def main(argv):
    __USE_DATA_SAMPLE = True
    
    
    # KNN Parameters
    K_VALUE = 5
    
    # Use this arguments to set the input directory of attributes files
    attributes_dir = "../../attributes_amostra"
    
    if not __USE_DATA_SAMPLE:
        attributes_dir = "../../attributes2"
    csv_file = './ADNI1_Complete_All_Yr_3T.csv'
    
    # Getting all data
    
    start_time = time.time()
    print('Loading all atributes data...')
    attribs, body_planes, slice_num, slice_amounts, output_classes = loadattribs.load_all_data(attributes_dir, csv_file)
    end_time = time.time()
    total_time = end_time - start_time
    print('...done (total time to load: {0})'.format(total_time))
    
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
    
    print('Getting a random valid slice grouping...')
    bplane, start_slice, total_slices = deap_alzheimer.buildRandomSliceGrouping(
            planes=valid_bplanes,
            length = 30,
            max_indexes = min_slices_values,
            dbug=False)
    print('...done')
    
    print('slice grouping found:\n\tbplane={0},first_slice={1},total_slices={2}'.format(bplane,start_slice,total_slices))
    print('Individual analysed: [{0}, {1}, {2}]'.format(bplane,start_slice,total_slices))
    
    
    start_time = time.time()
     
    # Getting some data partition 
    print('Getting some data partition using this last slice grouping ({0})...'.format((bplane,start_slice,total_slices)))
    data_partition = loadattribs.getAttribsPartitionFromSingleSlicesGrouping(attribs,
                                                          slice_amounts,
                                                          bplane,
                                                          start_slice,
                                                          total_slices)
    
    
    end_time = time.time()
    total_time = end_time - start_time
    print('...done \nTotal time to get the data partition (bplane={1},first_slice={2},total_slices={3}): {0}'.format(total_time,bplane,start_slice,total_slices))
    

    start_time = time.time()
    print('Starting to run knn classifier to evaluate this partition of data...')
    accuracy, conf_matrix = knn_alzheimer.runKNN(data_partition, output_classes, K_VALUE,use_smote=False,use_rescaling=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    print('...done (total time to run classifier: {0})'.format(total_time))
    
    print('Individual analysed: [{0}, {1}, {2}]'.format(bplane,start_slice,total_slices))
    print('\nConfusion matrix was:\n', conf_matrix)
    print ('KNN Acurracy with K={0} was: {1}'.format(K_VALUE, accuracy))
    
    
    ###########################################################################
    
    K_VALUE = 5
    # SMOTE BEGINS HERE
    smote_debug = True
    print('* a partition data shape=',data_partition.shape)
    
    # Data preparation
    try:
        # Transforming 3D data to a 2D data
        new_dimensions = (data_partition.shape[0],
                          data_partition.shape[1]*data_partition.shape[2])
    except IndexError:
        print('** IndexValue exception')
        print('\tdata_partition.shape=',data_partition.shape)
        print('\output_classes.shape=',output_classes.shape)
        print('\t')
        sys.exit(-1)
    
    
    new_partition = np.reshape(data_partition, new_dimensions)
    
    #scaled_new_partition = preprocessing.scale(new_partition)

    
    used_partition = new_partition
    
    for i in range(2):
        if i == 1:
            print('\n*** NOW we will do the same however using scaled and balanced data:')
            #used_partition = scaled_new_partition
        

        if smote_debug and False: 
            print('* DIMENSION for the new partition array=',new_dimensions)
            print('* the new partition data shape=',used_partition.shape)
            print('* the output array shape=',output_classes.shape)
            print('* shape of an input instance retrived from the new partition=', used_partition[0].shape)
    
        ## KNN preparation
        X_pandas = pd.DataFrame(data=used_partition)
        #print('X_pandas=\n',X_pandas)
        #_pandas = pd.DataFrame(data=np.ravel(output_classes,order='C'))
        y_pandas = pd.DataFrame(data=output_classes)
        y_pandas.columns = ['Class']
        #print('y_pandas=\n',y_pandas)
        #print('y_pandas values (without balancing)=\n',pd.value_counts(y_pandas['Class']))
        if i == 1: # STANDARDIZING...
            # Get column names first
            #names = ['Class']
            # Create the Scaler object
            scaler = preprocessing.StandardScaler()
            # Fit your data on the scaler object
            X_pandas = scaler.fit_transform(X_pandas)
            #X_pandas = pd.DataFrame(scaled_df, columns=names)
         
        # STEP 1: split data between test and train sets
        X_train, X_test, y_train, y_test = train_test_split(X_pandas, np.ravel(y_pandas), test_size=0.3, random_state=12)
        
        #import matplotlib.pyplot as plt
        if i == 0: 
            print('classes count(before SMOTE)= ',(sum(y_train==0),sum(y_train==1),sum(y_train==2)))
            #pd.value_counts(y_pandas['Class']).plot.bar()
            #plt.title('Unbalanced Alzheimer class histogram')
            #plt.xlabel('Class')
            #plt.ylabel('Frequency')
   
        elif i == 1:
            #pd.value_counts(y_pandas['Class']).plot.bar()
            #plt.title('Balanced and Normalized Alzheimer class histogram')
            #plt.xlabel('Class')
            #plt.ylabel('Frequency')

            
        
            from imblearn.over_sampling import SMOTE
            smt = SMOTE()
            X_train, y_train = smt.fit_sample(X_train, y_train)
            #print('classes count(after SMOTE)=\n',np.bincount(y_train))
            print('classes count(after SMOTE)=',(sum(y_train==0),sum(y_train==1),sum(y_train==2)))
        
        
         # STEP 2: train the model on the training set
        knn = KNeighborsClassifier(n_neighbors=K_VALUE)
        knn.fit(X_train, y_train)
        
        # STEP 3: make predictions on the testing set
        y_pred = knn.predict(X_test)
        #if smote_debug: 
            #print('y_pred=\n',y_pred)
            #print('y_pred.shape:',y_pred.shape)
        
        # compare actual response values (y_test) with predicted response values (y_pred)
        accuracy = metrics.accuracy_score(y_test, y_pred) 
        confusion_matrix = metrics.confusion_matrix(y_test,y_pred,labels=None,sample_weight=None)
        
        print('Individual analysed: [{0}, {1}, {2}]'.format(bplane,start_slice,total_slices))
        print ('KNN Acurracy with K={0} was: {1}'.format(K_VALUE, accuracy))
        print('confusion matrix:\n',confusion_matrix)
    

    
    
    
    '''
    # STEP 1: split data between test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X_pandas, y_pandas, test_size=0.3, random_state=12)
    
    # print the shapes of the new X objects
    if smote_debug: 
        print('X_train.shape:', X_train.shape)
        print('X_test.shape:', X_test.shape)
    
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    
    # print the shapes of the new y objects
    if smote_debug: 
        print('y_train.shape:',y_train.shape)
        print('y_test.shape:',y_test.shape)
    
    # STEP 1: adjust shape of y vectors
    np.ravel(y_train)
    
    # STEP 2: train the model on the training set
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train, y_train)
    
    # STEP 3: make predictions on the testing set
    y_pred = knn.predict(X_test)
    if smote_debug: 
        print('y_pred=\n',y_pred)
        print('y_pred.shape:',y_pred.shape)
    
    # compare actual response values (y_test) with predicted response values (y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred) 
    confusion_matrix = metrics.confusion_matrix(y_test,y_pred,labels=None,sample_weight=None)
    '''

    
    
    return 0
    
    

if __name__ == "__main__":    
    main(sys.argv)


