#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:11:15 2020

@author: rodrigo

This class helps to answer, from a given volumetric 
image, which is the reference slice at a specific body plane
"""

import evaluate_refslices_predictors as erp
import build_refslices_train_set as brts
import pandas as pd
import numpy as np

 
__SCALE_ALL_MEAN_LOAD_FILE = erp.__SCALER_ALL_MEAN_NP_FILE + '.npy'
__SCALE_ALL_STD_LOAD_FILE = erp.__SCALER_ALL_STD_NP_FILE + '.npy'

__MODEL_NAME = 'RF'


def find_reference_slice(bodyplane,image_id):
    global __SCALE_ALL_MEAN_LOAD_FILE,__SCALE_ALL_STD_LOAD_FILE
    
    #global brts.__ATTRIBUTES_DIR
    attribs_dir = brts.__ATTRIBUTES_DIR
    attribs_file = brts.find_attributes_file('I'+image_id,attribs_dir)
    
    attribs_df = pd.read_csv(attribs_file, delimiter=',',header=None)
    
    first_slice_position = 0
    
    all_mean = np.load(__SCALE_ALL_MEAN_LOAD_FILE)
    all_std = np.load(__SCALE_ALL_STD_LOAD_FILE)
    
    for slice_position  in range(first_slice_position,80):

        # Picking values for slice_position
        #print('# Picking values for {0}th slice position..'.format(slice_position))
        attribs_np = brts.get_attribs_from_dataframe(attribs_df, bodyplane, slice_position)
        
        # Rescaling slice values
        X_scaled = (attribs_np - all_mean) / all_std
        #print('X_scaled:\n',X_scaled)
        
        
        # Loading trained model
        __MODEL_DUMP_FILENAME = erp.__MODEL_DUMP_FILENAME
        global __MODEL_NAME
        from joblib import load
        dump_filename = '{0}{1}-{2}.joblib'.format(__MODEL_DUMP_FILENAME,bodyplane,__MODEL_NAME)
        model = load(dump_filename)
        print('# Classifying {0}th slice for the refslice status..'.format(slice_position))
        y_pred = model.predict(X_scaled.reshape(1,-1))
        
        if y_pred == 1:
            print('# The refslice was found at {0}th position. Leaving now...'.format(slice_position))
            break;

    print('Class found was {0} to image {1} at slice {2} in the body plane {3}'.format(y_pred,'I'+image_id,slice_position,bodyplane))
    
    
    
    # Classifying picked slice
    
    
    
find_reference_slice(0,'119735')
    