#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:47:22 2020

@author: rodrigo
"""


import evaluate_refslices_predictors as erp
import build_refslices_train_set as brts
import pandas as pd
import numpy as np
import deap_alzheimer as deapaz
import os.path
import matplotlib.pyplot as plt

__RESULTS_DIR = '../results/results_removing_blacklisted_images/'
__OUTPUT_CSV_FILENAME='synthetized_results.csv'
 
__ALL_MODELS = deapaz.build_models_names_list
__BODY_PLANES = [0,1,2]
__COLORS = ['b','g','r']
__ANATOMICAL_PLANES = ['tranversal plane','coronal plane','sagittal plane']


__EXPERIMENTS_GENDER_PROFILES = ['MF','F','M']
__ALL_BEST_SLICES_FILE_PREFIX = 'all_best_ind-'


def get_best_slices(gender_profile):
    
    global __EXPERIMENTS_GENDER_PROFILES, __ALL_BEST_SLICES_FILE_PREFIX
    global __BODY_PLANES
    
    best_ind_files = []
    best_ind_files.append(os.path.join(__RESULTS_DIR, (__ALL_BEST_SLICES_FILE_PREFIX + gender_profile + '.csv')))
    
    best_slices_list = [] # Tere will be one list to each body plane to represent slices found on that plane
    for i in range(0, len(__BODY_PLANES)):
        best_slices_list.append([])

    for file in best_ind_files:
        print('+++ Reading results file: ',file)
        df = pd.read_csv(file, delimiter=',',header=0)    
                
        for i in range (0,df.shape[0]):
            cur_bplane = int(df.iloc[i][0]) # getting row's bplane value
            first_slice = int(df.iloc[i][1]) # getting row's first slice value
            amount = int(df.iloc[i][2]) # getting row's amount value
            
            if amount > 0:
                for slice_num in range (first_slice,first_slice+amount):
                    try:
                        best_slices_list[cur_bplane].append(slice_num)                             
                    except IndexError:
                        print('cur_bplane={0} slice_num={1} amount={2}'.format(cur_bplane,slice_num,amount))

    return best_slices_list
                        

def build_histograms():
    global __SCALE_ALL_MEAN_LOAD_FILE,__SCALE_ALL_STD_LOAD_FILE, __BODY_PLANES,__COLORS,__ANATOMICAL_PLANES
    
    for cur_profile in __EXPERIMENTS_GENDER_PROFILES:
        
        best_slices_list = get_best_slices(cur_profile)
        
    
        plt.subplots_adjust(left=None, 
                            bottom=None, 
                            right=None, 
                            top=None, 
                            wspace=None, 
                            hspace=1.0)
    
        fig = plt.figure(figsize=(64,16))
        fig.suptitle('Slices Histogram grouped by anatomical plane')
        #my_bins=[50, 100, 150, 180, 195, 205, 220, 250, 280]

        
        all_df_slices = []
        #print('df_slices.head:\n',df_slices.head(10))
        
        for plane, bp in zip(__ANATOMICAL_PLANES, __BODY_PLANES):
            print('-- building DataFrame for: ',plane)
            df = pd.DataFrame(columns=[plane])
            #col = pd.DataFrame(best_slices_list[bp],columns=[plane])
            if not len(best_slices_list[bp]) == 0:
                df[plane] = best_slices_list[bp]
            print('col.head:\n',df.head(10))
            
            all_df_slices.append(df)
            
        #df_slices[__ANATOMICAL_PLANES[bp]] = best_slices_list[bp]
            
            
        #df_slices = pd.DataFrame(best_slices_list)
        #df_slices.columns=__ANATOMICAL_PLANES
        #print('df_slices.head:\n',df_slices.head)

        for df_slices,bp in zip(all_df_slices,__BODY_PLANES):
            
            if not df_slices.empty:
                ax = df_slices.plot(kind='hist',bins=200,figsize=(12,4),xlim=(40,140),color=__COLORS[bp])
                ax.set(title="Choosen Slices at body plane "+str(bp),xlabel="Slice Index")
            
#        for bp in __BODY_PLANES:
#            df_slices = pd.DataFrame(best_slices_list[bp])
#            df_slices.columns = __ANATOMICAL_PLANES
#            
#            if not df_slices.empty:
#                ax = df_slices.plot(kind='hist',bins=200,figsize=(12,4),xlim=(40,140),color=__COLORS[bp])
#                ax.set(title="Choosen Slices at body plane "+str(bp),xlabel="slice num")
                

def build_all_in_one_histogram():
    global __SCALE_ALL_MEAN_LOAD_FILE,__SCALE_ALL_STD_LOAD_FILE, __BODY_PLANES
    global __COLORS,__ANATOMICAL_PLANES
    
    
    for cur_profile in __EXPERIMENTS_GENDER_PROFILES:
        fig, ax1 = plt.subplots(figsize=(16, 8))
        #ax1.set_xlim(40,140)
        fig.suptitle('Best slices grouped by anatomical plane\nUsed genders = {0}'.format(cur_profile))
        
        best_slices_list = get_best_slices(cur_profile)
    
#        plt.subplots_adjust(left=None, 
#                            bottom=None, 
#                            right=None, 
#                            top=None, 
#                            wspace=None, 
#                            hspace=1.0)
    
        #fig = plt.figure(figsize=(16,12))
        #fig.suptitle('Slices Histogram grouped by body plane')
        
        bins=100
        #bins = np.linspace(40, 140, 140)
        
        ax1.hist(best_slices_list, bins, label=__ANATOMICAL_PLANES )
        plt.legend(loc='upper right')
        #plt.tight_layout()
        plt.show()
       
    
build_all_in_one_histogram()

    