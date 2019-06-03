#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 12:55:41 2019

@author: rodrigo
"""

import os
import numpy as np


def build_all_boxplot_using_pandas(all_models_results_csv_file):
    import pandas as pd
    directory, filename = os.path.split(all_models_results_csv_file)
    bplot_filename = 'bplot_' + os.path.splitext(filename)[0] + '.png'
    
    output_bplot_full_filename = os.path.join(directory,bplot_filename)
    print('boxplot full file name = ', output_bplot_full_filename)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    
    df = pd.read_csv(all_models_results_csv_file)
    bplot = df.drop('Exp',axis=1).boxplot()
    
    fig.savefig(output_bplot_full_filename)
#    
#    names = df.columns.tolist()
#    
#    print('df columns = ',names)
    
    print(df)
    
    


file = '/home/rodrigo/bin/full_run_global_results.csv'

#build_all_bloxplot(file)

build_all_boxplot_using_pandas(file)
    