#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:30:08 2019

@author: rodrigo
"""

import os, csv, sys#, re
#import numpy as np
import list_dir



def divide_ind(individual):
    return ','.join(map(str,individual.split(' ')))[1:-1]

def divide_cmat(cmat):
    new_cmat = ','.join(map(str,cmat.split(' ')))
    new_cmat = ''.join(map(str,new_cmat.split('[')))
    new_cmat = ''.join(map(str,new_cmat.split(']')))
    new_cmat = ','.join(map(str,new_cmat.split(',,')))
    new_cmat = ','.join(map(str,new_cmat.split(',,')))
#    for line in cmat:
#        new_cmat = new_cmat + line.
    return new_cmat

def get_new_header():
    return 'exp,bplane,first_slice,slices_amount,fitness,true_CN,CN_as_MCI,CN_as_AD,MCI_as_CN,true_MCI,MCI_as_AD,AD_as_CN,AD_as_MCI,true_AD\n'


def build_fixed_full_filename(original_full_filename):
    directory, filename = os.path.split(original_full_filename)
    new_file = 'FIXED_' + filename
    full_name_fixed_file = os.path.join(directory, new_file)
    return full_name_fixed_file, directory


def append_line_to_detailed_results_csv_file(
        line,
        original_full_filename, #built by build_experiment_output_filename inside saveExperimentsData function
        dbug =False):
    
    append_mode = "a"
        
    full_name_fixed_file, directory = build_fixed_full_filename(original_full_filename)
    
    # checking output file
    if not os.path.exists(full_name_fixed_file):
        # creates output dir when path doesnt exist
        if directory != '' and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except os.error:
                print ('*** ERROR: Output directory (%s) can not be created\n' % directory)
                sys.exit(1)
        
    # Writting to output file
    try :
        output_file = open(full_name_fixed_file, append_mode)
        output_file.write(line)
        output_file.close()
    except os.error:
        output_file.close()
        print(" *** ERROR: file %s can not be written" % full_name_fixed_file)
        exit(1)
    
    return full_name_fixed_file



def get_results_from_bad_format_results_file(results_file):

    print('Reading results file=',results_file)
    
    results = []
        
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] == 'Experiment': # Header!!
                        results.append(get_new_header())
                    else:
                        if len(row) == 4:
                            exp,ind,fit,cmat = row
                            line = '{exp},{ind},{fit},{cmat}\n'.format(
                                    exp=exp, 
                                    ind=divide_ind(ind), 
                                    fit=fit, 
                                    cmat=divide_cmat(cmat))
                            results.append(line)
                        else:
                            print('Error while processing file: ', results_file)
                            raise ValueError('Bad format row: ',row)
                            
        except os.error:
            print("*** ERROR: Attributes file %s can not be readed (os.error in load_attribs function)" % results_file)
        except UnicodeDecodeError:
            print('Error processing file ({0})'.format(results_file))
    else:
        print('File {file} can not be found!'.format(file=results_file))
    
    return results
    
def create_fixed_results_files(files_path, extension='.cvs'):

    all_files = list_dir.new_list_files(input_dir='/home/rodrigo/bin/full_run',extension='.csv')
    
    # first remove all destination files
    all_fixed_files = [build_fixed_full_filename(original_file) for original_file in all_files]
    for fixed_file in all_fixed_files:
        removed = True
        try:
            os.remove(fixed_file[0])
        except OSError:
            removed = False
        if removed: print('File removed: ',fixed_file[0])

    # now create files, one by one, and put fixeed lines there
    for file in all_files:
        #test_file = '/home/rodrigo/bin/results_2019-05-29_1559146694_KNN.csv'
        results = get_results_from_bad_format_results_file(file)
        
        
    
        for line in results:
            append_line_to_detailed_results_csv_file(line,file)

def main(argv):
    dir_path = '/home/rodrigo/bin/full_run'
    create_fixed_results_files(dir_path)

if __name__ == "__main__":    
    main(sys.argv)
    