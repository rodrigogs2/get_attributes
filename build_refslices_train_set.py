#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:02:35 2020

@author: rodrigo

This class builds the refslices database in order to use it to train a 
classifier which help to answer if either a given slice is a ref slice or 
it is not a ref slice
"""

import pandas as pd
import os, csv, sys, re
import getopt
#import numpy as np
import list_dir


__ADJUST_REFSLICE_POSITION = [2,0,0] # recalculates ref_slice values by axis
__NON_REFSLICE_SHIFT = [[0,-5,-5],[-4,-10,-10]] # used to point non ref_slice examples
__AXIS = 0 # default Axis

 # ADNI CSV file columns
__ADNI_CSVFILE_COLUMNS_DICTIONARY = {
        'ADNI #': 0,
        'Year' : 1,
        'Magnetic_Field': 2,
        'Image Data ID': 3,
        'Subject' : 4,
        'Group' : 5,
        'Sex' : 6,
        'Age' : 7,
        'Visit' : 8,
        'Modality' : 9,
        'Description' : 10,
        'GW' : 11,
        'Type' : 12,
        'Acq Date' : 13,
        'Format' : 14,
        'Downloaded' : 15,
        'Date Yield' : 16,
        'Format.1' : 17,
        'Data upload' : 18,
        'Axis0-RefSlice' : 19,
        'Axis1-RefSlice' : 20,
        'Axis2-RefSlice' : 21,
        'Axis2-RefSliceHard' : 22
        }

#__ADNI_CSV_IMAGE_ID_COLUMN_INDEX = 3
#__ADNI_CSV_COLUMN_OF_FIRST_SLICE_ATTRIBUTE = 19

__INPUT_CSV = "./com_OFFSET_ADNI1_Complete_All_Yr_3T.csv"
__ATTRIBUTES_DIR = '../../attributes2'
__VERBOSE = False
#__COLUMNS_TO_DROP_FROM_ADNI_CSV = ['ADNI #','Year','Magnetic_Field','Subject','Visit','Modality','Description','GW','Type','Acq Date','Format','Downloaded','Date Yield','Format','Data upload']

# REFSLICE CSV File columns                                   
__REFSLICE_CSVFILE_COLUMNS_DICTIONARY = {
    'image_id' : 0,
    'ref_slice' : 1,
    'attribs_file' : 2,
    'age' : 3,
    'gender' : 4,
    'ref_class' : 5
    }

__DEFAULT_OUTPUT_REFSLICES_CSV_FILE = '../ref-slices_attributes.csv'


def split_dataframe(refslices_csv_as_dataframe):    
    df = refslices_csv_as_dataframe
    
    # Getting right columns positions
    header = df.columns.values.tolist()
    column_of_ref_class = __REFSLICE_CSVFILE_COLUMNS_DICTIONARY['ref_class']
    column_of_first_attribute =  column_of_ref_class + 1

    X_data = df.iloc[:, column_of_first_attribute:]
    y_data = df.iloc[:, column_of_ref_class]
    M_data = df.iloc[:, :column_of_ref_class]
    
    return X_data, y_data, M_data, header


def load_dataframes_from_csv(axis_num):
    # Formatting file name
    input_refslice_csv_file = build_refslice_csv_filename(axis_num)
                       
    # Reading full CSV file
    df = pd.read_csv(input_refslice_csv_file)
    
    # Returnning splitted data as a (X_data, y_data, M_data) tuple
    return split_dataframe(df) # 


def get_attribs_from_dataframe(attribs_df, axis_num, slice_num):
    attribs = attribs_df.loc[(attribs_df[0]==axis_num) & (attribs_df[1]==slice_num)]
    return attribs.iloc[0,2:].values

def build_refslices_dataframe(csv_file=__INPUT_CSV, axis_num=0, attributes_dir=__ATTRIBUTES_DIR):
    global __ADJUST_REFSLICE_POSITION, __NON_REFSLICE_SHIFT
    global __ADNI_CSVFILE_COLUMNS_DICTIONARY, __REFSLICE_CSVFILE_COLUMNS_DICTIONARY
    
    image_id_column_index = __ADNI_CSVFILE_COLUMNS_DICTIONARY['Image Data ID']
    refslice_column_index = __ADNI_CSVFILE_COLUMNS_DICTIONARY['Axis0-RefSlice'] + axis_num
    age_column_index = __ADNI_CSVFILE_COLUMNS_DICTIONARY['Age']
    gender_column_index = __ADNI_CSVFILE_COLUMNS_DICTIONARY['Sex']
    
    image_id_list = []
    gender_list = []
    age_list = []
    slicenum_list = []
    files_list = []
    attribs_list = []
    ref_class_list = []
   
    print('Reading ADNI csv file: ' + csv_file)
    
    if os.path.exists(csv_file):
        try:
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    ref_slice = row[refslice_column_index]
                    age = row[age_column_index]
                    gender = row[gender_column_index]
                    
                    if ref_slice:
                        try:
                            image_id = row[image_id_column_index]
                            int_ref_slice = int(ref_slice)

                        except ValueError:
                            pass
                            #print('* Value of variable int_index ({0}) can not yield a integer (ID={1})!'.format(int_index, image_id))

                        attribs_file = find_attributes_file('I'+image_id,attributes_dir)
                        #print('attribs_file: ',attribs_file)
                        if attribs_file:
                            
                            attribs_df = pd.read_csv(attribs_file, delimiter=',',header=None)
                            #print(attribs_df)
                            corrected_position = int_ref_slice + __ADJUST_REFSLICE_POSITION[axis_num]
                            
                            #attribs_df = attribs_df.loc[(attribs_df[0]==axis_num) & (attribs_df[1]==int_ref_slice+adjustment)]
                            attribs_np = get_attribs_from_dataframe(attribs_df, axis_num, corrected_position)
                            #attribs_np = attribs_df.iloc[0,2:].values
                            
                            #dfObj.loc[dfObj['Name']=='jack']['Country']
                            
                            #print('attribs_df dimensions: ',attribs_df.size)
                            
                            attribs_list.append(attribs_np)
                            slicenum_list.append(int_ref_slice)
                            image_id_list.append(image_id)
                            files_list.append(attribs_file)
                            ref_class_list.append(1)
                            age_list.append(age)
                            gender_list.append(gender)
                            
                            for sublist in __NON_REFSLICE_SHIFT:
                                position = sublist[axis_num]+corrected_position
                                non_refslice_attribs_np = get_attribs_from_dataframe(attribs_df, axis_num, position)
                                
                                ref_class_list.append(0)
                                image_id_list.append(image_id)
                                files_list.append(attribs_file)
                                slicenum_list.append(position)
                                attribs_list.append(non_refslice_attribs_np)
                                age_list.append(age)
                                gender_list.append(gender)
                                                    
        except os.error:
            print("*** ERROR: CSV file %s can not be readed (os.error)" % csv_file)
        except UnicodeDecodeError:
            print('Error processing file ({0})'.format(csv_file))
    
    idl = pd.DataFrame(image_id_list,columns=['image_id'])
    scl = pd.DataFrame(slicenum_list,columns=['ref_slice'])
    afl = pd.DataFrame(files_list,columns=['attribs_file'])
    agl = pd.DataFrame(age_list,columns=['age'])
    gel = pd.DataFrame(gender_list,columns=['gender'])
    rcl = pd.DataFrame(ref_class_list,columns=['ref_class'])
    att = pd.DataFrame(attribs_list)
    
        
    #M_data.append((image_id,gender,age,alz_class,ref_class))
    
    df = pd.concat([idl,scl,afl,agl,gel,rcl,att],axis=1)
    
    return df


def build_refslice_csv_filename(axis_num):
    global __DEFAULT_OUTPUT_REFSLICES_CSV_FILE
    o_file = __DEFAULT_OUTPUT_REFSLICES_CSV_FILE
    
    output_dir,filename = os.path.split(o_file)
    
    output_filename_without_extension, file_extension = os.path.splitext(filename)
    
    new_output_file = "{0}-axis{1}{2}".format(
            output_filename_without_extension, 
            axis_num,
            file_extension)
    
    output_file = os.path.join(output_dir, new_output_file)
    return output_file


def build_and_save_refslices_dataframe(csv_file=__INPUT_CSV, 
                           axis_num=0, 
                           attributes_dir=__ATTRIBUTES_DIR, 
                           o_file=__DEFAULT_OUTPUT_REFSLICES_CSV_FILE):
        
    df = build_refslices_dataframe(csv_file,axis_num,attributes_dir)
    
    
    output_file = build_refslice_csv_filename(axis_num)

    print('DF ready to save to destination file ',output_file)
    print('current_dir: ', os.getcwd())

    df.to_csv(path_or_buf=output_file, index=False)
    
    return df
    
    

def get_image_ID(attributes_filename):
    all_image_id = re.findall(r'I[0-9]+',attributes_filename) # returns a array with all regular exp matches
    if len(all_image_id) > 0:
        return all_image_id[0]
    else:
        return ''


def find_attributes_file(target_image_id, attributes_dir=__ATTRIBUTES_DIR):
    
    # Getting all attributes files from attributes directory
    attribs_files = list_dir.list_files(attributes_dir,".txt")
    
    for file in attribs_files:
        file_image_id = get_image_ID(file)
        
        if file_image_id == target_image_id:
            return file


########
# FUNCAO MAIN
def main(argv):
    rebuild_ok = False
    bplanes = [0,1,2]
    global __ADJUST_REFSLICE_POSITION, __NON_REFSLICE_SHIFT
    __ADJUST_REFSLICE_POSITION = [2,0,0] # recalculates ref_slice values by axis
    __NON_REFSLICE_SHIFT = [[-2,-2,-2],[-6,-6,-6],[-4,-4,-4]] # used to point non ref_slice examples
    
    
    model_name = 'RF'
    #bplanes = [0]

    for plane in bplanes:
        if rebuild_ok:
            df = build_and_save_refslices_dataframe(axis_num=plane)
            X_data,y_data,M_data,head = split_dataframe(df)
        else:
            X_data,y_data,M_data,head = load_dataframes_from_csv(plane)
            
        print('X_data:\n', X_data.iloc[:9 , :6])
        print('y_data:\n', y_data.iloc[:9])
        print('M_data:\n', M_data.iloc[:9 ,:])
        
        import evaluate_refslices_predictors as ep
        result_dic = ep.evaluate_model(X_data, 
                          y_data,
                          model_name,
                          10)
        
        print('Result for plane {1}: mean_acc = {0} std_acc = {2}'.format(result_dic['mean_acc'],plane,result_dic['std_acc']))

#        evaluate_model(X_data, y_data, model_name,
#                   folds, cv_seed=7, cv_shuffle=True,
#                   smote=True, rescaling=True, cores_num=1, 
#                   maximization=True, stratified_kfold=True,
#                   pca=False, debug=False):        

    return 0

def display_help():
    print('Help screen is under construction...')

def main_final(argv):    
    bplanes = [0,1,2]
    input_csv_file = ''
    attribs_dir = ''
    out_dir = './'
    rebuild_ok = False
#    model = ''
#    seeds_file = ''
    csv_file_ok = False
    attribs_dir_ok = False
    out_dir_ok = False
#    model_ok = False
#    seeds_file_ok = False
    verbose_ok = False
#    multi_cpu_ok = False
#    number_of_experiments = 1
#    
    try:
        opts, args = getopt.getopt(argv[1:],"hrc:a:o:v",["rebuild","axis=","input_csv=","attributes_dir=","output_dir=","verbose"]) 
    except getopt.GetoptError:
        display_help()
        sys.exit(1)
    for opt, arg in opts:
        if opt in ('-h','--help'):
            display_help()
            sys.exit(0);
        elif opt in ("-r", "--rebuild"):
            rebuild_ok = True        
        elif opt in ("-c", "--input_csv"):
            input_csv_file = arg
            csv_file_ok = True
        elif opt in ("-a", "--attributes_dir"):
            attribs_dir = arg
            attribs_dir_ok = True
        elif opt in ("-o", "--output_dir"):
            out_dir = arg
            out_dir_ok = True
        elif opt in ("-v", "--verbose"):
            verbose_ok = True
        elif opt in ("-a", "--axis"):
            try:    
                axis_num = int(arg)
            except ValueError:
                print('Error: argument {0} must be a integer!'.format(arg))
                display_help()
                sys.exit(0)
            except Exception as err:
                print('Error: An exception has rised on try of integer conversion of the argument {0}.\n\tCause: '.format(arg,err.__cause__))
                display_help()
                sys.exit(0)
            if axis_num < 0 or axis_num > 2:
                print('Error: argument {0} must be between 0 and 2 (both inclusive)!'.format(arg))
                display_help()
                sys.exit(0)
            else:
                bplanes = [axis_num]
        
    
    if csv_file_ok and attribs_dir_ok:
            
        print('* Loading CSV input data...')
        print('\t* Attribs directory is: {0}'.format(attribs_dir))
        print('\t* Input CSV file is: {0}'.format(input_csv_file))

        if out_dir_ok:
            global __OUTPUT_DIRECTORY
            __OUTPUT_DIRECTORY = out_dir
            print ('\t* Output dir is: {0}'.format(__OUTPUT_DIRECTORY))
       
        if verbose_ok:
            global __VERBOSE
            __VERBOSE = True
                   
        # Loading all data just once
        #global __VALID_GENDERS, __MAX_AGE, __MIN_AGE
        
        
        #bplanes = [0,1,2]
        #bplanes = [0]

        for plane in bplanes:
            if rebuild_ok:
                df = build_and_save_refslices_dataframe(axis_num=plane)
                X_data,y_data,M_data,header = split_dataframe(df)
            else:
                X_data,y_data,M_data,header = load_dataframes_from_csv(plane)
                
            print('X_data:\n', X_data.iloc[:9 , :6])
            print('y_data:\n', y_data.iloc[:9])
            print('M_data:\n', M_data.iloc[:9 ,:])
    
        return 0
        
        
    else:
        display_help()
    

if __name__ == "__main__":    
    main(sys.argv)
    
