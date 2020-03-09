#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:02:35 2020

@author: rodrigo
"""

import pandas as pd
import os, csv, sys, re
import numpy as np
import list_dir


__ADJUST_REFSLICE_POSITION = [2,0,0] # recalculates ref_slice values by axis
__NON_REFSLICE_SHIFT = [[0,-5,-5],[-4,-10,-10]] # used to point non ref_slice examples
__AXIS = 0 # default Axis

# ADNI CSV file columns 
#__ADNI_CSVFILE_COLUMNS_DICTIONARY = {
#        'image_id' : 0,
#        'ref_slice' : 1,
#        'attribs_file' : 2,
#        'age' : 3,
#        'gender' : 4,
#        'ref_class' : 5
#        }

__ADNI_CSV_IMAGE_ID_COLUMN_INDEX = 3
__ADNI_CSV_COLUMN_OF_FIRST_SLICE_ATTRIBUTE = 19

__INPUT_CSV = "./com_OFFSET_ADNI1_Complete_All_Yr_3T.csv"
__ATTRIBUTES_DIR = '../../attributes2'
__COLUMNS_TO_DROP_FROM_ADNI_CSV = ['ADNI #','Year','Magnetic_Field','Subject','Visit','Modality','Description','GW','Type','Acq Date','Format','Downloaded','Date Yield','Format','Data upload']
                                   
__REFSLICE_CSVFILE_COLUMNS_DICTIONARY = {
    'image_id' : 0,
    'ref_slice' : 1,
    'attribs_file' : 2,
    'age' : 3,
    'gender' : 4,
    'ref_class' : 5
        }
                     
                     
def get_data_frame_from_csv(input_csv=__INPUT_CSV, columns_to_drop=__COLUMNS_TO_DROP_FROM_ADNI_CSV):
    df = pd.read_csv(input_csv, delimiter=',')
    return df#df.drop(columns=columns_to_drop)

def load_dataframes_from_csv(refslice_csv):
    #csv_file = '../ref-slices_attributes-axis{0}.csv'.format(plane)
    df = get_data_frame_from_csv(input_csv=refslice_csv)
    X_data = df.iloc[:, __REFSLICE_CSV__REFSLICE_CLASS_COLUMN + 1:]
    #print('X_data:\n',X_data)
    y_data = df.iloc[:, __REFSLICE_CSV__REFSLICE_CLASS_COLUMN]
    #print('y_data:\n',y_data)
    
    return 0

def get_attribs_from_dataframe(attribs_df, axis_num, slice_num):
    attribs = attribs_df.loc[(attribs_df[0]==axis_num) & (attribs_df[1]==slice_num)]
    return attribs.iloc[0,2:].values

def build_refslices_dataframe(csv_file=__INPUT_CSV, axis_num=0, attributes_dir=__ATTRIBUTES_DIR):
    global __ADJUST_REFSLICE_POSITION, __NON_REFSLICE_SHIFT, __ADNI_CSV_IMAGE_ID_COLUMN_INDEX , __ADNI_CSV_COLUMN_OF_FIRST_SLICE_ATTRIBUTE, __REFSLICE_CSV_IMAGEID_COLUMN
    
    image_id_column_index = __ADNI_CSV_IMAGE_ID_COLUMN_INDEX 
    slice_column_index = __ADNI_CSV_COLUMN_OF_FIRST_SLICE_ATTRIBUTE + axis_num
    
    
    
    refslice_csv_imageid_column_index = __REFSLICE_CSV_IMAGEID_COLUMN 
    
    #M_data.append((image_id,gender,age,alz_class,ref_class))
    gender_list = []
    age_list = []
    
    image_id_list = []
    slicenum_list = []
    files_list = []
    attribs_list = []
    ref_class_list = []
    
    #attributes_list = []
    print('Reading csv file: ' + csv_file)
    
    if os.path.exists(csv_file):
        try:
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    ref_slice = row[slice_column_index]
                    
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
                            
                            for sublist in __NON_REFSLICE_SHIFT:
                                position = sublist[axis_num]+corrected_position
                                non_refslice_attribs_np = get_attribs_from_dataframe(attribs_df, axis_num, position)
                                
                                ref_class_list.append(0)
                                image_id_list.append(image_id)
                                files_list.append(attribs_file)
                                slicenum_list.append(position)
                                attribs_list.append(non_refslice_attribs_np)
                                
                                                                
                            
                        
                    
                    
        except os.error:
            print("*** ERROR: CSV file %s can not be readed (os.error)" % csv_file)
        except UnicodeDecodeError:
            print('Error processing file ({0})'.format(csv_file))
    
    idl = pd.DataFrame(image_id_list,columns=['image_id'])
    scl = pd.DataFrame(slicenum_list,columns=['ref_slice'])
    afl = pd.DataFrame(files_list,columns=['attribs_file'])
    rcl = pd.DataFrame(ref_class_list,columns=['ref_class'])
    att = pd.DataFrame(attribs_list)
        
    #M_data.append((image_id,gender,age,alz_class,ref_class))
    
    return pd.concat([idl,scl,afl,rcl,att],axis=1)

def build_and_save_refslices_dataframe(csv_file=__INPUT_CSV, 
                           axis_num=0, 
                           attributes_dir=__ATTRIBUTES_DIR, 
                           o_file='/home/rodrigo/Documentos/_phd/git/ref-slices_attributes.csv'):
        
#                           ref_axis, 
#                           valid_genders=['M','F'], 
#                           min_age=0.0, 
#                           max_age=200.0, 
#                           debug=False, 
#                           black_list_id=[], 
#                           o_file='/home/rodrigo/Documentos/_phd/git/ref-slices_attributes.csv', 
#                           attribs_dir='../../attributes2'):
    #all_ref_attribs = []
    
    df = build_refslices_dataframe(csv_file,axis_num,attributes_dir)
    #dic = build_csv_dictionary2(input_csv)
    
    output_dir,filename = os.path.split(o_file)
    
    output_filename_without_extension, file_extension = os.path.splitext(filename)
    
    new_output_file = "{0}-axis{1}{2}".format(
            output_filename_without_extension, 
            axis_num,
            file_extension)
    
    output_file = os.path.join(output_dir, new_output_file)
    print('DF ready to save to destination file ',output_file)
    
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


#def get_slice_attribs(slicenum,axis=0,)

########
# FUNCAO MAIN
def main(argv):
    df = None
    refresh_ok = False
    bplanes = [0,1,2]
    global __COLUMN_OF_FIRST_SLICE_ATTRIBUTE
    global __REFSLICE_CSV__REFSLICE_CLASS_COLUMN    

    for plane in bplanes:
        if refresh_ok:
            df = build_and_save_refslices_dataframe(axis_num=plane)
        else:
            
            
        print(df.iloc[:9 , :6]) 
    


    
    
    
    
    
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
    


####df = build_refslices_dataframe()f

    

            

#target_id = ''

#all_files = []
#to_remove = []
#for row in df.head(df.size - 1).itertuples():
#    #print(row)
#    target_id = 'I' + str(row[1])
#    #print('target_id: ',target_id)
#    attribs_file = find_attributes_file(target_id)
#    
#    if not attribs_file:
#        to_remove.append(row[0])
#    else:
#        all_files.append(attribs_file)
#
#    #print('attribs_file: ',attribs_file)
#    
#    #attribs = pd.read_csv(attribs_file, delimiter=',',header=None)
#    #attribs.columns(names=['bp','slicenum'])
#    
#    #print( attribs.head(3))
#    #print('selected attribs:\n',attribs.loc[attribs['bp']==1])
#    #print(attribs)


#df = pd.concat([df,pd.DataFrame(all_files,columns=['attribs_filename'])],axis=1)   

#print(to_remove)






#print(data)

#data = get_data_frame_from_csv()
#print(data)
#remove_rows_with_blank_refslice(data)

#print(data)

