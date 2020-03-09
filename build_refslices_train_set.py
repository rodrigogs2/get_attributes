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

__INPUT_CSV = "./com_OFFSET_ADNI1_Complete_All_Yr_3T.csv"
__ATTRIBUTES_DIR = '../../attributes2'
__COLUMNS_TO_DROP = ['ADNI #','Year','Magnetic_Field','Subject','Visit','Modality','Description','GW','Type','Acq Date','Format','Downloaded','Date Yield','Format','Data upload']

                     
                     
def get_data_frame_from_csv(input_csv=__INPUT_CSV, columns_to_drop=__COLUMNS_TO_DROP):
    df = pd.read_csv(input_csv, delimiter=',')
    return df.drop(columns=columns_to_drop)

def get_attribs_from_dataframe(attribs_df, axis_num, slice_num):
    attribs = attribs_df.loc[(attribs_df[0]==axis_num) & (attribs_df[1]==slice_num)]
    return attribs.iloc[0,2:].values

def build_refslices_dataframe(csv_file=__INPUT_CSV, axis_num=0, attributes_dir=__ATTRIBUTES_DIR):
    global __ADJUST_REFSLICE_POSITION, __NON_REFSLICE_SHIFT
    image_id_column_index = 3
    slice_column_index = 19 + axis_num
    
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
    att = pd.DataFrame(attribs_list)
    rcl = pd.DataFrame(ref_class_list,columns=['ref_class'])
    
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



####df = build_refslices_dataframe()f
bplanes = [0,1,2]
for plane in bplanes:
    df = build_and_save_refslices_dataframe(axis_num=plane)
    print(df.iloc[:9 , :6])
    

            

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

