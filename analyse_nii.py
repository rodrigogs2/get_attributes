#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:52:36 2019

@author: rodrigo
"""
import list_dir as ld
import loadattribs as la
import nibabel as nb
import numpy as np
import os

def print_header(nii_file):
    print('Woohoo!')

black_list_id = ['I288905','I288906','I120446','I120441','I120426','I120436','I120423','I120416']

file1 = 'ADNI_002_S_1070_MR_MPR____N3__Scaled_Br_20070217034203890_S24206_I40840.nii'
file2 = 'ADNI_002_S_1070_MR_MPR____N3__Scaled_Br_20071227094032049_S43530_I86231.nii'
file3 = 'ADNI_002_S_1070_MR_MPR____N3__Scaled_Br_20081014173827443_S55382_I120784.nii'
file4 = 'ADNI_002_S_1070_MR_MPR____N3__Scaled_Br_20081224110557894_S60783_I132215.nii'

nii_dir = '../../Nii_files'

use_only_my_list = False
my_list = [file1, file2, file3, file4]

all_files_with_duplicates = ld.new_list_files(nii_dir,extension='.nii')
all_unique_files = []

included_filenames = []
for full_filename in all_files_with_duplicates:
    directory, file = os.path.split(full_filename)
    try:
        position = included_filenames.index(file)
    except ValueError:
        included_filenames.append(file)
        all_unique_files.append(full_filename)
 

all_files_array = np.array(all_unique_files)
print(all_files_array)

print('* Total files found = ',all_files_array.size)
print('* Use only my list = ',use_only_my_list)



all_qox = []
all_qoy = []
all_qoz = []

for nii_img_full_filename in all_files_array:
    directory, file = os.path.split(nii_img_full_filename)
    position = -1
    
    image_id = la.get_image_ID(file)
    is_black_listed = la.is_black_listed(image_id, black_list_id)
    
    #nii_img = nb.load(nii_img_full_filename)
    if is_black_listed:
        print('current image id ({0}) is black listed. skipping this 3d image'.format(image_id))
        continue
    
    if use_only_my_list:
        try:
            position = my_list.index(file)
            if position >= 0:
                #print('file={0} in in the list!'.format(file))
                my_list.pop(position)
                continue # skips for loop current iteration
                
#                nii_img = nb.load(nii_img_full_filename)
#    
#                qox = nii_img.header['qoffset_x']
#                qoy = nii_img.header['qoffset_y']
#                qoz = nii_img.header['qoffset_z']
#        
#                all_qox.append(qox)
#                all_qoy.append(qoy)
#                all_qoz.append(qoz)
                

                
                #nii_img_data = nii_img.get_fdata()
                
        except ValueError:
            pass
    
    
    nii_img = nb.load(nii_img_full_filename)
    
    #            #print('* HEADER data:')
    #            print('nii_img_header:\n', nii_img.header)
    #            print('offset_x=',nii_img.header['qoffset_x'])
    #            print('offset_y=',nii_img.header['qoffset_y'])
    #            print('offset_z=',nii_img.header['qoffset_z'])
    #            print('\n')

    qox = nii_img.header['qoffset_x']
    qoy = nii_img.header['qoffset_y']
    qoz = nii_img.header['qoffset_z']

    all_qox.append(qox)
    all_qoy.append(qoy)
    all_qoz.append(qoz)

print('* total number of headers analysed: ',len(all_qox))
            
array_qox = np.array(all_qox)
array_qoy = np.array(all_qoy)
array_qoz = np.array(all_qoz)

print('min qoX=',array_qox.min())
print('max qoX=',array_qox.max())

print('min qoY=',array_qoy.min())
print('max qoY=',array_qoy.max())

print('min qoZ=',array_qoz.min())
print('max qoZ=',array_qoz.max())
      
        

