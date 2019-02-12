#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Updates:
# 
# 2019, Jan 17: a função de carregar os dados de todos os arquivos de atributos
# foi modificada para retornar vetores do tipo np.array. Modificação foi 
# testada e está funcionando corretamente

"""
Created on Wed Dec 12 11:23:05 2018

@author: rodrigo
"""

import os, csv, sys 
import numpy as np
import list_dir

def build_txt_filename_from_3d_image(input_full_filename, output_directory=None):
    # splitting file and directory from input full filename
    input_file_dir,input_filename = os.path.split(input_full_filename)

    # removing extetion from input file
    input_filename_without_extension = os.path.splitext(input_filename)[0]
    
    # putting txt extension to output file
    output_filename = "%s.txt" % input_filename_without_extension
    
    txt_full_filename = os.path.join(output_directory, output_filename)
    
    # ENDED    
    return txt_full_filename


def load_attribs_and_metadata(attributes_file):
    body_plane_list = []
    slicenum_list = []
    attributes_list = []
    output_class= []
    
    if os.path.exists(attributes_file):
        try:
            with open(attributes_file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    attributes_list.append(row[2:])
                    body_plane_list.append(row[0])
                    slicenum_list.append(row[1])
                    
        except os.error:
            print("*** ERROR: Attributes file %s can not be readed (os.error in load_attribs function)" % attributes_file)
    
    attribs_as_floats_lists = []
    print("*** Processing attributes file %s)" % attributes_file)
    for attribs_as_string in attributes_list:
        #print("Attribs as strings: ", attribs_as_string)
        a = []
        for str_attrib in attribs_as_string:
            #print('String attribute: ', str_attrib)
            if str_attrib != '':
                try:
                    value = float(str_attrib)
                except ValueError:
                    print('*** ERROR: Fail to convert an string attribute ("{0}") to float in load_attribs_and_metadata().\n Attributes File: {1}'.format(str_attrib, attributes_file))
                    sys.exit(-1)
                
                a.append(value)
                    
        attribs_as_floats_lists.append(a)
    
    # Counting slices from each body axis
    plane0 = len(body_plane_list) - body_plane_list[::-1].index('0')
    plane1 = len(body_plane_list) - body_plane_list[::-1].index('1') - plane0
    plane2 = len(body_plane_list) - body_plane_list[::-1].index('2') - plane0 - plane1
    slice_amount_per_plane = [plane0,plane1,plane2]
    
    # NumPy transformations
    attribs = np.array(attribs_as_floats_lists, dtype=np.float64)
    body_plane = np.array(body_plane_list, dtype=np.int64)
    slice_numbers = np.array(slicenum_list, dtype=np.int64)
    slice_amount = np.array(slice_amount_per_plane, dtype=np.int64)
    
    return  attribs, body_plane, slice_numbers, slice_amount


def load_reshaped_attribs_and_metadata(attributes_file):
    attribs,body_plane,slice_numbers,slice_amount = load_attribs_and_metadata(attributes_file)
    reshaped = np.reshape(attribs,-1)
    return reshaped,slice_amount

def get_attributes_from_a_slice(image_attribs, 
                         slice_amounts,
                         specific_body_plane, 
                         specific_slice_num):

    index = specific_slice_num

    if specific_body_plane != 0:
        plane1_start = slice_amounts[0]
        if specific_body_plane == 1:
            index = index + plane1_start
        else:
            plane2_start = plane1_start + slice_amounts[1]
            index = index + plane2_start
    
    return image_attribs[index]
    
    '''
    if specific_body_plane == 0 and :
        if  plane0_start <= initial_slice_num < end_slice_num and end_slice_num < plane0_end:
                print('It is valid!')
        start_index = initial_slice_num
        end_index = end_slice_num
    elif specific_body_plane == 1:
        start_index = initial_slice_num
        end_index = end_slice_num
    elif specific_body_plane == 2:
        start_index = initial_slice_num
        end_index = end_slice_num
    else:
        raise(ValueError('*** ERROR: Invalid body plane ("{0}") in get_attributes_partition().\n'.format(specific_body_plane)))
        sys.exit(-1)
    '''

def get_attributes_from_a_range_of_slices(image_attribs,
                                        slice_amounts,
                                        specific_body_plane,
                                        start_slice,
                                        slices_range):
    attributes_list = []
    for s in range(start_slice, start_slice + slices_range):
        attributes_list.append(get_attributes_from_a_slice(image_attribs,slice_amounts,specific_body_plane,s))
    
    return np.array(attributes_list, dtype=np.float64)

'''
def get_slices_limits(all_slice_amounts):
    plane0_min, plane1_min, plane2_min = 0,0,0
    for slice_amounts in all_slice_amounts:
        if slice_amounts[0] < plane0_min: 
            plane0_min = slice_amounts[0]
        if slice_amounts[1] < plane1_min:
            plane1_min = slice_amounts[1]
        if slice_amounts[2] < plane2_min:
            plane2_min = slice_amounts[2]
    return plane0_min, plane1_min, plane2_min


def get_attributes_partition( image_attribs, 
                         body_planes, 
                         slice_numbers,
                         slice_amounts,
                         specific_body_plane, 
                         initial_slice_num, 
                         end_slice_num=None):
    plane0_start = 0
    plane0_end = slice_amounts[0]
    plane1_start = slice_amounts[0]
    plane1_end = plane1_start + slice_amounts[1]
    plane2_start = plane1_end
    plane2_end = plane2_start + slice_amounts[2]
    
    
    return 0
'''


def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]

    print ('Usage:\n    ', script_name, '[Options] -i <inputfile> -o <outputdir> ')
    print ('  Options:')
    print('\t-m, --multicpu\tset on computation over all cores (default: multicore is off)')
    print('\t-v, --verbose\tenables verbose mode (default: disabled)')
    print('\t-r, --resume\tresume extraction: output files are not overwritten (default: resume is off)')

        
def main(argv):
    ## Testting area
    
    # Use this arguments to set the input directory of attributes files
    attributes_dir = "/home/rodrigo/Downloads/fake_output_dir2/"
    
    # Getting all files
    attribs_files = list_dir.list_files(attributes_dir,".txt")
    
    # Checking how many files were found
    print("\n* %s attribs files were loaded." % len(attribs_files))

    # Used to check memory usage to load all attributes from all files
    total_memory_usage = 0
    
    # Use these to control how many files will be loaded
    first_file = 0
    total_printed_files = len(attribs_files)
    total_printed_files = 1    
    
    # Loop which loads attributes, checks their shape and gets memory usage
    print("* Loaded Files:")
    for n in range(first_file, total_printed_files):
        file = attribs_files[n]
        memory_size = 0
        attribs,body_plane,slice_num,slice_amounts = load_attribs_and_metadata(file)
        memory_size = attribs.size * attribs.itemsize + body_plane.size * body_plane.itemsize + slice_num.size * slice_num.itemsize
        total_memory_usage += memory_size
        
        print('\t-Attributes from the {0}th data file :\n{1}'.format(n,attribs))
        print('\t-Body planes from the {0}th data file :\n{1}'.format(n,body_plane))
        print('\t-Slice numbers from the {0}th data file :\n{1}'.format(n,slice_num))
        print('\t-Dimensions of each vector: attribs({0}), body_plane({1}) and slices_num({2})'.format(attribs.ndim, body_plane.ndim, slice_num.ndim))
        print('\t-Amount of slices per each body plane of the {0}th data file: {1}'.format(n,slice_amounts))
        print('\t-Memory size usage to load the {0}th data file: {1} bytes'.format(n,memory_size))
        print('\t-Shape of Attributes from the {0}th data file :\n{1}'.format(n,attribs.shape))
    
    print('\t-Total memory usage to load all the {0} data files is:\n\t\t{1} bytes'.format(total_printed_files, total_memory_usage))
    
    f = 53
    attribs,body_plane,slice_num,slice_amounts = load_attribs_and_metadata(attribs_files[f])
    p = 0  # plane of human body (can be 0, 1 or 2)
    fs = 80 # first slice of interval
    sr = 15 # last slice of interval 
    
    # getting partition from 80 to 100th slice from the f-th (f=0) attribs file
    partition = get_attributes_from_a_range_of_slices(attribs,slice_amounts,p,fs,sr)
    print('\n-Shape of Attributes Partition of the {0}th data file from the slices {2} and {3}: {1}'.format(f,partition.shape,fs,sr))
    
    first = fs
    for at in partition:
        print('\n-Attributes from {1}th slice of the {2}th attribs file:\n{0}'.format(at,first,f))
        first = first + 1
    
    print('\nPartition:\n{0}'.format(partition))
    
if __name__ == "__main__":    
    main(sys.argv)
