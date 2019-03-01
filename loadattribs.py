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

import os, csv, sys, re
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
    #print("*** Processing attributes file %s)" % attributes_file)
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

def build_classes_dictionary(csv_file):
    alzheimer_dic = {'CN': 0, 'MCI': 1, 'AD': 2}
    dic = {}

    if os.path.exists(csv_file):
        try:
            with open(csv_file, 'r') as file:
                #print('CSV File received: ', csv_file)
                reader = csv.reader(file)
                headers = next(reader) 
                for row in reader:
                    image_id = 'I' + row[3]
                    image_class = alzheimer_dic[row[5]]
                    dic[image_id] = image_class
        except os.error:
            print("*** ERROR: The csv file %s can not be readed (os.error in build_classes_dictionary)" % csv_file)    

    else:
        message = str("file %s does not exist!" % csv_file)
        raise ValueError(message)
    return dic

def get_class(attributes_file, all_classes_dictionary):
    image_id = re.findall(r'I[0-9]+',attributes_file)
    image_class = None
    
    if len(image_id) > 0:
        image_class = all_classes_dictionary[image_id[0]]
    return image_class


def load_all_data(attributes_dir, csv_file):
    all_attribs = []
    all_body_planes = []
    all_slice_num = []
    all_slice_amounts = []
    all_classes = []
    
    dic = build_classes_dictionary(csv_file)
    
    # Getting all attributes files from attributes directory
    attribs_files = list_dir.list_files(attributes_dir,".txt")
    
    # Loop which loads attributes, classes values and slicing info
    for file in attribs_files:
        attribs,body_plane,slice_num,slice_amounts = load_attribs_and_metadata(file)
        all_attribs.append(attribs)
        all_body_planes.append(body_plane)
        all_slice_num.append(slice_num)
        all_slice_amounts.append(slice_amounts)  
        image_class = get_class(file,dic)
        all_classes.append(image_class)
        
    array_all_classes = np.array(all_classes, dtype=np.int64)
    
    return all_attribs, all_body_planes, all_slice_num, all_slice_amounts, array_all_classes

    # getting partition from 80 to 100th slice from the f-th (f=0) attribs file
    #partition = get_attributes_from_a_range_of_slices(attribs,slice_amounts,p,fs,ls)
    #print('\n-Shape of Attributes Partition of the {0}th data file from the slices {2} and {3}: {1}'.format(f,partition.shape,fs,ls))
    
    #first = fs
    #for at in partition:
    #    print('\n-Attributes from {1}th slice of the {2}th attribs file:\n{0}'.format(at,first,f))
    #    first = first + 1
    
    #print('\nPartition:\n{0}'.format(partition))


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
                                        total_slices):
    attributes_list = []
    for s in range(start_slice, start_slice + total_slices):
        attributes_list.append(get_attributes_from_a_slice(image_attribs,
                                                           slice_amounts,
                                                           specific_body_plane,
                                                           s))
    
    return np.array(attributes_list, dtype=np.float64)


def get_slices_limits(all_slice_amounts):
    min_values = [0,0,0]
    
    for slice_amount in all_slice_amounts:
        for i in range(3):
            if slice_amount[i] > min_values[i]: 
                min_values[i] = slice_amount[i]
    return min_values


def get_attributes_partition(all_attribs, 
                         all_slice_amounts,
                         specific_body_plane, 
                         initial_slice_num, 
                         total_slices=1):
    attribs_partition = []
    for attribs,s_amount in zip(all_attribs,all_slice_amounts):
        attribs_partition.append(get_attributes_from_a_range_of_slices(attribs,
                                                                       s_amount,
                                                                       specific_body_plane,
                                                                       initial_slice_num,
                                                                       total_slices))
    
#    plane0_start = 0
#    plane0_end = slice_amounts[0]
#    plane1_start = slice_amounts[0]
#    plane1_end = plane1_start + slice_amounts[1]
#    plane2_start = plane1_end
#    plane2_end = plane2_start + slice_amounts[2]
    
    return np.array(attribs_partition, dtype=np.float64)



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
    csv_file = '/home/rodrigo/Documents/_phd/csv_files/ADNI1_Complete_All_Yr_3T.csv' 
    # Getting all files
    
    attribs, body_planes, slice_num, slice_amounts, output_classes = load_all_data(attributes_dir, csv_file)
    
    bplane = 2
    start_slice = 123
    total_slices = 25
    
    croped_attribs = get_attributes_from_a_range_of_slices(attribs[0], 
                                                           slice_amounts[0], 
                                                           bplane, 
                                                           start_slice, 
                                                           total_slices)
    
    print('shape of croped_attribs: ',croped_attribs.shape)
    print('cropped attribs:\n',croped_attribs)
    
    '''
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
    
    f = 0
    attribs,body_plane,slice_num,slice_amounts = load_attribs_and_metadata(attribs_files[f])
    p = 0  # plane of human body (can be 0, 1 or 2)
    fs = 80 # first slice of interval
    ls = 100 # last slice of interval 
    
    # getting partition from 80 to 100th slice from the f-th (f=0) attribs file
    partition = get_attributes_from_a_range_of_slices(attribs,slice_amounts,p,fs,ls)
    print('\n-Shape of Attributes Partition of the {0}th data file from the slices {2} and {3}: {1}'.format(f,partition.shape,fs,ls))
    
    first = fs
    for at in partition:
        print('\n-Attributes from {1}th slice of the {2}th attribs file:\n{0}'.format(at,first,f))
        first = first + 1
    
    print('\nPartition:\n{0}'.format(partition))
    '''
    
if __name__ == "__main__":    
    main(sys.argv)
