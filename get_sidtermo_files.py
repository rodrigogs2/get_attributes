#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:06:21 2019

@author: rodrigo
"""

import os, getopt, re, sys
import loadattribs

__TOTAL_BODY_PLANES = 3 #Axial, Sargital and Coronal


def get_body_plane_and_slice_nums(png_filename):
    body_plane = -1
    slice_num= -1
    all_body_planes_founded = re.findall(r'Axis[0-9]+',png_filename) # returns a array with all regular exp matches
    if len(all_body_planes_founded) > 0:
        body_plane = int(re.findall(r'[0-9]{1,3}',all_body_planes_founded[0])[0])
    
    
    all_slices_founded = re.findall(r'Slice[0-9]+',png_filename) # returns a array with all regular exp matches
    if len(all_slices_founded) > 0:
        slice_num = int(re.findall(r'[0-9]{1,3}',all_slices_founded[0])[0])
    
    if body_plane > -1 and slice_num > -1:
        return body_plane,slice_num
    else:
        return ''



def list_all_image_files(input_dir, body_plane, min_slice, max_slice, extension=".png"):
    returned_files = []
    
    for root, dirs, files in os.walk(input_dir,topdown=False):
        for filename in files:
            name,ext = os.path.splitext(filename)
            
            axis_num, slice_num = get_body_plane_and_slice_nums(filename)
            
            slice_is_ok = axis_num == body_plane and slice_num >= min_slice and slice_num < max_slice
            
            if ext == extension and slice_is_ok:
                full_filename = os.path.join(root,filename)
                #print(full_filename)
                if os.path.isfile(full_filename):
                    returned_files.append(full_filename)        
    return returned_files


#def get_class_from_filename(filename,csv_file):
    
    


#def copy_selected_slices(input_dir,body_plane,min_slice,max_slice, extension=".png"):
#    selected_files = list_all_files(input_dir,body_plane,min_slice,max_slice, extension)
    


def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]

    print('This tool builts a directory structure compatible with sid-termo tool using png files from specific slices extracted from ADNI nii volumes')
    print('Usage:\n    ', script_name, '[Options] -i <inputdir> -o <outputdir> -c <csv_file> -b <body_plane> -f <first_slice> -t <total_slices>')
    print('  Options:')
    #print('\t-m, --multicpu\tset on computation over all cores (default: multicore is off)')
    print('\t-v, --verbose\tenables verbose mode (default: disabled)')
    print('\t-h, --help\tdisplays this help screen')
    #print('\t-r, --resume\tresume extraction: output files are not overwritten (default: resume is off)')
    #print('\t-k, --keep_png\ttemporary png files extracted from MRI volume slices are not deleted (default: keep_png is False)')

def main(argv):
    
    inputdir = ''
    outputdir = ''
    csv_file = ''
    body_plane = 0
    first_slice = 0
    total_slices = 1
    idir_ok = False
    odir_ok = False
    csv_file_ok = False
    verbose_ok = False
    body_plane_ok = False
    first_slice_ok = False
    total_slices_ok = False
    slicing_ok = False
    #multi_cpu_ok = False
    #reset_txt_file_ok = True
    #keep_temporary_png = False
    
    try:
        #opts, args = getopt.getopt(argv[1:],"hi:o:vmrk",["idir=","odir=","verbose","multicpu","reset_output_file","keep_png"]) 
        opts, args = getopt.getopt(argv[1:],"hc:i:o:vb:f:t:",["csv_file=","idir=","odir=","verbose","body_plane=","first_slice=","total_slices="])
    except getopt.GetoptError:
        display_help()
        sys.exit(1)
    
    for opt, arg in opts:
        if opt == '-h':
            display_help()
            sys.exit(0);
        elif opt in ("-c", "--csv-file"):
            csv_file = arg
            csv_file_ok = True
        elif opt in ("-i", "--idir"):
            inputdir = arg
            idir_ok = True
        elif opt in ("-o", "--odir"):
            outputdir = arg
            odir_ok = True
        elif opt in ("-v", "--verbose"):
            verbose_ok = True
        elif opt in ("-b","--body_plane"):
            global __TOTAL_BODY_PLANES
            body_plane = int(arg) % __TOTAL_BODY_PLANES
            body_plane_ok = True
        elif opt in ("-f", "--first_slice"):
            first_slice = int(arg)
            if first_slice >= 0 and first_slice <= 255:
                first_slice_ok = True
        elif opt in ("-t", "--total_slices"):
            total_slices = int(arg)
            if total_slices > 0 and total_slices <= 255:
                total_slices_ok = True
        
        if total_slices_ok and first_slice_ok and body_plane_ok:
            if first_slice + total_slices - 1 <= 255:
                slicing_ok = True
        
        #elif opt in ("-m", "--multicpu"):
        #    multi_cpu_ok = True
        #elif opt in ("-r", "--resume"):
        #    reset_txt_file_ok = False
        #elif opt in ("-k","--keep_png"):
        #    keep_temporary_png = True
    
    if idir_ok and odir_ok and csv_file_ok and slicing_ok:
        
        if verbose_ok:
            print ('Output directory is: ', outputdir)
            print ('Input file is: ', inputdir)
            print ('ADNI CSV File is: ', csv_file)
            print ('First Slice is: ', first_slice)
            print ('Total slices is: ', total_slices)
            print ('Last slice is: ', first_slice + total_slices - 1)
            print ('\nEverything looks fine... lets go!')
        
        image_id_dic = loadattribs.build_cvs_dictionary(csv_file)
        
        all_image_filenames = list_all_image_files(inputdir, body_plane, first_slice, first_slice + total_slices, extension=".png")
        
        all_classes = []
        for filename in all_image_filenames:
            demographics_data = loadattribs.get_image_demographic_data(filename,image_id_dic)
            image_class = demographics_data['class']
            all_classes.append(image_class)
        
        unique_class_found = set(all_classes)
        all_dir_names = []
        
        for class_id in unique_class_found:
            dir_name = str(outputdir) + '/class_' + str(class_id)
            all_dir_names.append(dir_name)

        if verbose_ok:
            
            
            for dir_name in all_dir_names:
                print('New subdir: ', dir_name)
            
            print ('Total 2D slices found: ', len(all_image_filenames))
            print ('All classes found: ', unique_class_found)
            print('Class from all slices found: ', all_classes )
        
        for index in range(len(all_image_filenames)):
            filename = all_image_filenames[index]
            image_class = all_classes[index]
            
            target_dir = all_dir_names[int(image_class)]
            print('Target dir: ', target_dir)
            
            
        
        #loadattribs.load_all_data(attribs_dir, csv_file)
        #extract_attributes(input_path=inputdir,
        #                   keep_png_cache_files=keep_temporary_png,
        #                   output_directory=outputdir,
        #                   verbose=verbose_ok,
        #                   multi_cpu=multi_cpu_ok,
        #                   reset_output_file=reset_txt_file_ok,
        #                   )
        
    else:
        display_help()
    
if __name__ == "__main__":    
    main(sys.argv)
    
