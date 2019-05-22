#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:53:17 2018

@author: rodrigo
"""

import os
import numpy as np

import nibabel as nb
import sys
#import mahotas as mh

import imageio as iio
import matplotlib.pyplot as plt



def check_slice_is_valid(img_data,slicenum,body_axis_num):
    isValid = False
    if (body_axis_num >= 0
        and body_axis_num <= 2 
        and slicenum >= 0 
        and slicenum < img_data.shape[body_axis_num]):
            isValid = True
    
    return isValid
    

def get_slice_data(img_data, slicenum=0, body_axis=0):
    img_2d_data = None
    
    if not check_slice_is_valid(img_data, slicenum, body_axis):
        # raise exception
        raise ValueError('*** ERROR: invalid image, slice and axis combination')
    else:
        #axis_size = img_data.shape[body_axis]
        #print('axis size: ',axis_size,'\t')
        if body_axis == 0:
            img_2d_data = img_data[slicenum, :, :]
            #print ('returned 2d img data shape: ', img_2d_data.shape)
        elif body_axis == 1:
            img_2d_data = img_data[:, slicenum, :]
            #print ('returned 2d img data shape: ', img_2d_data.shape)
        elif body_axis == 2:
            img_2d_data = img_data[:, :, slicenum]
            #print ('returned 2d img data shape: ', img_2d_data.shape)
    return img_2d_data


def save_image_2d_data_as_png_file(img_data, filename, output=None):
       
    # Normalization
    max_value = img_data.max()
    img_data = img_data / max_value
    img_data = img_data * 255;
    img_data_uint8 = img_data.astype(np.uint8)
    
    try:
        # Try to save image
        import imageio as iio
        iio.imwrite(filename,img_data_uint8)
    except os.error:
        raise ValueError("Fail to write image 2d data to png file (%s)",filename)
        
def build_png_filename_from_slice(input_full_filename, body_axis_num=0, slicenum=0, output_directory=None):
    # splitting file and directory from input full filename
    input_file_dir,input_filename = os.path.split(input_full_filename)

    # removing extetion from input file
    input_filename_without_extension = os.path.splitext(input_filename)[0]
    
    # putting png extension to output file
    output_filename = "%s-A%.3d-S%.3d.png" % (
            input_filename_without_extension, 
            body_axis_num, 
            slicenum)
    
    png_full_filename = os.path.join(output_directory, output_filename)
    
    # ENDED    
    return png_full_filename

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
    
    
def save_slice_as_png(nii_img_full_filename, slicenum, body_axis_num, output_directory, show_image=False):
    
    # build filename structure
    if not os.path.exists(nii_img_full_filename):
        raise ValueError("*** ERROR: input nii image filename not exist or can not be readed")
        exit(1)
    
    input_file_dir,input_filename = os.path.split(nii_img_full_filename)
    
    if output_directory == None:
        # Use input file dir as output when output dir is None
        output_directory = input_file_dir
    elif not os.path.exists(output_directory):
        # create the output dir whether it doesnt exist
        try:
            os.makedirs(output_directory)
        except os.error:
            print ('*** ERROR: Output directory (%s) can not be created\n' % output_directory)
            sys.exit(1)
    
    nii_img = nb.load(nii_img_full_filename)
    nii_img_data = nii_img.get_fdata()
    # Get a slice and show it
    
    img_slice = get_slice_data(nii_img_data, slicenum, body_axis_num)
    
    # Show image
    if show_image:
        plt.imshow(img_slice,cmap="gray",origin="lower")
        #plt.show(img_slice)
        plt.show()
    
    # Normalization
    max_value = img_slice.max()
    if max_value != 0:
        img_slice = img_slice / max_value
    img_slice = img_slice * 255;
    img_slice_uint8 = img_slice.astype(np.uint8)
    
    # saving png file
    png_full_filename = build_png_filename_from_slice(nii_img_full_filename,
                                                      body_axis_num,
                                                      slicenum,
                                                      output_directory)
    
    iio.imwrite(png_full_filename,img_slice_uint8)
    
    return png_full_filename


def append_attributes_to_file(
        nii_img_full_filename, 
        attributes, 
        axis_num, 
        slice_num, 
        output_directory, 
        mode="a",
        verbose=False,
        limit_precision=True):
    
    # build filename structure
    if not os.path.exists(nii_img_full_filename):
        raise ValueError("*** ERROR: input nii image filename not exist or can not be readed")
        exit(1)
    
    input_file_dir,input_filename = os.path.split(nii_img_full_filename)
    
    if output_directory == None:
        # Use input file dir as output when output dir is None
        output_directory = input_file_dir
    elif not os.path.exists(output_directory):
        # create the output dir whether it doesnt exist
        try:
            os.makedirs(output_directory)
        except os.error:
            print ('*** ERROR: Output directory (%s) can not be created\n' % output_directory)
            sys.exit(1)
    
    # building txt file name
    txt_full_filename = build_txt_filename_from_3d_image(nii_img_full_filename,
                                                      output_directory)
    try :
        output_file = open(txt_full_filename,mode)
        output_file.write('%d,%d,' % (axis_num, slice_num))
        for attrib in attributes:
            if limit_precision:
                output_file.write("{0:.8f}".format(attrib))
            else:
                output_file.write(attrib)
            output_file.write(',')
            
        output_file.write('\n')
        output_file.close()
    except os.error:
        output_file.close()
        print(" *** ERROR: file %s can not be written" % txt_full_filename)
        exit(1)
    
    return txt_full_filename


def remove_attributes_file(
        nii_img_full_filename, 
        output_directory):
    
    attributes_full_filename = build_txt_filename_from_3d_image(
            nii_img_full_filename,
            output_directory)
    
    os.remove(attributes_full_filename)



def extract_slices_as_png(nii_img_full_filename, 
                          output_directory, 
                          keep_png_cache_files=False, 
                          extract_attributes=False, 
                          verbose=False,
                          reset_output_file=True,
                          limit_precision=True):
    
    nii_img = nb.load(nii_img_full_filename)
    
    total_used_axis = 3
    
    # Testing if output file should be removed
    if reset_output_file:
        txt_file = build_txt_filename_from_3d_image(
                nii_img_full_filename,output_directory)
        if os.path.exists(txt_file):
            try:
                os.remove(txt_file)
            except os.error:
                raise ValueError("*** File %s already exists but can not be removed.", txt_file)
    
    
    # Main loop (core task)
    for axis_number in range(total_used_axis): # picking a axis
        total_slices = nii_img.shape[axis_number] # picking total slices for selected axis
        for slice_number in range(total_slices): # picking a valid slice
            
            saved_png_full_filename = save_slice_as_png(nii_img_full_filename, slice_number, axis_number, output_directory)
            
            # getting image attributes (zernick pm and haralick stats)
            attribs = get_attributes(saved_png_full_filename)
            
            # appending attrbiutes to txt file
            append_attributes_to_file(nii_img_full_filename, attribs, axis_number, slice_number, output_directory)
            
            # verbose print
            if verbose==True:
                print("Getting slice ", slice_number, "inside body axis " , axis_number)
                print("\nAttributes (slice=%s,axis=%s): " % (slice_number, axis_number))
                print(attribs)
            else:
                print('.')
            
            # Check whether cache files must be kept
            if not keep_png_cache_files:
                os.remove(saved_png_full_filename)
            
        
    
#def get_attributes(nii_image_full_filename, slice_num, axis_num, ):
        
    

#def show_img_data_as_grey_image(img_data):
#    # Show saved image file:
#    fig,axes = plt.subplot(1,img_data_uint8)
#    plt.imshow(img_data_uint8,cmap="gray") #,origin="lower")
#    plt.show(img_slice)
#    plt.show()

  

"""
# This function will use get_slice and save_pnf_file functions inside a loop
    # which will search 
def extract_all_slices_to_png_files(nii_filename,axis=None,output_path=None):
    axis = None
    img_data = None
    img = None
    try:
        # read MRI and get data
        img = nb.load(nii_filename)
        img_data = img.get_fdata()
        
        if axis == None: # We must use the three body axis
            axis = [0,1,2]
        
        if(output_path == None): # Output directory will be the same as inputfile directory
            output_dir = os.path.dirname(nii_filename)
            
        #print ('*** testfile_directory: ', testfile_directory)
        #print('does this path above exists?\n')
        
        exists = os.path.exists(output_dir)
        
        if exists == False: # We must create the output directory!
            try:
                os.makedirs(testfile_directory)
            except os.error:
                print('\n Ops! Something wrong has happened! Directory can not be created.\n')

        
        #total_slices = img_data.shape
            
        for axis_num in axis:
            input_filepath,input_filename = os.path.split(nii_filename)
            input_filename_without_extension = os.path.splitext(input_filename)[0]
            
            if axis_num >= 0 or axis_num <= 2:
                # if slice is valid...
                
                png_filename = build_png_filename_from_slice(input_filename_without_extension,axis_num,slice_num)
                png_full_filename = os.path.join(output_dir,png_filename)
                save_png_file(img_data,png_full_filename)
            else:
                print("Invalid Body Axis. Image shape is: ", img_data.shape)
        
        
        else:
            print("Axis is not null")
        
        print("extract_all_slices_to_png *** img_data.shape= ", img_data.shape)
    except os.error:
        # something wrong has happened
        sys.exit(1)
    return None
"""

# Temporary Variables (should be removed later)
img_data_shape = ''
input_full_filename = '/home/rodrigo/Downloads/ADNI_136_S_0184_MR_MPR____N3__Scaled_Br_20090708094745554_S64785_I148265.nii'
input_filepath,input_filename = os.path.split(input_full_filename)
input_filename_without_extension = os.path.splitext(input_filename)[0] 


fake_output_dir = '/home/rodrigo/Downloads/fake_output_dir2/'
built_png_filename = build_png_filename_from_slice(input_full_filename, 0, 80, fake_output_dir)

print  ('*** input_file_path=', input_filepath,
      '\n*** input_filename=', input_filename,
      '\n*** input filename without extension=', input_filename_without_extension,
      '\n*** output png filename=', built_png_filename)

save_slice_as_png(input_full_filename,130,1,fake_output_dir)

extract_slices_as_png(input_full_filename,fake_output_dir,verbose=True)

#img = nb.load(input_full_filename)
#img_data = img.get_fdata()
#print('input file data shape: x=',img_data.shape[0],' y=',img_data.shape[1], ' z=',img_data.shape[2])

#png_filename = png_filename + '.png'


# TESTE numero -1
# Testing area
"""
file = 'some_filename'
slicenum = 80
body_axis_num = 1

output_directory = '/home/rodrigo/Downloads/'
png_file = build_png_filename(file,slicenum,body_axis_num,output_directory)
print ('*** build_png_filename = ',png_file)

png_file = build_png_filename(file,slicenum,body_axis_num)
print ('*** build_png_filename = ',png_file)

testfile_fullpath = '/home/rodrigo/Downloads/teste-folder/temp.py'

# Selects dir path
testfile_directory = os.path.dirname(testfile_fullpath)

print ('*** testfile_directory: ', testfile_directory)
print('does this path above exists?\n')


exists = os.path.exists(testfile_directory)

if exists == True: 
    print('Gratz! This path exists!')
else:
    print('This path doesn''t exist.\nCreating path...')
    try:
        os.makedirs(testfile_directory)
        print(' Done!')
    except os.error:
        print('\n Ops! Something wrong has happened! Directory can not be created.\n')
"""    


#TESTE 0
"""

adni_file_data_path = input_full_filename
filename = input_filename
png_filename = os.path.join()

png_full_filename = os.path.join(adni_file_data_path, png_filename + '.png')

example_filename = os.path.join(data_path, filename)
"""

#img = nb.load(input_full_filename)
#adni_img = nb.load(adni_full_filename)
#img2 = nb.load(example_filename2)


#print('Forma da imagem de exemplo: ' + repr(img.shape))
#print('Forma da imagem adni: ' + repr(adni_img.shape))
# print("Teste nb.load: shape da imagem=",img.shape)



#TESTE 1
# Getting a slice and showing it
"""
#adni_img_data = adni_img.get_fdata()
adni_img = nb.load(input_full_filename)
adni_img_data = adni_img.get_fdata()

# Get a slice and show it
slice_num = 120
#img_slice = adni_img_data[slice_num,:,:]
img_slice = adni_img_data[slice_num,:,:] # Coronarie slicing

# 
plt.imshow(img_slice,cmap="gray",origin="lower")
#plt.show(img_slice)
plt.show()

# Normalization
max_value = img_slice.max()
img_slice = img_slice / max_value
img_slice = img_slice * 255;
img_slice_uint8 = img_slice.astype(np.uint8)


nii_filename = input_full_filename

#extract_all_slices_to_png(nii_filename)
"""



#TESTE 2
# Saving image
"""
import imageio as iio
iio.imwrite('teste.png',img_slice_uint8)


print('Forma da matriz de dados da imagem adni: ' + repr(adni_img_data.shape))
print('\nTipo da variavel img_slice: ' + repr(type(img_slice)))
print('png file full path: ' + repr(png_full_filename))


#png_img = Image.fromarray(img_slice)
#png_img.save("adni_png.jpeg")
#plt.imsave('adni.png',png_img)

print('Tipo de dados do array img_slice: ' + repr(type(img_slice)))

#img_slice = (img_slice, dtype=np.uint16)
##write_png('adni-slice.png',img_slice)

#png_img.save("/home/r8odrigo/Downloads/teste.png")



fig,axes = plt.subplot(1,n_slice)
axes[1].imshow(n_slice, cmap='grey', origin='lower')
plt.suptitle('N Slice')
plt.show()

"""