#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:53:17 2018

@author: rodrigo
"""

import os
import numpy as np
#from numpngw import write_png
import nibabel as nb
import matplotlib.pyplot as plt
import matplotlib.image as im
from nibabel.testing import data_path
from PIL import Image

#filename = 'example4d.nii.gz'

#'ADNI_002_S_0413_MR_MPR____N3__Scaled_2_Br_20081124142730828_S22683_I128346.nii'
adni_filename = 'ADNI_136_S_0184_MR_MPR____N3__Scaled_Br_20090708094745554_S64785_I148265.nii'

adni_file_data_path = '/home/rodrigo/Documentos/_phd/Nii_files'

example_full_filename = os.path.join(data_path, filename)
adni_full_filename = os.path.join(adni_file_data_path, adni_filename)

png_filename,png_file_ext = os.path.split(adni_full_filename)
#png_filename = png_filename + '.png'
png_full_filename = os.path.join(adni_file_data_path, png_filename + '.png')

#example_filename2 = os.path.join(data_path, filename2)

img = nb.load(example_full_filename)
adni_img = nb.load(adni_full_filename)
#img2 = nb.load(example_filename2)

print('Forma da imagem de exemplo: ' + repr(img.shape))
print('Forma da imagem adni: ' + repr(adni_img.shape))
#print(img2.shape)


#adni_img_data = adni_img.get_fdata()
adni_img_data = adni_img.get_fdata()


# Get a slice and show it
slice_num = 120
#img_slice = adni_img_data[slice_num,:,:]
img_slice = adni_img_data[slice_num,:,:] # Coronarie

# 
plt.imshow(img_slice,cmap="gray",origin="lower")
#plt.show(img_slice)
plt.show()

# Normalization
max_value = img_slice.max()
img_slice = img_slice / max_value
img_slice = img_slice * 255;
img_slice_uint8 = img_slice.astype(np.uint8)



# Saving image
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


"""fig,axes = plt.subplot(1,n_slice)
axes[1].imshow(n_slice, cmap='grey', origin='lower')
plt.suptitle('N Slice')
plt.show()"""