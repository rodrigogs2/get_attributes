#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 07:35:02 2018

@author: rodrigo
"""

import mahotas as mh


def zernike2(img_path, rad):
    return mh.features.zernike_moments(mh.imread(img_path, as_grey=True), rad)


def haralick2(img_path):
    return mh.features.haralick(mh.imread(img_path))


def get_attributes(img_data):
    r = mh.bbox(mh.imread(img_path))[1]/2
    return zernike(img_path,r).tolist() + haralick(img_path).flatten('K').tolist()


def zernike(img_path, rad):
    return mh.features.zernike_moments(mh.imread(img_path, as_grey=True), rad)


def haralick(img_path):
    return mh.features.haralick(mh.imread(img_path))


def attributes(img_path):
    r = mh.bbox(mh.imread(img_path))[1]/2
    return zernike(img_path,r).tolist() + haralick(img_path).flatten('K').tolist()


if __name__ == '__main__':
    nii_filename = '/home/rodrigo/Downloads/ADNI_136_S_0184_MR_MPR____N3__Scaled_Br_20090708094745554_S64785_I148265.nii'
    
    path = 'imgs/Cisto/Cisto_LEMD_1.jpg'

    # zernike_moments = zernike(path, (240))
    # print(zernike_moments)

    # haralick_features = haralick(path)
    # print(haralick_features.shape)

    print(len(attributes(path)))