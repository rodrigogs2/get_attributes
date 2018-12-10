#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:21:13 2018

@author: rodrigo
"""

import os

def list_files(input_dir,extention=".nii"):
    returned_files = []
    dir_files = os.listdir(input_dir)
    for filename in dir_files:
        name,ext = os.path.splitext(filename)
        if ext == extention:
            full_filename = os.path.join(input_dir,filename)
            #print(full_filename)
            if os.path.isfile(full_filename):
                returned_files.append(full_filename)
        
    return returned_files

i_dir = "/home/rodrigo/Downloads/"
print("\n .nii files inside dir: %s" % list_files(i_dir))