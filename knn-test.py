#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 08:30:02 2019

@author: rodrigo
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
#import os, cvs, sys
import list_dir
import loadattribs


# csv description file


# Use this arguments to set the input directory of attributes files
attributes_dir = "/home/rodrigo/Downloads/fake_output_dir2/"

'''
# Getting all files names
attribs_files = list_dir.list_files(attributes_dir,".txt")

# Checking how many files were found
print("\n* %s attribs files were loaded." % len(attribs_files))

# Variable used to check memory usage to load all attributes from all files
total_memory_usage = 0

# Use these to control how many files will be loaded
first_file = 0
total_printed_files = len(attribs_files)

# Loop which loads attributes, checks their shape and gets memory usage
print("* Loaded Files:")
for n in range(first_file, total_printed_files):
    file = attribs_files[n]
    memory_size = 0
    attribs,body_plane,slice_num,slice_amount = loadattribs.load_attribs_and_metadata(file)
    memory_size = attribs.size * attribs.itemsize + body_plane.size * body_plane.itemsize + slice_num.size * slice_num.itemsize
    total_memory_usage += memory_size
    print('\t-Attributes from the {0}th data file :\n{1}'.format(n,attribs))
    print('\t-Body planes from the {0}th data file :\n{1}'.format(n,body_plane))
    print('\t-Slice numbers from the {0}th data file :\n{1}'.format(n,slice_num))
    print('\t-Dimensions of each vector: attribs({0}), body_plane({1}) slices_num({2}) and slice_amount({3})'.format(attribs.ndim, body_plane.ndim, slice_num.ndim, slice_amount.ndim))
    print('\t-Slices amounts of {0}th data file: {1}'.format(n,slice_amount))
    print('\t-Memory size usage to load the {0}th data file: {1} bytes'.format(n,memory_size))
    

print('-Total memory usage to load all the {0} data files is:\n\t\t{1} bytes'.format(total_printed_files, total_memory_usage))

#attribs,body_plane,slice_num = loadattribs.load_attribs_and_metadata(attribs_files[0])
'''


csv_file = '/home/rodrigo/Documents/_phd/csv_files/ADNI1_Complete_All_Yr_3T.csv' 
# load data
attribs, body_axis, slice_numbers, slice_amounts, output_classes = loadattribs.load_all_data(attributes_dir, csv_file)

print('Shape: ',attribs.shape)

iris = datasets.load_iris()
x_vals = np.array([x[0:4] for x in iris.data])
y_vals = np.array(iris.target)



#iris = some_partition

# one hot encoding
y_vals = np.eye(len(set(y_vals)))[y_vals]

# normalize
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

# train-test split
np.random.seed(59)
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices =np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


#KNN

feature_number = len(x_vals_train[0])

k = 5

x_data_train = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)
y_data_train = tf.placeholder(shape=[None, len(y_vals[0])], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)

# manhattan distance
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), axis=2)

# nearest k points
_, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
top_k_label = tf.gather(y_data_train, top_k_indices)

sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
prediction = tf.argmax(sum_up_predictions, axis=1)

# Training

sess = tf.Session()
prediction_outcome = sess.run(prediction, feed_dict={x_data_train: x_vals_train,
                               x_data_test: x_vals_test,
                               y_data_train: y_vals_train})

# evaluation
accuracy = 0
for pred, actual in zip(prediction_outcome, y_vals_test):
    if pred == np.argmax(actual):
        accuracy += 1

print("final accuracy:", accuracy / len(prediction_outcome))