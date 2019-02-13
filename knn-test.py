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

print('* all data list length=',len(attribs))
print('* an image attributes shape (retrived from all data)=',attribs[0].shape)

plane = 1
first_slice = 70
total_slices = 2
partition = loadattribs.get_attributes_partition(attribs,slice_amounts,plane,first_slice,total_slices)
print('* a partition data shape=',partition.shape)

new_dimensions = (partition.shape[0],partition.shape[1]*partition.shape[2])
#new_partition = np.reshape(partition, new_dimensions)
new_partition = np.reshape(partition, new_dimensions)
print('* DIMENSION for the new partition array=',new_dimensions)
print('* the new partition data shape=',new_partition.shape)
print('* the output array shape=',output_classes.shape)

print('* shape of an input instance retrived from the new partition=', new_partition[0].shape)


## KNN preparation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import pandas as pd


X_pandas = pd.DataFrame(data=new_partition)
print('X_pandas=\n',X_pandas)
y_pandas = pd.DataFrame(data=np.ravel(output_classes,order='C'))
print('y_pandas=\n',y_pandas)
print('unique(y)=',np.unique(y))



print('Counting classes instances=',output_classes)

X, y = SMOTE('auto').fit_resample(new_partition, output_classes)

#print(sorted(Counter(y_resampled).items()))
#clf_smote = LinearSVC().fit(X_resampled, y_resampled)
#
#X, y = ADASYN().fit_resample(X_pandas, y_pandas)
#print(sorted(Counter(y_resampled).items()))
#clf_adasyn = LinearSVC().fit(X_resampled, y_resampled)

#X = new_partition
#y = output_classes


# SMOTE Class Balancing
print('* Input data shape=',X.shape)
print('* Output values shape=',y.shape)


# STEP 1: split X and y into training and testing sets
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=12)

# print the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)

# print the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)

# STEP 2: train the model on the training set
logreg = LogisticRegression(max_iter=200,solver='lbfgs',multi_class='multinomial')
logreg.fit(X_train, y_train)

# STEP 3: make predictions on the testing set
y_pred = logreg.predict(X_test)

# compare actual response values (y_test) with predicted response values (y_pred)
print('Logistic Regression accuracy=',metrics.accuracy_score(y_test, y_pred))

paramk = 1
knn = KNeighborsClassifier(n_neighbors=paramk)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('KNN (k={0}) accuracy =\t{1}'.format(paramk, metrics.accuracy_score(y_test, y_pred)) )


paramk = 5
knn = KNeighborsClassifier(n_neighbors=paramk)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('KNN (k={0}) accuracy =\t{1}'.format(paramk, metrics.accuracy_score(y_test, y_pred)) )

paramk = 21
knn = KNeighborsClassifier(n_neighbors=paramk)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('KNN (k={0}) accuracy =\t{1}'.format(paramk, metrics.accuracy_score(y_test, y_pred)) )



'''
# Iris dataset load
iris = datasets.load_iris()
x_vals = np.array([x[0:4] for x in iris.data])
y_vals = np.array(iris.target)


# Bullshit
print('iris database (only x elements) shape=',x_vals.shape)
print('iris output set (y) shape=',y_vals.shape)


# changing values to alzheimer imagning data
x_vals = new_partition
y_vals = output_classes


# Esta linha cria uma matriz com 1 na diagonal principal (funcao eye) 
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

k = 15

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
'''