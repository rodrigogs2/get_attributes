#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:33:03 2018

@author: rodrigo
"""
import multiprocessing
from functools import partial
"""

#from multiprocessing import Pool

def f(x,y=0):
    quad = x*x + y
    print (quad)
    return quad

def f(x)

if __name__ == '__main__':
    cores_num = multiprocessing.cpu_count()
    with multiprocessing.Pool(cores_num) as p:
        print(p.map(f, [1,2,3]))
    #    print(p.map(f, [1, 2, 3]))
        print("Total Number of cores: ", cores_num)
        
"""

def multi_run_wrapper(args):
   return add(*args)
def add(x,y=1,z=10):
    return x+y+z


if __name__ == "__main__":
    from multiprocessing import Pool
    pool = Pool(4)
    results = pool.map(partial(add,y=2,z=8),[1,2,3])
    print (results)

"""    
if __name__ == "__main__":
    from multiprocessing import Pool
    pool = Pool(4)
    results = pool.map(multi_run_wrapper,[(1,2),(2,3),(3,4)])
    print (results)
"""