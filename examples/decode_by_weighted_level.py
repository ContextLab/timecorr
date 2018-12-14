# -*- coding: utf-8 -*-
"""
=============================
Decode by level
=============================

In this example, we load in some example data, and decode by level of higher order correlation.

"""
# Code source: Lucy Owen
# License: MIT

# load timecorr and other packages
import timecorr as tc
import hypertools as hyp
import numpy as np

# load helper functions
from timecorr.helpers import isfc, corrmean_combine
from scipy.optimize import minimize

#
# def objective(x, y):
#     x1 = x[0]
#     x2 = x[1]
#     x3 = x[2]
#     x4 = x[3]
#     return y * (x1*x4*(x1+x2+x3)+x3)
#
# def constraint1(x):
#     return x[0]*x[1]*x[2]*x[3]-25
#
# def constraint2(x):
#     sum_sqs=40
#     for i in range(4):
#         sum_sqs = sum_sqs - x[i]**2
#
#     return sum_sqs
#
# x0 = [1,5,5,1]
#
# print(objective(x0, 1))
# b = (1, 5)
# bns = (b,b,b,b)
#
# con1= {'type':'ineq', 'fun': constraint1}
# con2= {'type':'ineq', 'fun': constraint2}
#
# cons = [con1, con2]
#
# sol = minimize(objective, x0, args=1, method='SLSQP', bounds=bns, constraints=cons)

# load example data
data = hyp.load('weights').get_data()

# define your weights parameters
width = 10
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}

# set your number of levels
# if integer, returns decoding accuracy, error, and rank for specified level
level = 2

## optimize mu:


mu = [.5,.3,.2]

# run timecorr with specified functions for calculating correlations, as well as combining and reducing
results = tc.optimize_weighted_timepoint_decoder(np.array(data), mu=mu, level=level, combine=corrmean_combine,
                               cfun=isfc, rfun='eigenvector_centrality', weights_fun=laplace['weights'],
                               weights_params=laplace['params'])

# returns only decoding results for level 2
print(results)

