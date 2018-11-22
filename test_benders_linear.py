# -*- coding: utf-8 -*-

from sets import Set
from scipy.optimize import linprog
import numpy as np
from benders_linear import benders_decomposition_linear_alternative, benders_decomposition_linear_original, benders_decomposition_linear_original_bounded
import sys
from plot_linear_inequalities import plot_linear_inequalities

def test_simplex(A, B, b, c, d):
  ''' 
  Solves the problem 
  
    max cx + dy
    s.t.: Ax + By <= b
  
  with only one simplex. 
  
  This problem is equivalent to
  
    max cx + dy
    s.t.: |A B| |x| <= |b|
                |y|
  '''
  AB = np.concatenate((A, B),axis=1)
  cd = np.concatenate((c, d),axis=0)
  res = linprog(A_ub = AB, b_ub = b, c= -np.array(cd))
  print "Results of simplex:"
  print res

def print_results(x0, x, y, it, n):
  '''Print results of Benders decomposition.'''
  print "x0 = ", x0
  print "x = ", x
  print "y = ", y
  print "it = ", it
  print "n = ", n

def test1(A, B, b, c, d, M):
  '''
  Test 1: Comparing results of simplex and benders_decomposition_linear_original_bounded. 
  '''
  # solves using benders decomposition 
  (x0, x, y, it, n) = benders_decomposition_linear_original_bounded(c, d, A, B, b, M)
  print "Results from benders_decomposition_linear_original_bounded:"
  print_results(x0, x, y, it, n)
  test_simplex(A, B, b, c, d)

def test2(A, B, b, c, d):
  '''
  Test 2: Comparing results of simplex and benders_decomposition_linear_original. 
  '''
  # solves using benders decomposition 
  (x0, x, y, it, n) = benders_decomposition_linear_original(c, d, A, B, b)
  print "Results from benders_decomposition_linear_original:"
  print_results(x0, x, y, it, n)
  test_simplex(A, B, b, c, d)

def test3(A, B, b, c, d, M):
  '''
  Test 3: Comparing results of simplex and benders_decomposition_linear_alternative. 
  '''
  # solves using benders decomposition 
  (x0, x, y, it, n) = benders_decomposition_linear_alternative(c, d, A, B, b, M)
  print "Results from benders_decomposition_linear_alternative:"
  print_results(x0, x, y, it, n)
  test_simplex(A, B, b, c, d)


M = 10000000000.0

c = [8.]
d = [6]
A = np.matrix([[2.]])
B = np.matrix([[1.]])
b = [4.]

test1(A,B,b,c,d,M)
test2(A,B,b,c,d)
test3(A,B,b,c,d,M)

c =  [8, 6, -2]
d =  [-42,-18,-33]
A = np.matrix([[2, 1, -1.],
               [1, 1.,  1]])  
B = np.matrix([[-10, -8,  0],
               [-5,  0, -8]])
b = [-4,-3]

test1(A,B,b,c,d,M)
test2(A,B,b,c,d)
test3(A,B,b,c,d,M)

c = [-4]
d = [0.]
A = np.matrix([[-2], 
               [-1]])
B = np.matrix([[1.], 
               [4.]])
b = [-8, -6]

test1(A,B,b,c,d,M)
test2(A,B,b,c,d)
test3(A,B,b,c,d,M)



