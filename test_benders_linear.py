# -*- coding: utf-8 -*-

from sets import Set
from scipy.optimize import linprog
import numpy as np
from benders_linear import benders_decomposition_linear_alternative, benders_decomposition_linear_original, benders_decomposition_linear_original_bounded
import sys
from plot_linear_inequalities import plot_linear_inequalities

def test_simplex(A, b, c):
  ''' Solves original problem with only one simplex. '''
  res = linprog(A_ub = A, b_ub = b, c= -np.array(c))
  print "Results of simplex:"
  print res

def print_results(x0, x, y, it, n):
  '''Print results of Benders decomposition.'''
  print "x0 = ", x0
  print "x = ", x
  print "y = ", y
  print "it = ", it
  print "n = ", n

def sizes(A,c):
  sa = A.shape[1]
  sa2 = A.shape[1] // 2
  sc = len(c)
  sc2 = len(c) // 2
  return (sa, sa2, sc, sc2)  

def test1(A, b, c, M):
  '''
  Test 1: Comparing results of simplex and benders_decomposition_linear_original_bounded. 
  '''
  (sa, sa2, sc, sc2) = sizes(A,c)
  
  # solves using benders decomposition 
  (x0, x, y, it, n) = benders_decomposition_linear_original_bounded(c[0:sc2], c[sc2:sc], A[:,0:sa2], A[:,sa2:sa], b, M)
  print "Results from benders_decomposition_linear_original_bounded:"
  print_results(x0, x, y, it, n)
  
  test_simplex(A, b, c)

def test2(A, b, c):
  '''
  Test 2: Comparing results of simplex and benders_decomposition_linear_original. 
  '''
  (sa, sa2, sc, sc2) = sizes(A,c)

  # solves using benders decomposition 
  (x0, x, y, it, n) = benders_decomposition_linear_original(c[0:sc2], c[sc2:sc], A[:,0:sa2], A[:,sa2:sa], b)
  print "Results from benders_decomposition_linear_original:"
  print_results(x0, x, y, it, n)

  test_simplex(A, b, c)

def test3(A, b, c, M):
  '''
  Test 3: Comparing results of simplex and benders_decomposition_linear_alternative. 
  '''
  
  (sa, sa2, sc, sc2) = sizes(A,c)
  
  # solves using benders decomposition 
  (x0, x, y, it, n) = benders_decomposition_linear_alternative(c[0:sc2], c[sc2:sc], A[:,0:sa2], A[:,sa2:sa], b, M)
  print "Results from benders_decomposition_linear_alternative:"
  print_results(x0, x, y, it, n)
  
  test_simplex(A, b, c)


M = 10000000000.0

c =  [8., 6]
A = np.matrix([[2., 1.] ])
b = [4.]

test1(A,b,c,M)
test2(A,b,c)
test3(A,b,c,M)

c =  [8, 6, -2, -42,-18,-33]
A = np.matrix([[2, 1, -1, -10, -8,  0],
              [1, 1,  1,  -5,  0, -8]])
b = [-4,-3]

test1(A,b,c,M)
test2(A,b,c)
test3(A,b,c,M)

c =  np.array([4,0])
A = np.matrix([[-2, 0], [-1, 0]])
b = [-8, -6]

test1(A,b,c,M)
test2(A,b,c)
test3(A,b,c,M)
