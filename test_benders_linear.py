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
d = [6.]
A = np.matrix([[2.]])
B = np.matrix([[1.]])
b = [4.]

test1(A,B,b,c,d,M)
test2(A,B,b,c,d)
test3(A,B,b,c,d,M)

c =  [8., 6, -2]
d =  [-42.,-18,-33]
A = np.matrix([[2, 1, -1.],
               [1, 1.,  1]])  
B = np.matrix([[-10., -8,  0],
               [-5,  0., -8.]])
b = [-4,-3.]

test1(A,B,b,c,d,M)
test2(A,B,b,c,d)
test3(A,B,b,c,d,M)

c = [-4.]
d = [0.]
A = np.matrix([[-2.], 
               [-1.]])
B = np.matrix([[1.], 
               [4.]])
b = [-8., -6.]

test1(A,B,b,c,d,M)
test2(A,B,b,c,d)
test3(A,B,b,c,d,M)

c1 = [868., 207, 702, 999, 119]

rho1 =  [[ 401,   74,    9,  741,  744],
      [ 959,  412,  625,  671,  721],
      [ 520.,  205,  912,  114,  438],
      [1000,   81,   28,   45,  206]]

pi1 =  [[716,  66, 432, 543, 981],
      [ 87, 638., 147, 255, 378],
      [650, 363,  42, 447, 898],
      [211, 104., 145, 975,   6]]

rho = np.matrix(rho1)

pi = np.matrix(pi1)

c = rho[:,4].T.tolist()[0]
d = pi[:,4].T.tolist()[0]
A = rho.T
B = pi.T
b = c1

test1(A,B,b,c,d,M)

rho = [[908,965,583,45,883,455,184,816,398,198]
,[57,589,935,141,191,410,889,419,458,813]
,[947,500,454,354,254,113,661,469,297,583]
,[983,855,951,857,984,664,632,547,370,954]
,[218,230,268,550,31,67,893,418,764,646]
,[615,259,21,664,482,577,375,440,411,962]
,[419,92,595,428,383,189,158,552,713,302]
,[696,205,27,336,768,186,328,460,513,539]
,[707,753,934,782,766,205,793,756,478,916]]

pi = [[983,823,652,720,354,497,514,296,135,6]
,[548,934,534,472,802,262,587,889,497,678]
,[77,148,913,273,596,399,455,991,74,73]
,[44,687,696,451,861,554,336,17,743,381]
,[605,511,162,568,709,638,412,512,249,796]
,[586,10,308,581,15,857,473,325,394,695]
,[618,586,567,66,668,713,89,58,291,457]
,[132,157,358,858,478,949,223,954,216,338]
,[679,170,332,187,187,473,80,402,226,810]]

b= [66,334,745,595,409,854,465,91,380,893]

rho= np.matrix(rho)

pi = np.matrix(pi)

c = rho[:,-1].T.tolist()[0]
d = pi[:,-1].T.tolist()[0]
A = rho.T
B = pi.T

test1(A,B,b,c,d,M)

rho = [[908,965,583,45,883,455,184,816,398,198]
,[57,589,935,141,191,410,889,419,458,813]
,[947,500,454,354,254,113,661,469,297,583]
,[983,855,951,857,984,664,632,547,370,954]
,[218,230,268,550,31,67,893,418,764,646]
,[615,259,21,664,482,577,375,440,411,962]
,[419,92,595,428,383,189,158,552,713,302]
,[696,205,27,336,768,186,328,460,513,539]
,[707,753,934,782,766,205,793,756,478,916]]

pi = [[983,823,652,720,354,497,514,296,135,6]
,[548,934,534,472,802,262,587,889,497,678]
,[77,148,913,273,596,399,455,991,74,73]
,[44,687,696,451,861,554,336,17,743,381]
,[605,511,162,568,709,638,412,512,249,796]
,[586,10,308,581,15,857,473,325,394,695]
,[618,586,567,66,668,713,89,58,291,457]
,[132,157,358,858,478,949,223,954,216,338]
,[679,170,332,187,187,473,80,402,226,810]]

b = [66,334,745,595,409,854,465,91,380,893]

rho= np.matrix(rho)

pi = np.matrix(pi)

d = rho[:,-1].T.tolist()[0]
c = pi[:,-1].T.tolist()[0]
B = rho.T
A = pi.T

test1(A,B,b,c,d,M)

