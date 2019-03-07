# -*- coding: utf-8 -*-
"""
This module contains two implementations of classic Benders' procedure
for linear functions f and F. It implements the procedures described in
sections 4 and 5 of paper 
    
    J. F. Benders. Particioning procedures for solving mixed-variables
    programming problems. In: Numerische Matematik, 4. 1962. p. 238-252
"""

'''---------------------------------------------------------------------
Use monospace font and editors without replacing spaces features to see
comments properly.
---------------------------------------------------------------------'''

import operator
from sets import Set
from scipy.optimize import linprog
import numpy as np
import math 

#extreme direction when linprog returns unbounded solution
extremeDirection = 0

#the "d-row", i.e., the zi - ci row in the optimum simplex tableau.
d_row = 0


def extremeRay_and_dRow(xk, **kwargs):
  """
  A callback function for linprog to return extreme ray when solution is 
  unbounded. Also, when linprog terminates, returns the "d-row", i.e., 
  the zi - ci row in the optimum simplex tableau. 
  
  ** It overwrites global variables extremeDirection and d_row. **
  
  Parameters:
    xk : array_like
        The current solution vector.
    **kwargs : dict
        A dictionary containing the following parameters:
  
        tableau : array_like
            The current tableau of the simplex algorithm.
            Its structure is defined in _solve_simplex.
        phase : int
            The current Phase of the simplex algorithm (1 or 2)
        nit : int
            The current iteration number.
        pivot : tuple(int, int)
            The index of the tableau selected as the next pivot,
            or nan if no pivot exists
        basis : array(int)
            A list of the current basic variables.
            Each element contains the name of a basic variable and its 
            value.
        complete : bool
            True if the simplex algorithm has completed
            (and this is the final call to callback), otherwise False.
  """
  complete = kwargs["complete"]
  if complete:
    tableau = kwargs["tableau"]
    global d_row 
    d_row = tableau[-1,:]
    pivrow, pivcol = kwargs["pivot"]
    if math.isnan(pivrow) and not math.isnan(pivcol):
      nit = kwargs["nit"]
      phase = kwargs["phase"]
      basis = kwargs["basis"]
      tableauSize = tableau.shape
      m = tableauSize[0]-1
      n = tableauSize[1]-1
      k = pivcol
      yk = np.matrix(tableau[:-1,k]).T 
      # N is list of  non-basic variables and
      # B is basis variables set:
      #         N = {0, 1, ..., n+m-1} / B
      N = list(set(range(0,n+m)).difference(set(basis)))
      # Let yk be a vector in ℝ^m. 
      # Extreme ray is [-yk, 0, 0, ..., 1, ..., 0] with 1 in k'-th 
      # position,  where k' is the index of entering variable k. 
      # See Bazaraa et al, Linear Programming and Network Flows, 
      # Fourth Edition, p.118
      global extremeDirection
      e = np.zeros((n,1))
      e[N.index(k)]=1;
      extremeDirection = np.concatenate((-yk,e))

def chooseSubsetZero(A, B, c, d):
  '''
  Choose a set with vector zero as finite subset of 
                    T
       C = {(u ,u)|A  u − cu  ≥ 0,u ≥ 0,u ≥ 0}
              0             0            0
                       m
  with u ∈ ℝ, and u ∈ ℝ .
        0
  
  Arguments
    A: matrix m x p (lines x columns);
    B: matrix m x q;
    c: vector with p elements;
    d: vector with q elements;
  Return
    A finite subset of C. 
  '''
  (m,p) = A.shape
  return set( [(0.0,)*(m + 1)] );

def chooseSubsetu0_1(A, B, c, d):
  '''
  Choose a set with one vector as finite subset of 
                    T
       C = {(u ,u)|A  u − cu  ≥ 0,u ≥ 0,u ≥ 0}
              0             0            0
                       m
  with u  = 1, and u ∈ ℝ .
        0
  
  Arguments
    A: matrix m x p (lines x columns);
    B: matrix m x q;
    c: vector with p elements;
    d: vector with q elements;
  Return
    A finite subset of C. 
  '''
  (m,p) = A.shape
  res = linprog(c=(0,)*m, A_ub = -A.T, b_ub = -c)
  if res.status == 0: 
    vec = np.append(1.0,res.x)
    return set([tuple(vec)])
  else: 
    return set( [(0.0,)*(m + 1)] )

def linearSubproblem(Q, B, d, b):
  '''
  Solves maximization subproblem in first part of iterative step in 
  original Bender's procedure, i.e,
         ν   ν                           ν
       (x , y ) =  max{x  | (x , y) ∈ G(Q )}, 
         0              0     0
  with
                                          T                  T
      G(Q) =    ∩     {(x  ,y) | u  x  + u F(y) - u  f(y) ≤ u b, y ∈ S},
           (u , u) ∈ Q   0        0  0             0
             0                                                    T
  with F and f linear functions, such that F(y) = By, and f(y) = d y.
  Arguments
    Q: a Set data structure representing a subset of C with 
                   m+1
       vectors in ℝ ;
    B: matrix  m x q;
    d: vector with q elements;
    b: vector with m elements.
  Return
                                    q
    x0, y: a value and a vector in ℝ  that solves problem, if exists. 
           otherwise, inf is returned in x0 and y.
    iter: number of iterations to solve this subproblem
  '''
  '''---------------------------------------------------------------
  As we have:
                                        T           T     T        q
    G(Q) =    ∩     {(x  ,y) | u  x  + u B y  - u  d y ≤ u b, y ∈ ℝ },
         (u , u) ∈ Q   0        0  0             0
           0
                                       ⎛  T        T⎞      T        q
         =    ∩     {(x  ,y) | u  x  + ⎜ u B - u  d ⎟ y ≤ u b, y ∈ ℝ },
         (u , u) ∈ Q   0        0  0   ⎝        0   ⎠
           0
  we need to solve a linear maximization problem:
              T 
      max.  c1 (x ,y)
                 0
  s.t.:             ⎛  T        T⎞      T
            u  x  + ⎜ u B - u  d ⎟ y ≤ u b
             01 0   ⎝  1     01  ⎠      1
                    ⎛  T        T⎞      T
            u  x  + ⎜ u B - u  d ⎟ y ≤ u b
             02 0   ⎝  2     02  ⎠      2
                           .
                           .
                           .
                    ⎛  T        T⎞      T
            u  x  + ⎜ u B - u  d ⎟ y ≤ u b
             0s 0   ⎝  s     0s  ⎠      s
                    q                             T   q+1
  with x  ∈ ℝ, y ∈ ℝ, |Q| = s, c1 = (1, 0, ..., 0) ∈ ℝ,   and 
        0                       
  ⎛                  ⎞T   
  ⎜u  , u  , ..., u  ⎟ = u  for i in {1, ..., s}.
  ⎝ 0i   1i        qi⎠    i

  However, linear programming in scipy uses minimization problem. So
  we have to negate vector c1, resulting

                                          T   q+1
                      c1 = (-1, 0, ..., 0) ∈ ℝ

  In order to construct b1 and A1 from G(Q), we see that:
                           ⎛ T    T         T ⎞T
                      b1 = ⎜u b, u b, ..., u b⎟, and
                           ⎝ 1    2         s ⎠

                        ⎡                      ⎤
                        ⎢       ⎛  T        T⎞ ⎥
                        ⎢ u     ⎜ u B - u  d ⎟ ⎥
                        ⎢  01   ⎝  1     01  ⎠ ⎥
                        ⎢                      ⎥
                        ⎢       ⎛  T        T⎞ ⎥
                        ⎢ u     ⎜ u B - u  d ⎟ ⎥
                        ⎢  02   ⎝  2     02  ⎠ ⎥
                  A1 =  ⎢                      ⎥
                        ⎢          .           ⎥
                        ⎢          .           ⎥
                        ⎢          .           ⎥
                        ⎢                      ⎥
                        ⎢       ⎛  T        T⎞ ⎥
                        ⎢ u     ⎜ u B - u  d ⎟ ⎥
                        ⎢  0s   ⎝  s     0s  ⎠ ⎥
                        ⎣                      ⎦
  ---------------------------------------------------------------'''
  q = d.size
  c1 = (-1.0,)+(0.0,)*q;
  b1 = [np.dot(u[1:],b) for u in Q]
  A1 = np.matrix([ [u[0]] + (np.matmul(np.matrix(u[1:]),B)
       - u[0]*d.T).tolist()[0] for u in Q])
  res = linprog(c=c1, A_ub=A1, b_ub=b1)
  it = res.nit;
  if res.status == 0:
    x0 = res.x[0]; y = res.x[1:]
    return (x0, y, it)
  else:
    return ( float('inf'), (float('inf'),)*q, it)


def linearSubproblemBounded(Q, B, d, b, M):
  '''
  Solves maximization subproblem in first part of iterative step in 
  original Bender's procedure, i.e,
         ν   ν                           ν
       (x , y ) =  max{x  | (x , y) ∈ G(Q )}, 
         0              0     0
  with
                                          T                  T
      G(Q) =    ∩     {(x  ,y) | u  x  + u F(y) - u  f(y) ≤ u b, y ∈ S},
           (u , u) ∈ Q   0        0  0             0
             0                                                    T
  with F and f linear functions, such that F(y) = By, and f(y) = d y.
                               q
  In this procedure, S = [0, M] .
  Arguments
    Q: a Set data structure representing a subset of C with 
                   m+1
       vectors in ℝ ;
    B: matrix  m x q;
    d: vector with q elements;
    b: vector with m elements.
    M: a big real value.
  Return
                                    q
    x0, y: a value and a vector in ℝ  that solves problem, if exists. 
           otherwise, inf is returned in x0 and y.
    iter: number of iterations to solve this subproblem
  '''
  '''---------------------------------------------------------------
  As we have:
                                        T           T     T        q
    G(Q) =    ∩     {(x  ,y) | u  x  + u B y  - u  d y ≤ u b, y ∈ ℝ },
         (u , u) ∈ Q   0        0  0             0
           0
                                       ⎛  T        T⎞      T        q
         =    ∩     {(x  ,y) | u  x  + ⎜ u B - u  d ⎟ y ≤ u b, y ∈ ℝ },
         (u , u) ∈ Q   0        0  0   ⎝        0   ⎠
           0
  we need to solve a linear maximization problem:
              T 
      max.  c1 (x ,y)
                 0
  s.t.:             ⎛  T        T⎞      T
            u  x  + ⎜ u B - u  d ⎟ y ≤ u b
             01 0   ⎝  1     01  ⎠      1
                    ⎛  T        T⎞      T
            u  x  + ⎜ u B - u  d ⎟ y ≤ u b
             02 0   ⎝  2     02  ⎠      2
                           .
                           .
                           .
                    ⎛  T        T⎞      T
            u  x  + ⎜ u B - u  d ⎟ y ≤ u b
             0s 0   ⎝  s     0s  ⎠      s
                    q                             T   q+1
  with x  ∈ ℝ, y ∈ ℝ, |Q| = s, c1 = (1, 0, ..., 0) ∈ ℝ,   and 
        0                       
  ⎛                  ⎞T   
  ⎜u  , u  , ..., u  ⎟ = u  for i in {1, ..., s}.
  ⎝ 0i   1i        qi⎠    i

  However, linear programming in scipy uses minimization problem. So
  we have to negate vector c1, resulting

                                          T   q+1
                      c1 = (-1, 0, ..., 0) ∈ ℝ

  In order to construct b1 and A1 from G(Q), we see that:
                           ⎛ T    T         T ⎞T
                      b1 = ⎜u b, u b, ..., u b⎟, and
                           ⎝ 1    2         s ⎠

                        ⎡                      ⎤
                        ⎢       ⎛  T        T⎞ ⎥
                        ⎢ u     ⎜ u B - u  d ⎟ ⎥
                        ⎢  01   ⎝  1     01  ⎠ ⎥
                        ⎢                      ⎥
                        ⎢       ⎛  T        T⎞ ⎥
                        ⎢ u     ⎜ u B - u  d ⎟ ⎥
                        ⎢  02   ⎝  2     02  ⎠ ⎥
                  A1 =  ⎢                      ⎥
                        ⎢          .           ⎥
                        ⎢          .           ⎥
                        ⎢          .           ⎥
                        ⎢                      ⎥
                        ⎢       ⎛  T        T⎞ ⎥
                        ⎢ u     ⎜ u B - u  d ⎟ ⎥
                        ⎢  0s   ⎝  s     0s  ⎠ ⎥
                        ⎣                      ⎦
  ---------------------------------------------------------------'''
  q = d.size
  c1 = (-1.0,)+(0.0,)*q;
  b1 = [np.dot(u[1:],b) for u in Q]
  A1 = np.matrix([ [u[0]] + (np.matmul(np.matrix(u[1:]),B)
        - u[0]*d.T).tolist()[0] for u in Q])
  res = linprog(c=c1, A_ub=A1, b_ub=b1, bounds=(0,M))
  it = res.nit;
  if res.status == 0:
    x0 = res.x[0]; y = res.x[1:]
    return (x0, y, it)
  else:
    return ( float('inf'), (float('inf'),)*q, it)

def chooseAnyPointG(Q, B, d, b):
  '''
  Selects an arbitrary point in G(Q) using linear programming.
  Arguments
    Q: a Set data structure representing a subset of C with 
                   m+1
       vectors in ℝ ;
    B: matrix  m x q;
    d: vector with q elements;
    b: vector with m elements.
  Return
    y: an arbitrary vector with q elements, such that (x ,y) ∈ G(Q).
                                                        0
    it: number of iterations that linprog uses for finding y. 
  '''
  '''-------------------------------------------------------------------
  As we have:
                                        T           T     T        q
    G(Q) =    ∩     {(x  ,y) | u  x  + u B y  - u  d y ≤ u b, y ∈ ℝ },
         (u , u) ∈ Q   0        0  0             0
           0
                                       ⎛  T        T⎞      T        q
         =    ∩     {(x  ,y) | u  x  + ⎜ u B - u  d ⎟ y ≤ u b, y ∈ ℝ },
         (u , u) ∈ Q   0        0  0   ⎝        0   ⎠
           0
  we need to solve a linear maximization problem:
              T 
      max.   0 (x ,y)
                 0
  s.t.:             ⎛  T        T⎞      T
            u  x  + ⎜ u B - u  d ⎟ y ≤ u b
             01 0   ⎝  1     01  ⎠      1
                    ⎛  T        T⎞      T
            u  x  + ⎜ u B - u  d ⎟ y ≤ u b
             02 0   ⎝  2     02  ⎠      2
                           .
                           .
                           .
                    ⎛  T        T⎞      T
            u  x  + ⎜ u B - u  d ⎟ y ≤ u b
             0s 0   ⎝  s     0s  ⎠      s
                    q         
  with x  ∈ ℝ, y ∈ ℝ, |Q| = s, and 
        0                       
  ⎛                  ⎞T   
  ⎜u  , u  , ..., u  ⎟ = u  for i in {1, ..., s}.
  ⎝ 0i   1i        qi⎠    i

  In order to construct b1 and A1 from G(Q), we see that:
                           ⎛ T    T         T ⎞T
                      b1 = ⎜u b, u b, ..., u b⎟, and
                           ⎝ 1    2         s ⎠

                        ⎡                      ⎤
                        ⎢       ⎛  T        T⎞ ⎥
                        ⎢ u     ⎜ u B - u  d ⎟ ⎥
                        ⎢  01   ⎝  1     01  ⎠ ⎥
                        ⎢                      ⎥
                        ⎢       ⎛  T        T⎞ ⎥
                        ⎢ u     ⎜ u B - u  d ⎟ ⎥
                        ⎢  02   ⎝  2     02  ⎠ ⎥
                  A1 =  ⎢                      ⎥
                        ⎢          .           ⎥
                        ⎢          .           ⎥
                        ⎢          .           ⎥
                        ⎢                      ⎥
                        ⎢       ⎛  T        T⎞ ⎥
                        ⎢ u     ⎜ u B - u  d ⎟ ⎥
                        ⎢  0s   ⎝  s     0s  ⎠ ⎥
                        ⎣                      ⎦
  ---------------------------------------------------------------'''
  q = d.size
  c1 = (0.0,)*(q+1);
  b1 = [np.dot(u[1:],b) for u in Q]
  A1 = np.matrix([ [u[0]] + (np.matmul(np.matrix(u[1:]),B)
       - u[0]*d.T).tolist()[0] for u in Q])
  res = linprog(c=c1, A_ub=A1, b_ub=b1)
  it = res.nit;
  if res.status == 0:
    y = res.x[1:]
    return  (y, it)
  else:
    print "There was an error finding a point in G(Q)."
    return ((0,)*q, it)



def benders_decomposition_linear_original(c, d, A, B, b,
                                          chooseSubsetC = chooseSubsetZero):
  '''
  Original Benders' procedure for linear functions. This procedure 
  solves the problem
  
       T                            p     
  max{c x + f(y)|Ax + F(y) ≤ b,x ∈ ℝ ,y ∈ S}
            q         T
  with S = ℝ, f(y) = d y, and F(y) = By,
  
  where
         p       q       m×p       m×q       m
    c ∈ ℝ , d ∈ ℝ , A ∈ ℝ   , B ∈ ℝ   , b ∈ ℝ .
  
  Arguments:
    c: vector with p elements;
    d: vector with q elements;
    A: matrix m x p;
    B: matrix m x q;
    b: vector with m elements;
    chooseSubsetC: a function that takes a m x p matrix A, a p elements 
      vector c and return a set of vectors with m+1 elements. 
  Return:
    (max, x, y, it, n)
      max: solution of problem, if exists. A message is delivered if 
        problem has no solution and inf is returned.
      x, y: vectors with p and q elements respectively that solves the 
        problem, if solution exists. inf are returned otherwise.
      it: sum of iterations of subproblems
      n: number of iterations in main loop
  '''
  (m_A,p_A) = A.shape
  c = np.array(c)
  b = np.array(b)
  d = np.array(d)
  p_c = c.size
  q_d = d.size
  m_b = b.size
  (m_B,q_B) = B.shape
  if (p_c != p_A or m_b != m_A or q_d != q_B or m_B != m_b):
    print("Incompatible sizes of c, d, A, B and b.")
    return (float('inf'), float('inf'), float('inf'))
  else:
    m, p, q = m_b, p_c, q_d
  it = 0
  '''-------------------------------------------------------------------
  Original algorithm from
    J. F. Benders. Particioning procedures for solving mixed-variables
    programming problems. In: Numerische Matematik, 4. 1962. p. 238-252:

                      m
  1) Let u  ∈ ℝ, u ∈ ℝ  and  set:
          0
                  T
     C = {(u ,u)|A  u − cu  ≥ 0,u ≥ 0,u ≥ 0}
            0             0            0
                                           T                  T
     G(Q) =      ∩     {(x  ,y) | u  x  + u F(y) - u  f(y) ≤ u b, y ∈ S}
           (u , u) ∈ Q    0        0  0             0
             0
              T
     P =  {u|A u ≥ c, u ≥ 0}

              T
     C  = {u|A u ≥ 0, u ≥ 0}
      0
  -------------------------------------------------------------------'''

  '''-------------------------------------------------------------------
  2) Set Q ⊂ C
  -------------------------------------------------------------------'''
  Q = chooseSubsetC(A, B, c, d)
  '''-------------------------------------------------------------------
                                                 0
  3) if u  > 0 for at least one point (u , u) ∈ Q  then
         0                              0        

       3.1) goto 4, the first part of iterative step
                                              0
     else if  u  = 0 for any point (u , u) ∈ Q  then
               0                     0
             0
       3.2) x  = +∞
             0

             0                                    0
       3.3) y  = u, for any point of (u , u) ∈ G(Q )
                                       0
              Remark: In Benders original paper, he says 
               0                   0
              y  = any point of G(Q ).
                            0   q+1
              In this way, y ∈ ℝ.   However, in 4), we see that
                           ν                       q
              (x , y) ∈ G(Q ). Then, y must be in ℝ.
                0
       3.4) goto 6, the second part of iterative step
  -------------------------------------------------------------------'''
  goto_first_part = goto_second_part = False
  if reduce(operator.or_, [x[0] > 0 for x in Q]):
    goto_first_part = True
  else:
    x0 = float('inf')
    (y, it2) = chooseAnyPointG(Q, B, d, b)
    it = it + it2
    goto_second_part = True
  n = 0
  new_x0 = None
  new_y = [None]
  while True:
    '''-----------------------------------------------------------------
     4) first part of ν-th iteration: Solve
         ν   ν                           ν
       (x , y ) =  max{x  | (x , y) ∈ G(Q )}
         0              0     0
    -----------------------------------------------------------------'''
    if goto_first_part:
      (x0, y, ite) = linearSubproblem(Q, B, d, b)
      it = it + ite
      previous_x0 = new_x0
      previous_y = new_y
      new_x0 = x0
      new_y = y
      '''---------------------------------------------------------------
      5) if problem in 4 is not feasible then

           5.1) terminate and warn that problem is infeasible

         else

           5.2) go to 6, the second part of iterative step
      ---------------------------------------------------------------'''
      if np.isinf(x0):
        print "Problem is infeasible."
        return ( float('inf'), (float('inf'),)*p, 
                 (float('inf'),)*q, it, n)
      goto_second_part = True
    '''-----------------------------------------------------------------
    6) second part of ν-th iteration: Solve
        ν               ν  T      T
       u  = min{(b - F(y ))  u | A u ≥ c, u ≥ 0}
    -----------------------------------------------------------------'''
    if goto_second_part:
      c1 = b - (B.dot(y)).tolist()[0]
      res = linprog(c = c1, A_ub = (-np.matrix(A)).T,
                    b_ub = -c, callback = extremeRay_and_dRow,
                    options={"bland": True})
      u, it = res.x, it + res.nit
      '''---------------------------------------------------------------
      7) if problem in 6 is not feasible then

           7.1) terminate and warn that problem is infeasible
                         ν  T  ν    ν      ν
         else if (b - F(y ))  u  = x  - f(y )
                                    0

           7.2) Solve dual of 2nd part problem, i.e.,
            ν           T                 ν
           x  = max {( c x | A x ≤ b - F(y ), x ≥ 0}


                         ν   ν  ν
           7.3) return (x , x, y )
                         0
                         ν  T  ν    ν      ν
         else if (b - F(y ))  u  < x  - f(y )
                                    0
                 ν+1    ν         ν
           7.4) Q    = Q  ∪ {(1, u )}

           7.5) ν = ν + 1

           7.6) go to 4
      ---------------------------------------------------------------'''
      if res.status == 2 or res.status == 1:
        print "Problem is infeasible."
        return ( float('inf'), (float('inf'),)*p, 
                 (float('inf'),)*q, it, n)
      else:
        value1 = np.dot(c1, u)
        value2 = x0 - np.dot(d,y)
        if res.status == 0:
          if value1 == value2:
            x = d_row[c.size:-1]
            return (x0, x, y, it, n)
          elif value1 < value2:
            Q = Q.union([(1,)+tuple(u)])
          else:
            print "It is not possible (b - F(y ))^T u  > x  - f(y )."
            print "Something is wrong in step 7 of algorithm."
            return ( float('inf'), (float('inf'),)*p, 
                   (float('inf'),)*q, it, n)
          '''-----------------------------------------------------------
           8) if objective function in problem in 6 tends to infinity 
               along the halfline 
                           ν     ν
                 {u | u = u  + λv , λ ≥ 0}
                 
                    ν         ν
              with u ∈ P and v  the direction of an extreme halfline of
              C  then
               0
                              ν  T  ν     ν      ν
              8.1) if (b - F(y ))  u  ≥  x  - f(y ) then
                                          0
                             ν+1    ν         ν
                     8.1.1) Q    = Q  ∪ {(0, v )}
                                       ν  T  ν     ν      ν
                   else, i.e., (b - F(y ))  u  <  x  - f(y )
                                                   0
                             ν+1    ν         ν        ν
                     8.1.2) Q    = Q  ∪ {(1, u ), (0, v )}
              
              8.2) ν = ν + 1
              
              8.3) go to 4
          -----------------------------------------------------------'''
        else: #i.e., res.status == 3, or, problem in 6 is unbounded
          v = extremeDirection[0:len(u),0].T.tolist()[0]
          if value1 >= value2:
            Q = Q.union([(0.0,)+tuple(v)])
          else:
            Q = Q.union([(1.0,)+tuple(u),(0.0,)+tuple(v)])
    n = n + 1
    ''' Avoiding infinite loop. '''
    if(goto_first_part and (previous_x0 == new_x0) and np.array_equal(previous_y, new_y)):
      return (x0, x, y, it, n)
    goto_first_part = True


def benders_decomposition_linear_original_bounded(c, d, A, B, b, M, 
                                                  chooseSubsetC = chooseSubsetZero):
  '''
  Original Benders' procedure for linear functions with subproblems
  working with values bounded in [0, M]. This procedure solves 
  the problem
  
       T                            p     
  max{c x + f(y)|Ax + F(y) ≤ b,x ∈ ℝ ,y ∈ S}
                q         T
  with S = [0,M], f(y) = d y, and F(y) = By,
  
  where
         p       q       m×p       m×q       m
    c ∈ ℝ , d ∈ ℝ , A ∈ ℝ   , B ∈ ℝ   , b ∈ ℝ .
  
  Arguments:
    c: vector with p elements;
    d: vector with q elements;
    A: matrix m x p;
    B: matrix m x q;
    b: vector with m elements;
    M: a big real value; 
    chooseSubsetC: a function that takes a m x p matrix A, a p elements 
      vector c and return a set of vectors with m+1 elements. 
  Return:
    (max, x, y, it, n)
      max: solution of problem, if exists. A message is delivered if 
        problem has no solution and inf is returned.
      x, y: vectors with p and q elements respectively that solves the 
        problem, if solution exists. inf are returned otherwise.
      it: sum of iterations of subproblems
      n: number of iterations in main loop
  '''
  (m_A,p_A) = A.shape
  c = np.array(c)
  b = np.array(b)
  d = np.array(d)
  p_c = c.size
  q_d = d.size
  m_b = b.size
  (m_B,q_B) = B.shape
  if (p_c != p_A or m_b != m_A or q_d != q_B or m_B != m_b):
    print("Incompatible sizes of c, d, A, B and b.")
    return (float('inf'), float('inf'), float('inf'))
  else:
    m, p, q = m_b, p_c, q_d
  it = 0
  '''-------------------------------------------------------------------
  Original algorithm from
    J. F. Benders. Particioning procedures for solving mixed-variables
    programming problems. In: Numerische Matematik, 4. 1962. p. 238-252:

                      m
  1) Let u  ∈ ℝ, u ∈ ℝ  and  set:
          0
                  T
     C = {(u ,u)|A  u − cu  ≥ 0,u ≥ 0,u ≥ 0}
            0             0            0
                                           T                  T
     G(Q) =      ∩     {(x  ,y) | u  x  + u F(y) - u  f(y) ≤ u b, y ∈ S}
           (u , u) ∈ Q    0        0  0             0
             0
              T
     P =  {u|A u ≥ c, u ≥ 0}

              T
     C  = {u|A u ≥ 0, u ≥ 0}
      0
  -------------------------------------------------------------------'''

  '''-------------------------------------------------------------------
  2) Set Q ⊂ C
  -------------------------------------------------------------------'''
  Q = chooseSubsetC(A, B, c, d)
  '''-------------------------------------------------------------------
                                                 0
  3) if u  > 0 for at least one point (u , u) ∈ Q  then
         0                              0        

       3.1) goto 4, the first part of iterative step
                                              0
     else if  u  = 0 for any point (u , u) ∈ Q  then
               0                     0
             0
       3.2) x  = +∞
             0

             0                                    0
       3.3) y  = u, for any point of (u , u) ∈ G(Q )
                                       0
              Remark: In Benders original paper, he says 
               0                   0
              y  = any point of G(Q ).
                            0   q+1
              In this way, y ∈ ℝ.   However, in 4), we see that
                           ν                       q
              (x , y) ∈ G(Q ). Then, y must be in ℝ.
                0
       3.4) goto 6, the second part of iterative step
  -------------------------------------------------------------------'''
  goto_first_part = goto_second_part = False
  if reduce(operator.or_, [x[0] > 0 for x in Q]):
    goto_first_part = True
  else:
    x0 = float('inf')
    (y, it2) = chooseAnyPointG(Q, B, d, b)
    it = it + it2
    goto_second_part = True
  n = 0
  new_x0 = None
  new_y = [None]
  while True:
    '''-----------------------------------------------------------------
     4) first part of ν-th iteration: Solve
         ν   ν                           ν
       (x , y ) =  max{x  | (x , y) ∈ G(Q )}
         0              0     0
    -----------------------------------------------------------------'''
    if goto_first_part:
      (x0, y, ite) = linearSubproblemBounded(Q, B, d, b, M)
      it = it + ite
      previous_x0 = new_x0
      previous_y = new_y
      new_x0 = x0
      new_y = y
      '''---------------------------------------------------------------
      5) if problem in 4 is not feasible then

           5.1) terminate and warn that problem is infeasible

         else

           5.2) go to 6, the second part of iterative step
      ---------------------------------------------------------------'''
      if np.isinf(x0):
        print "Problem is infeasible."
        return ( float('inf'), (float('inf'),)*p, 
                 (float('inf'),)*q, it, n)
      goto_second_part = True
    '''-----------------------------------------------------------------
    6) second part of ν-th iteration: Solve
        ν               ν  T      T
       u  = min{(b - F(y ))  u | A u ≥ c, u ≥ 0}
    -----------------------------------------------------------------'''
    if goto_second_part:
      c1 = b - (B.dot(y)).tolist()[0]
      res = linprog(c = c1, A_ub = (-np.matrix(A)).T, 
                    b_ub = -c, callback = extremeRay_and_dRow,
                    options={"bland": True})
      u, it = res.x, it + res.nit
      '''---------------------------------------------------------------
      7) if problem in 6 is not feasible then

           7.1) terminate and warn that problem is infeasible
                         ν  T  ν    ν      ν
         else if (b - F(y ))  u  = x  - f(y )
                                    0

           7.2) Solve dual of 2nd part problem, i.e.,
            ν           T                 ν
           x  = max {( c x | A x ≤ b - F(y ), x ≥ 0}


                         ν   ν  ν
           7.3) return (x , x, y )
                         0
                         ν  T  ν    ν      ν
         else if (b - F(y ))  u  < x  - f(y )
                                    0
                 ν+1    ν         ν
           7.4) Q    = Q  ∪ {(1, u )}

           7.5) ν = ν + 1

           7.6) go to 4
      ---------------------------------------------------------------'''
      if res.status == 2 or res.status == 1:
        print "Problem is infeasible."
        return ( float('inf'), (float('inf'),)*p, 
                 (float('inf'),)*q, it, n)
      else:
        value1 = np.dot(c1, u)
        value2 = x0 - np.dot(d,y)
        if res.status == 0:
          if value1 == value2:
            x = d_row[c.size:-1]
            return (x0, x, y, it, n)
          elif value1 < value2:
            Q = Q.union([(1,)+tuple(u)])
          else:
            print "It is not possible (b - F(y ))^T u  > x  - f(y )."
            print "Something is wrong in step 7 of algorithm."
            return ( float('inf'), (float('inf'),)*p, 
                   (float('inf'),)*q, it, n)
          '''-----------------------------------------------------------
           8) if objective function in problem in 6 tends to infinity 
               along the halfline 
                           ν     ν
                 {u | u = u  + λv , λ ≥ 0}
                 
                    ν         ν
              with u ∈ P and v  the direction of an extreme halfline of
              C  then
               0
                              ν  T  ν     ν      ν
              8.1) if (b - F(y ))  u  ≥  x  - f(y ) then
                                          0
                             ν+1    ν         ν
                     8.1.1) Q    = Q  ∪ {(0, v )}
                                       ν  T  ν     ν      ν
                   else, i.e., (b - F(y ))  u  <  x  - f(y )
                                                   0
                             ν+1    ν         ν        ν
                     8.1.2) Q    = Q  ∪ {(1, u ), (0, v )}
              
              8.2) ν = ν + 1
              
              8.3) go to 4
          -----------------------------------------------------------'''
        else: #i.e., res.status == 3, or, problem in 6 is unbounded
          v = extremeDirection[0:len(u),0].T.tolist()[0]
          if value1 >= value2:
            Q = Q.union([(0.0,)+tuple(v)])
          else:
            Q = Q.union([(1.0,)+tuple(u),(0.0,)+tuple(v)])
    n = n + 1
    ''' Avoiding infinite loop. '''
    if(goto_first_part and (previous_x0 == new_x0) and np.array_equal(previous_y, new_y)):
      return (x0, x, y, it, n)
    goto_first_part = True

def benders_decomposition_linear_alternative(c, d, A, B, b, M,
                                 chooseSubsetC = chooseSubsetZero):
  '''
  Alternative Benders' procedure for linear functions. This procedure 
  solves the problem
       T                            p     
  max{c x + f(y)|Ax + F(y) ≤ b,x ∈ ℝ ,y ∈ S}
            q         T
  with S = ℝ, f(y) = d y, and F(y) = By,
  
  where
         p       q       m×p       m×q       m
    c ∈ ℝ , d ∈ ℝ , A ∈ ℝ   , B ∈ ℝ   , b ∈ ℝ .
  
  In this procedure, it is added the restriction to set P
     T                              T
    e u ≤ M, with e = (1, 1, ..., 1),
  
  in order to avoid first part subproblem unbounded solution.
  
  Arguments:
    c: vector with p elements;
    d: vector with q elements;
    A: matrix m x p;
    B: matrix m x q;
    b: vector with m elements;
    M: a big real value;
    chooseSubsetC: a function that takes a m x p matrix A, a m x q 
      matrix B, a p elements vector c and a q elements vector 
      d and return a set of vectors with m+1 elements. 
  Return:
    (max, x, y, it, n)
      max: solution of problem, if exists. A message is delivered if 
        problem has no solution and inf is returned.
      x, y: vectors with p and q elements respectively that solves the 
        problem, if solution exists. inf are returned otherwise.
      it: sum of iterations of subproblems
      n: number of iterations in main loop
  '''
  (m_A,p_A) = A.shape
  c = np.array(c)
  b = np.array(b)
  d = np.array(d)
  p_c = c.size
  q_d = d.size
  m_b = b.size
  (m_B,q_B) = B.shape
  if (p_c != p_A or m_b != m_A or q_d != q_B or m_B != m_b):
    print("Incompatible sizes of c, d, A, B and b.")
    return (float('inf'), float('inf'), float('inf'))
  else:
    m, p, q = m_b, p_c, q_d
  it = 0
  '''-------------------------------------------------------------------
  Alternative algorithm from
    J. F. Benders. Particioning procedures for solving mixed-variables
    programming problems. In: Numerische Matematik, 4. 1962. p. 238-252:

                      m
  1) Let u  ∈ ℝ, u ∈ ℝ  and  set:
          0
                  T
     C = {(u ,u)|A  u − cu  ≥ 0,u ≥ 0,u ≥ 0}
            0             0            0
                                           T                  T
     G(Q) =      ∩     {(x  ,y) | u  x  + u F(y) - u  f(y) ≤ u b, y ∈ S}
           (u , u) ∈ Q    0        0  0             0
             0
               T          T                                T
     P(M) =  {u|A u ≥ c, e u ≤ M, u ≥ 0, e = (1, 1, ..., 1) }

              T
     C  = {u|A u ≥ 0, u ≥ 0}
      0
  -------------------------------------------------------------------'''

  '''-------------------------------------------------------------------
  2) Set Q ⊂ C
  -------------------------------------------------------------------'''
  Q = chooseSubsetC(A,B,c,d)
  '''-------------------------------------------------------------------
                                                 0
  3) if u  > 0 for at least one point (u , u) ∈ Q  then
         0                              0        

       3.1) goto 4, the first part of iterative step
                                              0
     else if  u  = 0 for any point (u , u) ∈ Q  then
               0                     0
             0
       3.2) x  = +∞
             0

             0                                    0
       3.3) y  = u, for any point of (u , u) ∈ G(Q )
                                       0
              Remark: In Benders original paper, he says 
               0                   0
              y  = any point of G(Q ).
                            0   q+1
              In this way, y ∈ ℝ.   However, in 4), we see that
                           ν                       q
              (x , y) ∈ G(Q ). Then, y must be in ℝ.
                0
       3.4) goto 6, the second part of iterative step
  -------------------------------------------------------------------'''
  goto_first_part = goto_second_part = False
  if reduce(operator.or_, [x[0] > 0 for x in Q]):
    goto_first_part = True
  else:
    x0 = float('inf')
    (y, it2) = chooseAnyPointG(Q, B, d, b)
    it = it + it2
    goto_second_part = True
  n = 0
  new_x0 = None
  new_y = [None]
  while True:
    '''-----------------------------------------------------------------
     4) first part of ν-th iteration: Solve
         ν   ν                           ν
       (x , y ) =  max{x  | (x , y) ∈ G(Q )}
         0              0     0
    -----------------------------------------------------------------'''
    if goto_first_part:
      (x0, y, ite) = linearSubproblem(Q, B, d, b)
      it = it + ite
      previous_x0 = new_x0
      previous_y = new_y
      new_x0 = x0
      new_y = y
      '''---------------------------------------------------------------
      5) if problem in 4 is not feasible then

           5.1) terminate and warn that problem is infeasible

         else

           5.2) go to 6, the second part of iterative step
      ---------------------------------------------------------------'''
      if np.isinf(x0):
        print "Problem is infeasible."
        return ( float('inf'), (float('inf'),)*p, 
                 (float('inf'),)*q, it, n)
      goto_second_part = True
    '''-----------------------------------------------------------------
    6) second part of ν-th iteration: Solve
        ν   ν         T                            ν
      (x , z ) = max{c x - M z | A x - z e ≤ b - By , x ≥ 0, z ≥ 0}
            0                 0         0                     0
                          T   m
      with e = (1, ..., 1) ∈ ℝ.
    -----------------------------------------------------------------'''
    if goto_second_part:
      c1 = np.append(c,-M) 
      A1 = np.concatenate((A, np.matrix((-1.0,)*m).T),axis=1)
      b1 = b - (B.dot(y)).tolist()[0]
      res = linprog(c = -c1, A_ub = A1, b_ub = b1, 
                    callback = extremeRay_and_dRow,
                    options={"bland": True})
      '''---------------------------------------------------------------
      7) if problem in 6 is finite then
                   ν         T ν   T ν   ν
          7.1) if z = 0 and c x + d y = x  then
                   0                     0
                                   ν   ν
                 7.1.1) return (x , y )
      ---------------------------------------------------------------'''
      if res.status == 0:
        x, z0, it = res.x[:-1], res.x[-1], it + res.nit
        cx = np.dot(c,x)
        dy = np.dot(d,y)
        cxdy = cx + dy
        if z0 == 0 and cxdy == x0:
          return (x0, x, y, it, n)
          '''-----------------------------------------------------------
                        ν         T ν   T ν   ν      ν
              else if (z = 0 and c x + d y < x ) or z > 0
                        0                     0      0
                                                   1,ν     2,ν
                7.1.2) determine the M-components d   and d    of the
                       d-row in the optimum simplex tableau and the 
                                     1,ν     2,ν
                       M-components u   and u    of the optimum 
                       solution of the dual problem.
          -----------------------------------------------------------'''
        elif (z0 == 0 and cxdy < x0) or z0 > 0:
          # Recalculating d-row by replacing the objective function
          #                        T
          # in optimum tableau by c x.
          d1 = d_row[:-1] * ( cx/d_row[-1]) 
          # Recalculating d-row by replacing the objective function
          # in optimum tableau by -z .                             
          #                         0                              
          d2 = d_row[:-1] * (-z0/d_row[-1])
          #  1,ν     1,ν
          # u   and u    are obtained by taking respectively d1 and d2 
          # components corresponding to slack variables in primal 
          # simplex tableau 
          u1 = d1[res.x.size:len(d1)]
          u2 = d2[res.x.size:len(d2)]
          '''-----------------------------------------------------------
                           2,ν
                7.1.3) if u   = 0
                                   ν   1,ν      ν
                         7.1.3.1) u = u    and v = 0

                       else
                                            ⎧   1,ν  │         ⎫
                                            ⎪  d     │         ⎪
                                            ⎪   j    │  2,ν    ⎪
                         7.1.3.2) M   = max ⎨- ――――  │ d    > 0⎬
                                   min   j  ⎪   2,ν  │   j     ⎪ 
                                            ⎪  d     │         ⎪
                                            ⎩   j    │         ⎭

                                   ν   1,ν         2,ν      ν   2,ν
                         7.1.3.3) u = u    + M    u    and v = u
                                              min
          -----------------------------------------------------------'''
          if not u2.any():
            u = u1
            v = np.zeros(u1.size)
            '''
              In Benders' paper, he do not tell what to do if
                    ν+1     ν
              with Q    if u = 0. So, I assume the same in original 
              Benders procedure, i.e.,
              
                          ν  T  ν     ν      ν
               if (b - F(y ))  u  ≥  x  - f(y ) then
                                      0
                      ν+1    ν         ν
                     Q    = Q  ∪ {(0, v )}
                                   ν  T  ν     ν      ν
               else, i.e., (b - F(y ))  u  <  x  - f(y )
                                               0
                     ν+1    ν         ν        ν
                    Q    = Q  ∪ {(1, u ), (0, v )}
            '''
            c1_ = b - (B.dot(y)).tolist()[0]
            value1 = np.dot(c1_, u)
            value2 = x0 - np.dot(d,y)
            if value1 >= value2:
              Q = Q.union([(0.0,)+tuple(v)])
            else:
              Q = Q.union([(1.0,)+tuple(u),(0.0,)+tuple(v)])
          else:
            Mmin = max(map(lambda (x,y): -x/y, 
                       filter(lambda (x,y): y > 0, zip(d1,d2))))
            u = u1 + Mmin * u2
            v = u2
            '''---------------------------------------------------------
                           T  ν        ν    ν      ν
                7.1.4) if c  x  - M   z  < x  - f(y )
                                   min 0    0

                                   ν+1    ν         ν        ν
                         7.1.4.1) Q    = Q  ∪ {(1, u ), (0, v )}

                       else
                                   ν+1    ν         ν
                         7.1.4.2) Q    = Q  ∪ {(0, v )}
            ---------------------------------------------------------'''
            if cx - Mmin * z0 < x0 - dy:
              Q = Q.union([(1.0,)+tuple(u),(0.0,)+tuple(v)])
            else:
              Q = Q.union([(0.0,)+tuple(v)])
      else:
        '''-------------------------------------------------------------
        else // from if in 7

          7.2) terminate and warn that problem is infeasible

        8) ν = ν + 1

        9) go to 4
        -------------------------------------------------------------'''
        print "Problem is infeasible."
        return ( float('inf'), (float('inf'),)*p, 
                 (float('inf'),)*q, it, n)
    n = n + 1
    ''' Avoiding infinite loop. '''
    if(goto_first_part and (previous_x0 == new_x0) and np.array_equal(previous_y, new_y)):
      return (x0, x, y, it, n)
    goto_first_part = True
