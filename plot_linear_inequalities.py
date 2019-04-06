import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def surpress_non_zeros(zz):
  return np.matrix([map(lambda n: 0. if (n <= 0.) else n, zz[i]) for i in range(0,zz.shape[0])])

def plot_linear_inequalities3d(A,b):
  '''
  Plot linear restrictions for linear programming problems using a 3d plane
  '''
  # Set here the number of points per axis. The more points, the slower it gets.
  POINTS_PER_AXIS = 10
  (m,n) = A.shape
 
  # plot surfaces
  plt3d = plt.figure().gca(projection='3d')
  for i in range(0,m):
    # calculate corresponding third axis, depending if coefficient is not zero
    if A[i,2] != 0:
      xx, yy = np.meshgrid(range(POINTS_PER_AXIS), range(POINTS_PER_AXIS))
      zz = (-A[i,0]*xx - A[i,1]*yy + b[i])*1./A[i,2]
      zz = surpress_non_zeros(zz)
    elif A[i,1] !=0:
      xx, zz = np.meshgrid(range(POINTS_PER_AXIS), range(POINTS_PER_AXIS))
      yy = (-A[i,0]*xx - A[i,2]*zz + b[i])*1./A[i,1]
      yy = surpress_non_zeros(yy)
    elif A[i,0] !=0:
      yy, zz = np.meshgrid(range(POINTS_PER_AXIS), range(POINTS_PER_AXIS))
      xx = (-A[i,1]*yy - A[i,2]*zz + b[i])*1./A[i,0]
      xx = surpress_non_zeros(xx)
    else:
      continue
    n1,n2,n3 = np.random.randint(0,m)/(m+1.), np.random.randint(0,m)/(m+1.0), np.random.randint(0,m)/(m+1.0)
    plt3d.plot_surface(xx,yy,zz, color=(n1,n2,n3,0.75))
  plt.show()

def plot_linear_inequalities2d(A,b):
  '''
  Plot linear restrictions for linear programming problems using a 2d plane
  '''
  m_b = len(b)
  (m,n) = A.shape
  if m_b != m:
    print "Sizes of A and b are incompatible."
    return
  i=0
  min = np.matrix.min(np.absolute(A))
  if min == 0: min = 1
  xmax = ymax = np.max(np.absolute(b))/min + 1.0
  for a in A:
    if a[:,1][0,0] != 0:
      X=(0.0,float(xmax))
      Y=( (b[i]-a[:,0][0,0]*X[0])/a[:,1][0,0], (b[i]-a[:,0][0,0]*X[1])/a[:,1][0,0])
    else:
      X=(b[i],b[i])
      Y=(0.0,float(ymax))
    plt.plot(X,Y)
    i=i+1
  plt.xlim(0,xmax)
  plt.ylim(0,ymax)
  plt.show()

def plot_linear_inequalities(A,b):
  '''
  Plot linear restrictions for linear programming problems.
  '''
  (m,n) = A.shape
  if n > 3 or n <= 0:
    print "This function only works in 2 or 3 dimensions."
    return
  elif n <= 2:
    plot_linear_inequalities2d(A,b)
  else:
    plot_linear_inequalities3d(A,b)

def test_this():
  A = np.matrix([[1,1],[1,4]])
  b = [2,5]
  plot_linear_inequalities(A,b)

def test_2():
  b1 = [0,16]
  A1 = np.matrix([[0,0],[ 1., -2.]])
  plot_linear_inequalities(A1,b1)

def test_3():
  b1 = [-1.,1.,1.]
  A1 = np.matrix([[-1.,-1.],[ -1., 1.],[1., -1.]])
  plot_linear_inequalities(A1,b1)

def test_4():
  b1 = [0.,8.,-9.]
  A1 = np.matrix([[1.,-2.,1],[ 0., 2, -8,],[-4., 5.,9.]])
  plot_linear_inequalities(A1,b1)


test_with = plot_linear_inequalities

if __name__ == '__main__':
  b1 = [0, 16]
  A1 = np.matrix([[0.,1],[1,-6]])
  test_with(A1,b1)
