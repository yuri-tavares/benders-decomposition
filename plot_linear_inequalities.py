import numpy as np
import matplotlib.pyplot as plt


def plot_linear_inequalities(A,b):
  '''
  Plot linear restrictions for linear programming problems.
  '''
  (m,n) = A.shape
  if n > 2:
    print "This function works only in 2 dimensions."
    return
  m_b = len(b)
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

test_with = plot_linear_inequalities

if __name__ == '__main__':
  b1 = [0,400,16,100,100]
  A1 = np.matrix([[0,0],[1,94],[1,-2],[0,1],[1,0]])
  test_with(A1,b1)
