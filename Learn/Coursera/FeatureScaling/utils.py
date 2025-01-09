import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_house_data():
    data = np.loadtxt('houses.txt', delimiter=',', skiprows=1)
    print(data.shape)
    X = data[:, :2]
    Y = data[:, 2]
    return X, Y

def compute_cost(x, y, w, b):
    """
        Compute Multivariable Vector Array
        X: m by n matrix (3, 4)
        Y: n value array (:, 4)
        b: integer (1)
        w: n value array (:, 4)
        note: cost represent J(w, b) before dividing m
        
        return 
            cost (scalar): cost (scalar mean single value)
    """
    cost = 0.0
    m = x.shape[0] # row length
    
    for i in range(m):
        # x[i] is a row from x matrix 
        fwb_i = np.dot(x[i], w) + b
        cost = cost + (fwb_i - y[i])**2    
        
        
    return cost/(2*m)

def dJdw(x, y, w, b):
    
    # row "m" for number of examples, column "n" for number of features 
    m, n = x.shape
    dJdw = np.zeros((n, )) # zero array length n (i.e. 4)
    
    for i in range(m):
        
        fwb = np.dot(w, x[i]) + b
        # x[i, :] return i-th row of all features. Since we calc row by row this will be useful
        cost = np.dot(fwb - y[i], x[i, :])
        # print('cost:', cost)
        dJdw += cost # calc sum of 2 array size (, 4) 
        # print('dJdw:', dJdw)
    
    # print("--- final result ---")
    return dJdw/m # final m division

def dJdb(x, y, w, b):
        
    m, n = x.shape
    dJdb = 0
    
    for i in range(m):
        fwb = np.dot(w, x[i]) + b
        cost = fwb - y[i]
        dJdb += cost         

    return dJdb/m 

def gradient_descent(x, y, wj, bj, iter=40, learning_rate=0.001):
  """
      Performs batch gradient descent to learn w and b. Updates w and b by taking 
      num_iters gradient steps with learning rate alpha
      
      Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
      Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
  """
  w = wj
  b = bj
  J_history = []
  
  for j in range(iter):
    w = w - learning_rate*dJdw(x, y, w, b)
    b = b - learning_rate*dJdb(x, y, w, b)
    cost =  compute_cost(x, y, w, b)
    
    if j < 100000:
      J_history.append(cost)
    
    print(f"Iteration {j:4d}: Cost {cost}")
    
  return w, b, J_history

def mean_normalization(x):
    pass
        

def zscore_standardization(x):
    pass


def main():
    X = load_house_data()
    mean_normalization(X)


    
if __name__ == '__main__':
    main()