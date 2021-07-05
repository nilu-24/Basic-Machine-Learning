#Implementation of Gradient Descent Algorithm 
import numpy as np
#enter x , y, count is number of iterations, and learning rate
#adjust learning rate and count as per needed


def grad_descent(x,y,count,learning_rate): #input x and y are numpy arrays
      n = len(x)
      m = 0 #gradient is 0
      b = 0 #y-intercept is 0
      
      for i in range(count):
          y_pred = m*x + b #prediction function

          #cost function
          cost_func = (1/n)*sum((y-y_pred)*(y-y_pred))
         
          #calculating partial derivatives
          partial_d_m = (2/n)*sum(-x*(y-y_pred)) 
          partial_d_b = (2/n)*sum(-(y-y_pred))

          #adjusting m and b with learning rate
          m = m - learning_rate*partial_d_m
          b = b - learning_rate*partial_d_b
          

          #print
          print(f"m: {m} b: {b} cost: {cost_func}")


