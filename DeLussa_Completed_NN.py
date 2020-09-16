#!/usr/bin/env python
# coding: utf-8

# In[188]:


# Imports the neccessary libraries
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt


# In[189]:


#Creates an array of the data we will analyze
features_and_targets = np.array( 
                                   [ [0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1, 0, 1],
                                     [0, 0, 0, 1, 1, 0, 1],
                                     [0, 0, 1, 1, 1, 0, 1],
                                     [0, 1, 1, 1, 1, 0, 1],
                                     [1, 1, 1, 1, 0, 0, 1],
                                     [1, 1, 1, 0, 0, 0, 1],
                                     [1, 1, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 1, 0, 0, 1],
                                     [1, 0, 1, 1, 0, 0, 1],
                                     [1, 1, 0, 1, 0, 0, 1],
                                     [0, 1, 0, 1, 1, 0, 1],
                                     [0, 0, 1, 0, 1, 0, 1],
                                     [1, 0, 1, 1, 1, 1, 0],
                                     [1, 1, 0, 1, 1, 1, 0],
                                     [1, 0, 1, 0, 1, 1, 0],
                                     [1, 0, 0, 0, 1, 1, 0],
                                     [1, 1, 0, 0, 1, 1, 0],
                                     [1, 1, 1, 0, 1, 1, 0],
                                     [1, 1, 1, 1, 1, 1, 0],
                                     [1, 0, 0, 1, 1, 1, 0]  ]
                           , dtype=float)

# Shuffle our cases by rows 
np.random.shuffle(features_and_targets)

# Creates the variables features, targets_observed, number_of_features, 
# and number_of_cases; which is the above data separated to include
# the first 5 columns on the left as features and the last 2 columns 
# on the right as targets. The data is then transposed in order
# to be multiplied later
features           = np.transpose(features_and_targets[:,0:5])
targets_observed   = np.transpose(features_and_targets[:,5:7])
number_of_features = features.shape[0]
number_of_cases    = features.shape[1]


# In[190]:


# Set initial weights and biases in an array of randomly selected numbers

w1   = np.random.random((4,5))
b1   = np.random.random((4,22))

w2   = np.random.random((3,4))
b2   = np.random.random((3,22))

w3   = np.random.random((2,3))
b3   = np.random.random((2,22))


# In[191]:


# Defined sigmoid function to use in the feed_forward function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Defined relu function to use in the feed_forward function
def relu(x):
    return np.maximum(0.0,x)

# Defined loss function which uses relu for the first two activation
# functions, then usues the sigmoid function to uniformly place all
# results between 0 and 1.
def loss(features,w1,b1,w2,b2,w3,b3,targets_observed):
    for i in range(number_of_cases):
        Meas_1            = np.matmul(w1,features)
        Meas_1_add_b1     = np.add(Meas_1, b1)
        Meas_1_activation = relu(Meas_1_add_b1)
    
        Meas_2            = np.matmul(w2,Meas_1_activation)
        Meas_2_add_b2     = np.add(Meas_2, b2)
        Meas_2_activation = relu(Meas_2_add_b2)
    
        Meas_3            = np.matmul(w3,Meas_2_activation)
        Meas_3_add_b3     = np.add(Meas_3, b3)
        Meas_3_activation = sigmoid(Meas_3_add_b3)
    return np.sum((Meas_3_activation-targets_observed)**2)


# In[192]:


# Defines the Neural Network function which ultimately performs the
# back propigation and arrives at the final weights and biases after
# they've finished learning
def NN(features,w1,b1,w2,b2,w3,b3):
# Creates an empty vector
    loss_values = []                     
# Controls how fast the network will learn    
    learning_rate = 0.01                 

# Loop to calculate the slopes after the loss function has been applied
    for it in range(number_of_cases):    
        d_cost_w1 = grad(loss,1)
        d_cost_b1 = grad(loss,2)
        d_cost_w2 = grad(loss,3)
        d_cost_b2 = grad(loss,4)
        d_cost_w3 = grad(loss,5)
        d_cost_b3 = grad(loss,6)
        
# Prints the current row of targets the function is trying to predict    
        print('Target: ', targets_observed)        

# Sets the number of iterations for learning
        epoch = 100                       
           
# Uses the learning rate and the gradeint of the slope calculated above,
# to generate/ update a new weight/ bias. The loss funciton is then applied,
# to each new weight/ bias where the values are saved to the a list
        for iter in range(epoch): 
            w1 -= learning_rate*d_cost_w1(features,w1,b1,w2,b2,w3,b3,targets_observed)
            b1 -= learning_rate*d_cost_b1(features,w1,b1,w2,b2,w3,b3,targets_observed)
            w2 -= learning_rate*d_cost_w2(features,w1,b1,w2,b2,w3,b3,targets_observed)
            b2 -= learning_rate*d_cost_b2(features,w1,b1,w2,b2,w3,b3,targets_observed)
            w3 -= learning_rate*d_cost_w3(features,w1,b1,w2,b2,w3,b3,targets_observed)
            b3 -= learning_rate*d_cost_b3(features,w1,b1,w2,b2,w3,b3,targets_observed)
            current_loss = loss(features,w1,b1,w2,b2,w3,b3,targets_observed)
            loss_values = np.append(loss_values, current_loss)

# Applies the activation to the new values then applies the result
# to the sigmoid function. From this calculation the function predicts the
# new output
            Meas_1            = np.matmul(w1,features)
            Meas_1_add_b1     = np.add(Meas_1, b1)
            Meas_1_activation = relu(Meas_1_add_b1)

            Meas_2            = np.matmul(w2,Meas_1_activation)
            Meas_2_add_b2     = np.add(Meas_2, b2)
            Meas_2_activation = relu(Meas_2_add_b2)
    
            Meas_3            = np.matmul(w3,Meas_2_activation)
            Meas_3_add_b3     = np.add(Meas_3, b3)
            Meas_3_activation = sigmoid(Meas_3_add_b3)
            
            
# Prints the current iteration and the output from the function at that
# iteration followed by a line for visual ease of interpretation
        print('Iteration: ',it,'Output after learning: ', np.round(Meas_3_activation[0],3))
        print('-----------------------------------------------------------------------------')   
                
# Creates a scatter plot of the loss vaules over each iteration    
        plt.scatter(x=list(range(0,len(loss_values))), y=loss_values)
        plt.title("Loss per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.ylim(bottom=0)
        


# In[193]:


NN(features,w1,b1,w2,b2,w3,b3)

