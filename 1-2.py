#!/usr/bin/env python3
# encoding: utf-8

# GRADED FUNCTION: basic_sigmoid

import math

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    :param x: A scalar
    :return: s -- sigmoid(x)
    """

    ### START CODE HERE ###
    s = 1/ (1+ math.exp(-x))
    ### END CODE HERE ###
    return s

print(basic_sigmoid(3))


import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid of x.

    Arguments:
        x -- A scalar or numpy array of any size.

    Return:
        s = sigmoid(x)
    """
    s = 1/(1 + np.exp(-x))
    return s

x = np.array(list(range(1, 4)))
print('x: {}'.format(x))
print('sigmoid: ', sigmoid(x))

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative)
    of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables
    and then use it to calculate the gradient.

    :param x: A scalar or numpy array.
    :return: ds -- Your computed gradient.
    """

    s = sigmoid(x)
    ds = s*(1-s)

    return ds

print('sigmoid derivative: ', sigmoid_derivative(x))


def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Return:
    v - a vector of shape (length*height*depth, 1)
    """

    # v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
    v = image.reshape((np.prod(image.shape) , 1))
    return v

image = np.random.rand(3,3,2)
print("image2vector(image): \n", image2vector(image))


##############################################################
# normalize rows
x = np.array([0,3,4,2,6,4]).reshape(2,3)

def normalizeRows(x):
    """
    Implement a function that normalizes each rows of the matrix x
    to have unit length.

    Arguments:
         x: A numpy matrix of shape (n, m)
    Return:
        x--The normalized (by row) numpy matrix.
        You are allowed to modifty x.
    """
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = x/x_norm

    return x

print("normalizeRows(x):\n", normalizeRows(x))


######################################################
# softmax
def softmax(x):
    """Calculate the softmax for each row of the input x.
    The code work for a row vector and also for matrices of shape(n, m)

    Argument:
        x -- A numpy matrix of shape(n, m)

    Return:
        s -- A numpy matrix equal to the softmax x, of shape(n, m)
    """

    x_exp = np.exp(x)
    x_exp_sum = np.sum(x_exp, axis=1, keepdims=True)

    s = x_exp / x_exp_sum

    return s

x = np.array([[9, 2, 5, 0, 0],
              [7, 5, 0, 0, 0]])

print("softmax(x):\n", softmax(x))

print(np.sum(softmax(x), axis=1, keepdims=True))

###############################################
# 向量化
print('\n'*2 + '*'*50)
import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

###################
print('\n'*5, '='*50)
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
dot = np.dot(W,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

###
print("\n"*5)
def L1(yhat, y):
    loss = np.sum(np.abs(y-yhat))
    return loss

yhat = np.array([.9, .2, .1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = ", L1(yhat, y))

def L2(yhat, y):
    loss = np.sum(np.power((yhat-y), 2))
    return loss

print("L2 = {}".format(L2(yhat, y)))