import numpy as np
import matplotlib.pyplot as plt
import mnist as mn

# GETS MNIST DATASET TO TRAIN NUERAL NET MODEL
img, lbl = mn.getMnist()

# Matrix for Weights randomly distributed
# 20 "Neurons" X 784 pixels
inputWeights = np.random.uniform(-0.5,0.5, (20,784))
hiddenWeights = np.random.uniform(-0.5,0.5,(10,20))
# Matrix for Biases Starts at 0 then trains model 
inputBias = np.zeros((20,1))
hiddenBias = np.zeros((10,1))

# Rate of learning
rol = 0.01

# How many times the training data is ran through the net
epochs = 1

for epoch in range(epochs):
    for img, l in zip(img,lbl):
        # Reshapes vectors into matrix of 1 col
        img.shape += (1,)
        l.shape += (1,)

    # Forward propagation bias + weight @ img for input weights
    inputPropagation = inputBias + inputWeights @ img
    inputProp = 1/(1+np.exp(-inputPropagation))



