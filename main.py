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

print("WORKING")

