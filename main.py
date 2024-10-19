from mnist import getMnist
import numpy as np
import matplotlib.pyplot as plt


"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""
img, lbl = getMnist()
wInpToHidden = np.random.uniform(-0.5, 0.5, (20, 784))
wHiddenToOut = np.random.uniform(-0.5, 0.5, (10, 20))
bInpToHidden = np.zeros((20, 1))
bHiddenToOut = np.zeros((10, 1))

rol = 0.01
numCorrect = 0
epochs = 3
for epoch in range(epochs):
    for img, l in zip(img, lbl):
        img.shape += (1,)
        l.shape += (1,)

        # Forward propagation
        inpToHidden = bInpToHidden + wInpToHidden @ img
        hidden = 1 / (1 + np.exp(-inpToHidden))

        hiddenToOut = bHiddenToOut + wHiddenToOut @ hidden
        out = 1 / (1 + np.exp(-hiddenToOut))

        # Error calculation
        e = 1 / len(out) * np.sum((out - l) ** 2, axis=0)
        numCorrect += int(np.argmax(out) == np.argmax(l))

        # Backpropagation output
        delta_o = out - l
        wHiddenToOut += -rol * delta_o @ np.transpose(hidden)
        bHiddenToOut += -rol * delta_o
        
        delta_h = np.transpose(wHiddenToOut) @ delta_o * (hidden * (1 - hidden))
        wInpToHidden += -rol * delta_h @ np.transpose(img)
        bInpToHidden += -rol * delta_h

    # Show accuracy for this epoch
    print(f"Accuracy: {round((numCorrect / img.shape[0]) * 100, 2)}%")
    numCorrect = 0


