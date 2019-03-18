import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import model_selection
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

print("Loading binned openness data")
testBinnedOpennessReshaped = np.load("data/pairedData/human/testBinnedOpennessReshaped.npy")
testBinnedOpennessReshaped = np.reshape(testBinnedOpennessReshaped, (200, 2000, 201)) # original shape
testBinnedOpennessReshaped = np.reshape(testBinnedOpennessReshaped, (200, 2000*201))
testBinnedOpennessReshaped_mean = (testBinnedOpennessReshaped - np.mean(testBinnedOpennessReshaped, axis=1)[:, np.newaxis])
testBinnedOpennessReshaped = testBinnedOpennessReshaped_mean/(np.std(testBinnedOpennessReshaped, axis=1)[:, np.newaxis] + 10**-8)
testBinnedOpennessReshaped = np.reshape(testBinnedOpennessReshaped, (200, 2000, 201))
testBinnedOpennessReshaped = np.swapaxes(testBinnedOpennessReshaped, 1, 2)
testBinnedOpennessReshaped.shape # should be (200, 201, 2000)
X = np.reshape(testBinnedOpennessReshaped, (200*201, 2000))

print("Loading gene expression data")
Y = np.genfromtxt("data/pairedData/human/testGeneExpression.txt", delimiter = '\t')
Y = np.log(Y + 10**-8)
Y_mean = Y - np.mean(Y, axis=1)[:,np.newaxis]
Y = Y_mean / (np.std(Y_mean, axis=1)[:,np.newaxis] + 10**-8)
Y = np.reshape(Y, (200*201, 1))

#Linear Regression Baseline

m,n = X.shape
print(Y.shape)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=10)
X_dev, X_test, Y_dev, Y_test = model_selection.train_test_split(X_test, Y_test, test_size=0.50, random_state=10)

clf = Lasso(alpha=0.002)
clf.fit(X_train, Y_train)
y_hat = clf.predict(X_train)[:,np.newaxis]
min_mse = np.sum(np.square(y_hat - Y_train)/y_hat.shape[0])
print(min_mse)

alphas = np.arange(0.001,0.01,0.001)
print(alphas)
mse_array = np.zeros(len(alphas))
for i in range(len(alphas)):
    print('%d out of %d'%(i,len(alphas)))
    clf = Lasso(alpha=alphas[i])
    clf.fit(X_train, Y_train)
    y_hat = clf.predict(X_dev)[:,np.newaxis]
    mse_array[i] = np.sum(np.square(y_hat - Y_dev)/y_hat.shape[0])

plt.figure()
plt.plot(alphas, mse_array)
plt.show()


minAlpha = alphas[np.argmin(mse_array)]

clf = Lasso(alpha=minAlpha)
clf.fit(X_train, Y_train)
y_hat = clf.predict(X_test)[:,np.newaxis]
min_mse = np.sum(np.square(y_hat - Y_test)/y_hat.shape[0])

print(minAlpha)
print(min_mse)



