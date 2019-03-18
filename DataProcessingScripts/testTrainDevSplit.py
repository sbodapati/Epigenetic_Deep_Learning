import numpy

numpy.random.seed(123)

Y = numpy.load("data/logNormalizedGeneExpressionReshaped.npy")
X = numpy.load("data/logNormalizedBinnedOpennessReshaped.npy")
assert Y.shape[0] == X.shape[0]

indices = numpy.arange(0, Y.shape[0], 1)
numpy.random.shuffle(indices)

Y_train = Y[indices[0:int(0.9*Y.shape[0])], :]
Y_dev = Y[indices[int(0.9*Y.shape[0]):int(0.95*Y.shape[0])],:]
Y_test =Y[indices[int(0.95*Y.shape[0]):Y.shape[0]],:]

for i in range(100):
  filename = "data/splits/Merged/logNormalizedGeneExpressionReshapedTrain_" + str(i) + ".npy"
  numpy.save(arr = Y_train[int(i*0.01*Y_train.shape[0]):int((i+1)*0.01*Y_train.shape[0]),:], file = filename)

numpy.save(arr = Y_dev, file = "data/splits/Merged/logNormalizedGeneExpressionReshapedDev.npy")
numpy.save(arr = Y_test, file = "data/splits/Merged/logNormalizedGeneExpressionReshapedTest.npy")

X_train = X[indices[0:int(0.9*X.shape[0])], :]
X_dev = X[indices[int(0.9*X.shape[0]):int(0.95*X.shape[0])],:]
X_test =X[indices[int(0.95*X.shape[0]):X.shape[0]],:]

for i in range(100):
  filename = "data/splits/Merged/logNormalizedBinnedOpennessReshapedTrain_" + str(i) + ".npy"
  numpy.save(arr = X_train[int(i*0.01*X_train.shape[0]):int((i+1)*0.01*X_train.shape[0]),:], file = filename)

numpy.save(arr = X_dev, file = "data/splits/Merged/logNormalizedBinnedOpennessReshapedDev.npy")
numpy.save(arr = X_test, file = "data/splits/Merged/logNormalizedBinnedOpennessReshapedTest.npy")
