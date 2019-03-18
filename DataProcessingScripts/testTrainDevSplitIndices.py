import numpy

numpy.random.seed(123)

Y = numpy.load("data/logNormalizedGeneExpressionReshaped.npy")
X = numpy.load("data/logNormalizedBinnedOpennessReshaped.npy")
assert Y.shape[0] == X.shape[0]

indices = numpy.arange(0, Y.shape[0], 1)
numpy.random.shuffle(indices)

train = indices[0:int(0.9*Y.shape[0])]
dev = indices[int(0.9*Y.shape[0]):int(0.95*Y.shape[0])]
test =indices[int(0.95*Y.shape[0]):Y.shape[0]]

numpy.savetxt(arr = dev, file = "data/splits/Merged/devIndices.txt")
numpy.savetxt(arr = test, file = "data/splits/Merged/testIndices.txt")
