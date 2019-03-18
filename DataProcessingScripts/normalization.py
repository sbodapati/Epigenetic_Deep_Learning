import numpy

binnedOpenness = numpy.load('data/binnedOpenness.npy') # shape = (17794, 2000, 201)
Y = numpy.genfromtxt("data/pairedData/human/geneExpression.txt", delimiter = '\t') # shape = (17794, 201)

Y = numpy.log(Y + 1.0)
Y = Y - numpy.mean(Y, axis=1)[:,numpy.newaxis]
Y = Y / (numpy.std(Y, axis=1)[:,numpy.newaxis] + 10**-8)
Y = numpy.reshape(Y, newshape = (Y.shape[0]*Y.shape[1], 1))
numpy.save(file = "data/logNormalizedGeneExpressionReshaped.npy", arr = Y)

binnedOpenness = numpy.log(binnedOpenness + 1.0)
binnedOpenness = binnedOpenness - numpy.mean(binnedOpenness, axis = 2)[:, :, numpy.newaxis]
binnedOpenness = binnedOpenness / (numpy.std(binnedOpenness, axis = 2)[:, :, numpy.newaxis] + 10**-8)
binnedOpenness = numpy.swapaxes(binnedOpenness, 1, 2)
binnedOpenness = numpy.reshape(binnedOpenness, newshape = (binnedOpenness.shape[0]*binnedOpenness.shape[1], binnedOpenness.shape[2]))

numpy.save(file = "data/logNormalizedBinnedOpennessReshaped.npy", arr = binnedOpenness)
