import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn
import matplotlib.pyplot as plt

print("Loading gene expression data")
geneExpression = np.genfromtxt("data/pairedData/human/geneExpression.txt", delimiter = "\t")
eps = 1e-8
logGeneExpression = np.log(geneExpression + eps)
logGeneStd = np.std(logGeneExpression, axis = 1)
print("logGeneStd.shape = ", logGeneStd.shape)

seaborn.distplot(logGeneStd, hist=True, kde=True, bins=100, color = 'darkblue', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4})
plt.show()

genes2keep = np.greater(logGeneStd, 0.5)
logGeneExpressionFiltered = logGeneExpression[genes2keep]
np.savetxt(X = genes2keep, fname = "data/pairedData/human/genes2keep.txt", delimiter = "\n")
logGeneExpressionFiltered = logGeneExpression[genes2keep]
logGeneExpressionFiltered.shape
logGeneExpressionFilteredNormalized = (logGeneExpressionFiltered - np.mean(logGeneExpressionFiltered, axis = 1)[:, np.newaxis])/np.std(logGeneExpressionFiltered, axis = 1)[:, np.newaxis]
np.savetxt(fname = "data/pairedData/human/logGeneExpressionFilteredNormalized.txt", X = logGeneExpressionFilteredNormalized, delimiter = "\t")

