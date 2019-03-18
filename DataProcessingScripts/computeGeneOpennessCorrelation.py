# in this script we will compute the openness across 1Kb regions.  
# for each region we will take the measured openness times the number of bases overlapping the region
from __future__ import print_function

import numpy
import pandas
import sys

# read in data
openness_data = pandas.read_csv('./data/pairedData/human/Element_opn_uniq.txt', sep = '\t', header=None, index_col=0)
openness_data.index.names = ['chrom_regionStart_regionEnd']
print("openness data \n", file = sys.stderr)
print(openness_data.head(), file = sys.stderr)
print(openness_data.shape, file = sys.stderr)

gene_expression = pandas.read_csv('./data/pairedData/human/gene_ms.txt', sep = '\t', header=None, index_col = 0)
gene_expression.index.names = ['gene']
print("gene expression data\n", file = sys.stderr)
print(gene_expression.head(), file = sys.stderr)
print(gene_expression.shape, file = sys.stderr)

gene2regionDistances = pandas.read_csv('./data/pairedData/human/gene2regionDistances.txt', sep = '\t', header = 0)
print("gene to region distances\n", file = sys.stderr)
print(gene2regionDistances.head(), file = sys.stderr)
print(gene2regionDistances.shape, file = sys.stderr)

genes = list(gene_expression.index)
gene2regionDistances['regionName'] = gene2regionDistances['chrom'] + '_' + gene2regionDistances['regionStart'].map(str) + '_' + gene2regionDistances['regionEnd'].map(str)

import numpy
binnedOpenness = numpy.zeros((gene_expression.shape[0], 2000, 201)) # 1,000,000 divided into 1Kb regions
# I don't understand Python
#columnNames = ('+' + range(0, 999000, 1000) + ':' + range(1000, 1000000, 1000), '-' + range(0, 999000, 1000).map(str) + ':' + range(1000, 1000000, 1000).map(str)) 
#binnedOpennes = pandas.DataFrame(binnedOpennes, index = gene_expression.index.values, columns = columnNames)

for i in range(gene_expression.shape[0]):  # how to access rownames directly?  
#for i in range(10):
  print("i = ", i, file = sys.stderr)
  gene = genes[i]
  d = gene2regionDistances[(gene2regionDistances['gene'] == gene)] 
  if d.shape[0] > 0: # make sure there's at least one region
    TSS  = d['TSS'].iloc[0] # TSS is always the same, use first
    strand = 1 if d['strand'].iloc[0] == '+' else -1 # convert strand to value
    for j in range(0, 1000):
      #print("j = ", j, file = sys.stderr)
      # lowerLim and upperLim of bin depend on strand
      lowerLim = TSS + strand*1000*j if strand > 0 else TSS + strand*1000*(j + 1)  
      upperLim = TSS + strand*1000*(j + 1) if strand > 0 else TSS + strand*1000*j
      # get regions in the bin
      regions = d[numpy.logical_and(d['regionEnd'] > lowerLim, d['regionEnd'] < upperLim)] 
      # o is a place-holder for openness in region
      o = numpy.zeros((201, ))
      for r in range(regions.shape[0]):  # no iteration if regions.shape[0] == 0
        o_r = openness_data.loc[regions['regionName'].iloc[r], : ]
        o_r = o_r.values
        o_r.reshape(201, )
        # normalize by length of overlap
        overlap = (min(upperLim, regions['regionEnd'].iloc[r]) - max(lowerLim, regions['regionStart'].iloc[r]))/1000.0
        o += o_r.reshape(201,)*overlap
      binnedOpenness[i, j, :] = o
    for j in range(0, 1000):
      # other direction
      #print("j = ", j + 1000, file = sys.stderr)
      lowerLim = TSS - strand*1000*j if strand < 0 else TSS - strand*1000*(j + 1)
      upperLim = TSS - strand*1000*(j + 1) if strand < 0 else TSS - strand*1000*j
      regions = d[numpy.logical_and(d['regionEnd'] > lowerLim, d['regionEnd'] < upperLim)]
      o = numpy.zeros(201, )
      for r in range(regions.shape[0]):  # no iteration if regions.shape[0] == 0        
        o_r = openness_data.loc[regions['regionName'].iloc[r], : ]
        o_r = o_r.values
        o_r.reshape(201, )
        overlap =(min(upperLim, regions['regionEnd'].iloc[r]) - max(lowerLim, regions['regionStart'].iloc[r]))/1000.0
        o += o_r*overlap
      # 1000 for other direction
      binnedOpenness[i, j + 1000, :] = o

geneOpennnessCorrelation = numpy.zeros((binnedOpenness.shape[0], binnedOpenness.shape[1])) 
nObs = 201
for g in range(geneOpennnessCorrelation.shape[0]): # iterate over genes
  for r in range(geneOpennnessCorrelation.shape[1]): # iterate over regions
    x = gene_expression.iloc[g]
    y = binnedOpenness[g][r][:]
    if numpy.count_nonzero(y) > 0:
      geneOpennnessCorrelation[g][r] = numpy.corrcoef(x,y)[0][1]

numpy.savetxt('data/geneOpennnessCorrelation.txt', geneOpennnessCorrelation)
