# in this script we will compute the openness across 1Kb regions.  
# for each region we will take the measured openness times the number of bases overlapping the region
from __future__ import print_function

import numpy
import pandas
import sys

with open("openness_out.txt", "w") as f:
  print("starting", file = f)
# read in data
openness_data = pandas.read_csv('./data/pairedData/human/Element_opn_uniq.txt', sep = '\t', header=None, index_col=0)
openness_data.index.names = ['chrom_regionStart_regionEnd']
print(openness_data.head(), file = sys.stderr)
with open("openness_out.txt", "a") as f:
  print("openness data shape = ", openness_data.shape, file = f)

gene_expression = pandas.read_csv('./data/pairedData/human/gene_ms.txt', sep = '\t', header=None, index_col = 0)
gene_expression.index.names = ['gene']
print(gene_expression.head(), file = sys.stderr)
with open("openness_out.txt", "a") as f:
  print("gene expression data shape = ", gene_expression.shape, file = f)

gene2regionDistances = pandas.read_csv('./data/pairedData/human/gene2regionDistances.txt', sep = '\t', header = 0)
print(gene2regionDistances.head(), file = sys.stderr)
with open("openness_out.txt", "a") as f:
  print("gene2region distances = ", gene2regionDistances.shape, file = f)

genes = list(gene_expression.index)
gene2regionDistances['regionName'] = gene2regionDistances['chrom'] + '_' + gene2regionDistances['regionStart'].map(str) + '_' + gene2regionDistances['regionEnd'].map(str)

import numpy
binnedOpenness = numpy.zeros((gene_expression.shape[0]*201, 2000)) # 1,000,000 divided into 1Kb regions
# I don't understand Python
#columnNames = ('+' + range(0, 999000, 1000) + ':' + range(1000, 1000000, 1000), '-' + range(0, 999000, 1000).map(str) + ':' + range(1000, 1000000, 1000).map(str)) 
#binnedOpennes = pandas.DataFrame(binnedOpennes, index = gene_expression.index.values, columns = columnNames)

for i in range(gene_expression.shape[0]):  # how to access rownames directly?  
#for i in range(10):
  with open("openness_out.txt", "a") as f:
    print("i = ", i, file = f)
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
      for k in range(201):
        if o[k] > 100:
          with open("openness_out.txt", "a") as f:
            print("outlier detected\t", o[k], "\t", gene, "\n", regions, file = f)
        binnedOpenness[i*201 + k][j] = o[k]
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
      for k in range(201):
        if o[k] > 100:
          with open("openness_out.txt", "a") as f:
            print("outlier detected\t", o[k], "\t", gene, "\n", regions, file = f)
        binnedOpenness[i*201 + k][1000 + j] = o[k]

numpy.save('data/binnedOpennessReshaped.npy', binnedOpenness)
