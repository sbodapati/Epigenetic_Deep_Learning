import DataProcessing as dp
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def main():

    numFeatures = 100
    max_distance = 1000000

    if not os.path.isfile('./pickle/raw_data_files.p'):
        dp.ReadEpigeneticDataFiles()
    (data_df, gene_ms, index_names) = pickle.load( open( './pickle/raw_data_files.p', "rb" ) )

    if not os.path.isfile('./pickle/distance_file.p'):
        dp.ReadGeneDistanceFile()
    distanceFile = pickle.load( open( './pickle/distance_file.p', "rb" ) )

    if not os.path.isfile('./pickle/FinalData_%d_%d.p' %(numFeatures,max_distance)):
        dp.ConvertGeneDistanceFile(data_df, distanceFile, gene_ms, numFeatures, max_distance)
    finalX, finalY = pickle.load( open('./pickle/FinalData_%d_%d.p' %(numFeatures,max_distance), "rb" ) )

main()
