import DataProcessing as dp
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nn as nnpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    # dp.ConvertGeneDistanceFile(data_df, distanceFile, gene_ms, numFeatures, max_distance)
    finalX, finalY = pickle.load( open('./pickle/FinalData_%d_%d.p' %(numFeatures,max_distance), "rb" ) )

    net = nnpy.Net(numFeatures)
    print(net)


    input = torch.tensor(finalX.values.transpose())
    target = torch.tensor(finalY.values.transpose()).float()
    loss_criterion = nn.MSELoss()
    for i in range(1000):
        optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
        optimizer.zero_grad()
        net_out = net.forward(input)
        loss = loss_criterion(target,net_out)
        loss.backward()
        optimizer.step()
        print(loss.item())



main()
