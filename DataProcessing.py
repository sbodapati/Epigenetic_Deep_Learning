import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt



def ReadEpigeneticDataFiles():
    print('starting to read raw data files')

    # A list of the chromatin locations. Shape = 184665, 1
    index_names = pd.read_csv('./data/pairedData/human/Element_name.txt', header=None)
    index_names = index_names.rename(columns={0: "chrom_position"})
    # print(index_names.head())
    # print(index_names.shape)

    # Chromatin openness data. Index = Chromatin position. Columns = Cell Types. Shape = 184665, 201
    # values = Measure of Chromatin openness
    data_df = pd.read_csv('./data/pairedData/human/Element_opn.txt', sep = '\t', header=None, index_col=False)
    data_df = data_df.set_index(0)
    data_df.index.names = ['chrom_pos']
    # print(data_df.head())
    # print(data_df.shape)


    # unclear
    PosToGeneMap = pd.read_csv('./data/pairedData/human/RE_TG.txt', sep = '\t', header=None)
    # print(PosToGeneMap.head())


    # Gene to Protein Expression. Index = genes. Columns = Cell Types. Shape = 17794,201
    # values = protein expression level
    gene_ms = pd.read_csv('./data/pairedData/human/gene_ms.txt', sep = '\t', header=None)
    gene_ms = gene_ms.set_index(0)
    gene_ms.index.names = ['gene']
    # print(gene_ms.head())
    # print(gene_ms.shape)

    pickle.dump((data_df, gene_ms, index_names), open( './pickle/raw_data_files.p', "wb" ))
    print('finished creating raw data pickle files')

def ReadGeneDistanceFile():
    print('starting to read distance file')
    distanceFile = pd.read_csv('./data/gene2regionDistances.txt', sep = '\t', header=0)
    pickle.dump(distanceFile, open( './pickle/distance_file.p', "wb" ))
    print('finished reading distance file')

def ConvertGeneDistanceFile(data_df, distanceFile, gene_ms, numFeatures, maxDistance):

    # print(data_df.head())
    # l = data_df.index.values
    # l = [i.split('_',2) for i in l]
    # chrom = [i[0] for i in l]
    # start = [i[1] for i in l]
    # end = [i[2] for i in l]

    geneList= gene_ms.index.values
    distanceFile['naming_conv'] = distanceFile['chrom'] + '_' + distanceFile['regionStart'].map(str) + '_' + distanceFile['regionEnd'].map(str)
    indexNamesX = np.arange(1,numFeatures+1,1)
    indexNamesY = ['expression']
    finalX = pd.DataFrame(index=indexNamesX)
    finalY = pd.DataFrame(index=indexNamesY)

    for i in range(1): #geneList
        gene = geneList[i]
        geneDistance= distanceFile[(distanceFile.loc[:, 'gene'] == gene) & (distanceFile.loc[:, 'gene2regionDistance']<maxDistance)]
        if geneDistance.shape[0] > numFeatures:
            geneDistance.sort_values(by='gene2regionDistance', inplace=True)
            tempX = data_df[data_df.index.isin(geneDistance.iloc[0:numFeatures,:]['naming_conv'])].reset_index()
            print(tempX.head())
            
            # print(gene_ms[gene_ms.index.isin([gene])].set_index(indexNamesY ))




    # print(data_df.head())
    # print(distanceFile.head())
    # print(gene_ms.head())


    # distanceHist = set()
    # print(distanceFile.head())
    #
    # sns.distplot(distanceFile.groupby(['gene'])['ones'].sum())
    # plt.show()
    # for gene in geneList:
    #     geneDistance= distanceFile[distanceFile.loc[:,'gene'] == gene]
    #     distanceHist.add(geneDistance.shape[0])

    # sns.distplot(distanceHist)
    # plt.show()
    #
    # print(start[0:5])
    # print(end[0:5])
    # print(distanceFile.head())
    # print(distanceFile[ (distanceFile.loc[:,'regionStart'] == int(start[0])) & (distanceFile.loc[:,'regionEnd'] == int(end[0]))])
    # print(data_df.head())
    # LOXL4 = distanceFile[distanceFile.loc[:,'gene'] == 'LOXL4']
    # print(LOXL4.head())
    # print(LOXL4.sort_index(by = 'gene2regionDistance'))

