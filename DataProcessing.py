import numpy as np
import pandas as pd
import pickle



def readEpigeneticDataFiles():
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

