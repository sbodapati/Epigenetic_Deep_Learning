import DataProcessing as dp
import os
import pickle


def main():

    if not os.path.isfile('./pickle/raw_data_files.p'):
        dp.readEpigeneticDataFiles()

    (data_df, gene_ms, index_names) = pickle.load( open( './pickle/raw_data_files.p', "rb" ) )
    


main()
