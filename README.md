# Epigenetic_Deep_Learning
Using Deep Learning to predict gene expression from epigenetic information

##Initial Startup
To start (assumes you have conda):
* run 'conda env create -f environment.yml' to create the environment from the .yml file. After running the above command once, you can start the environment with 'source activate deeplearning'
* create a './data' directory. 
* Move the 'pairedData.tar.gz' file into this './data/' directory and unzip it. You can then delete the .tar.gz file.\
.gitignore has been setup to not read from /data and /pickle
*  you should now have a directory that has './data/pairedData/.'
* run 'python Analysis.py'

##Data Files:\
Element_name.txt
- A list of the chromatin locations. Shape = 184665, 1 

Element_opn.txt 
- Chromatin openness data. Index = Chromatin position. Columns = Cell Types. Shape = 184665, 201
- values = Measure of Chromatin openness

gene_ms.txt
- Gene to Protein Expression. Index = genes. Columns = Cell Types. Shape = 17794,201
- values = protein expression level

RE_TG.txt
- unclear, appears to be some mapping of chromatin position to gene (1 to many)

sample_201_new
- list of all 201 cell types 
