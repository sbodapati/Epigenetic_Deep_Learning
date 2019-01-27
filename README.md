# Epigenetic_Deep_Learning
Using Deep Learning to predict gene expression from epigenetic information


To start (assumes you have conda), run 'conda env create -f environment.yml' to create the environment from the .yml file. 
After running the above command once, you can start the environment with 'source activate deeplearning'

Data Files:
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
