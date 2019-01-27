# Epigenetic_Deep_Learning
Using Deep Learning to predict gene expression from epigenetic information


To start (assumes you have conda), run 'conda env create -f environment.yml' to create the environment from the .yml file. 
After running the above command once, you can start the environment with 'source activate deeplearning'

Data Files:
Element_opn.txt 
- Chromatin openness data. Index = Chromatin position. Columns = Cell Types. Shape = 184665, 201
- values = Measure of Chromatin openness

gene_ms.txt
- Gene to Protein Expression. Index = genes. Columns = Cell Types. Shape = 17794,201
- values = protein expression level

RE_TG.txt
- unclear 
