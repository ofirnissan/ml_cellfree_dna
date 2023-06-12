# ml_cellfree_dna

# Requirements:
clone https://github.com/jerryji1993/DNABERT/tree/master
then:
install wsl
pip install -r req.txt

# Data:
Data files are the BAM files.
parse_data.py generates new tsv file that the DNABERT can work with, based on chosen data samples and metadata.

# For regression:
Right now the regression script just run the linear regression model on a tsv files (train + test). 
Meaning - for now, the regression script run on the raw data instead of the BERT output.
Script Requirments: you will need database with sequeces and labels and a vocabulary file.
