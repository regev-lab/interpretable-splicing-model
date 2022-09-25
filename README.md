# interpretable-splicing-model
Scripts for preprocessing and training the interpretable splicing model. 

# Installation

No special hardware or GPU is required. A recent version of Python is required (version 3.8 tested). The following packages should be installed using "pip install", preferably under a virtual environment (tested version indicated):
1. tensorflow 2.10
2. numpy 1.22.4
3. pandas 1.5.0
4. joblib 1.2.0
5. sklearn 1.1.2

In addition, the following packages are required for generating figures:
1. matplotlib 3.6.0
2. seaborn 0.12.0
3. logomaker 0.8

Finally, the Vienna RNA package (version 2.4.17) should be installed and in the PATH. In Ubuntu, this can be done using "sudo apt install vienna-rna".

# Running 
Two scripts are provided:
1. preprocess.sh: This script takes the raw FASTQ files and converts them into a training and testing dataset. FASTQ files should be stored under the "fasta_files" folder, as explained in the readme.txt file. Typical running time is about 3 hours.
2. train_model.sh: This script reads the preprocessed datasets (the four pkl.gz files under the "data" folder), and trains the interpretable splicing model. Its output is the trained model, as well as two intermediate models generated as part of the custom training schedule. These files are stored in the "output" folder. Typical running time is about 2 hours. 

# Examples
The preprocessed datasets are provided in the "data" folder. Moreover, a trained model is included under the `output/' folder. This is the model used to generate all the figures in the paper. 

# Citation
Please cite: Liao SE, Sudarshan M, and Regev O. Machine learning for discovery: deciphering RNA splicing logic. In submission.
