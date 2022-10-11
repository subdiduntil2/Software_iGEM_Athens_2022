# Improve a part repository 

This Sub-repository includes the main .csv files and scripts in order to simulate the improve-a-part procedure that our team followed 
to generate the context indepedent RBS BBa_0034 variants

#Description

The sub-repository includes three main folders

1. The folder 'sequence_generator' which includes the random_seq_generator.ipynb script, which creates the random RBS sequences and modifies the initial dataset. It also exports all of the RBS sequences to Vienna_RNA_16s_bef.csv and Vienna_RNA_22s_bef.csv files, which will be used in the second folder.
2. The folder 'vienna_rna_modification' which includes the Vienna_RNA_script_final.py which saves the ensemble energy frequencies for all of the RBS sequences. It generates Vienna_RNA_22s_after.csv and Vienna_RNA_16s_after.csv which will be used by the third folder.
3. The 'Unsupervised Learning' folder where the results from files Vienna_RNA_22s_after.csv and Vienna_RNA_16s_after.csv are imported 
to the the modified_dataset_final.xlsx file via the create_final_dataset.ipynb script. Finally, we run the ML_pycaret_cls.ipynb script to perform the entire ML analysis and receive the resulting sequences.

#Installation

On each folder, we have included .sh files which include the Terminal commands ran to install the various packages.
For the Improve-A-Part repo, those files are biopython.sh, vienna_rna_cmds.sh and ml_libraries.sh  for the installation of the biopython, Vienna RNA and pycaret libraries respectivelly.

#Usage

The main results from the python scripts are the .csv and .xlsx files created on each folder as described above and which have already been imported manually by us for exhibition. In terms of the ML classification, the resulted output sequences can be view on the final cell. 

