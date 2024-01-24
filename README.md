# SFINN:inferring gene regulatory networks from single-cell expression data with shared factor neighborhood and integrated neural networks

## 1. Introduction

SFINN is a gene regulatory network construction algorithm based on shared factor neighborhoods and integrated neural networks. It is applicable to both single-cell and spatial transcriptomic gene expression data for inferring potential interactions between genes. 

For single-cell data, SFINN uses a cell neighborhood graph generated from combined gene expression data and a shared factor neighborhood strategy as input for the integrated neural network. In the case of spatial transcriptomic data, the main difference lies in the handling of the cell neighborhood graph. This involves computing the Euclidean distance and similarity transformation of spatial position information and then XOR-fusing it with the network graph generated from spatial gene expression data based on the shared factor neighborhood strategy. The two view inputs are processed through neural network operations, successfully transforming the inference of potential interactions between gene pairs into a binary classification problem.

The datasets analyzed in the paper are available at: https://zenodo.org/records/10558871

## 2.Dependencies

    Python == 3.6.2 
    Pytorch == 2.6.2
    tensorflow == 2.6.2
    tensorflow-gpu == 2.6.0
    scikit-learn == 0.24.2
    spektral == 1.2.0
    pandas == 1.1.5
    numpy == 1.19.5
 ## 3.TASK 1, Evaluate SFINN on eight single-cell datasets
### 3.1 Prepare data
    # If you have a ground truth data list with pairs like (gene a, x1) labeled as (0, 1, 2), please use the following code for processing to generate a file that divides the candidate gene pairs into interval indices:

    python GenerateSplitIndexFile.py --GroundTrueFilePath /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/NeedDataFile/bonemarrow_TF_target_gene_pairs.txt --SplitIndexSavePath /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/NeedDataFile/bonemarrow_TF_target_gene_pairs_d2  
    
    # If you want to divide the dataset into a training set and a test set, please use:
    
    python Dataset_Split.py --TFlength 13 --TrainDataSaveDir /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/train_data --TestDataSaveDir /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/test_data --UnionGenePairExpDataSavePath /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/UnionGenePariExpData

    # Generate joint representation data for gene pairs, please use:

    python GetUnionGenePairExpData.py  --UnionGenePairExpDataSavePath  /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/UnionGenePariExpData --ExpDataSavePath /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/NeedDataFile/bonemarrow_cell.h5 --GeneNameFilesSave_Dir  --SplitIndexFilesSave_Dir /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/NeedDataFile/bonemarrow_TF_target_gene_pairs_d2  --GroundTrueSave_Dir /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/NeedDataFile/bonemarrow_TF_target_gene_pairs.txt  --IsBoneOrDend True

    # Retrieve the cell-cell adjacency matrix generated based on the shared factor neighborhood strategy.

    python GetSSAdjmatrix.py --TFlength 13  --ExpDataSavePath /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/NeedDataFile/bonemarrow_cell.h5 --GeneNameFilesSave_Dir /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/NeedDataFile/sc_gene_list.txt --AdjDataSave_Dir /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/AdjData --UnionGenePairExpDataSavePath /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/UnionGenePariExpData  --IsBoneOrDend  True --ExpDataFileisH5orCsv  h5


    


### 3.2 Command to run SFINN
    Parameters:
    TFlength:Number of transcription factors
    AdjDataSave_Dir:Deposit paths for cell-cell adjacency matrices
    ModelResSavePath:Model Storage Path
    UnionGenePairExpDataSavePath:Deposit paths for gene union representation pairs
    TrainDataLoadDir:Storage path for training dataset
    TestDataLoadDir:Storage path for testing dataset

    python SFINN.py --TFlength 13 --AdjDataSave_Dir /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/AdjData --ModelResSavePath /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/ModelRes [--Epochs] 200   [--Lr] 1e-6 [--L2_Reg] 5e-4 [--Es_Patience] 100 [--Batch_Size] 512 --UnionGenePairExpDataSavePath /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/UnionGenePariExpData  --TrainDataLoadDir  /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/train_data --TestDataLoadDir  /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/test_data


                   
### 3.3 Evaluation and visualization
    Parameters:
    TFlength:Number of transcription factors
    AdjDataSave_Dir:Deposit paths for cell-cell adjacency matrices
    ModelResSavePath:Model Storage Path
    UnionGenePairExpDataSavePath:Deposit paths for gene union representation pairs
    TrainDataLoadDir:Storage path for training dataset
    TestDataLoadDir:Storage path for testing dataset

    python SavePrResPic.py  --TFlength 13 --Epochs 200 --ModelResSave_Dir /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/ModelRes --PrResPicSave_Dir /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/PR_Res --UnionGenePairExpDataSavePath /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqDatabone/bone/UnionGenePariExpData
    python SaveRocResPic.py  --TFlength 13 --Epochs 200 --ModelResSave_Dir /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/ModelRes  --RocResPicSave_Dir x/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/ROC_Res --UnionGenePairExpDataSavePath /home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/UnionGenePariExpData
 ## 4.TASK 2, Evaluate SFINN on five spatial transcriptome datasets
### 4.1 Prepare data
    # Generate input data, please use:

    Parameters:

    TFlength:Number of transcription factors
    UnionGenePairExpDataSavePath:Deposit paths for gene union representation pairs
    ExpDataSavePath:Gene expression data file storage location
    SplitIndexFilesSave_Dir:Gene Pair List Interval Index Division File
    GroundTrueSave_Dir:Groundtrue storage location
    LocaldataPath:Path to the spatial location data file
    DatasetName:The name of the dataset. For example, "seqFISH", "MERFISH","ST_SCC_P2_1".

    python GetUnionGenePairExpData.py  --UnionGenePairExpDataSavePath  /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_dataGenerate/seqFISH  --ExpDataSavePath /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/cortex_svz_counts.csv --SplitIndexFilesSave_Dir /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/L_Rpairs_split_index.txt  --GroundTrueSave_Dir /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/L_Rpairs.txt --LocaldataPath /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/cortex_svz_cellcentroids.csv --DatasetName seqFISH
    python GetSSAdjmatrix.py --TFlength 286 --ExpDataSavePath /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/cortex_svz_counts.csv  --AdjDataSave_Dir /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/Adajdata/seqFISH --UnionGenePairExpDataSavePath /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_dataGenerate/seqFISH  --LocaldataPath /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/cortex_svz_cellcentroids.csv --DatasetName seqFISH

### 4.2 Command to run SFINN
    python SFINN.py --TFlength 286 --AdjDataSave_Dir /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/Adajdata/seqFISH --ModelResSavePath /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_result/seqFISH [--Epochs] 100   [--Lr] 1e-6 [--L2_Reg] 5e-7 [--Es_Patience] 50 [--Batch_Size] 32 --UnionGenePairExpDataSavePath /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_dataGenerate/seqFISH


### 4.3 Evaluation and visualization
    python SavePrResPic.py  --TFlength 286 --Epochs 100--ModelResSave_Dir /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_result/seqFISH  --PrResPicSave_Dir /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_result/seqFISH_pr_res --UnionGenePairExpDataSavePath /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_dataGenerate/seqFISH
    python SaveRocResPic.py  --TFlength 286 --Epochs 100 --ModelResSave_Dir /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_result/seqFISH  --RocResPicSave_Dir /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_result/seqFISH_roc_res  --UnionGenePairExpDataSavePath /home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_dataGenerate/seqFISH
                   
