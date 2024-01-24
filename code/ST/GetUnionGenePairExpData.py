import pandas as pd
from numpy import *
import json, re,os, sys
from sklearn import preprocessing
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
seed_value= 1234
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
import argparse 
parse = argparse .ArgumentParser()
parse.add_argument('--UnionGenePairExpDataSavePath',type=str,default='/home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_dataGenerate/seqFISH',help='Syndicated gene pair expression data output pathway')
parse.add_argument('--ExpDataSavePath',type=str,default='/home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/cortex_svz_counts.csv',help='The path in which the gene expression data is located')
parse.add_argument('--SplitIndexFilesSave_Dir',type=str,default='/home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/L_Rpairs_split_index.txt',help='Delineation of gene index interval files')
parse.add_argument('--GroundTrueSave_Dir',type=str,default='/home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/L_Rpairs.txt',help='Candidate gene pair list file')
parse.add_argument('--LocaldataPath',type=str,default='/home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/cortex_svz_cellcentroids.csv',help='Spatial location information data loading path')
parse.add_argument('--DatasetName',type=str,default='seqFISH',help='DatasetName')
args = parse.parse_args()

UnionGenePairExpDataSavePath=args.UnionGenePairExpDataSavePath
ExpDataSavePath=args.ExpDataSavePath
SplitIndexFilesSave_Dir=args.SplitIndexFilesSave_Dir
GroundTrueSave_Dir=args.GroundTrueSave_Dir
DatasetName=args.DatasetName
LocaldataPath=args.LocaldataPath

if not os.path.isdir(UnionGenePairExpDataSavePath):
        os.makedirs(UnionGenePairExpDataSavePath)
        
def GetSeprationIndex (file_name):
    import numpy as np
    index_list = []
    s = open(file_name, 'r')
    for line in s:
        index_list.append(int(line))
    return (np.array(index_list))

if DatasetName=='seqFISH':
    print('The seqFISH dataset is loaded!')
    cortex_svz_counts = pd.read_csv(ExpDataSavePath)
    cortex_svz_counts_N =cortex_svz_counts.div(cortex_svz_counts.sum(axis=1)+1, axis='rows')*10**4
    cortex_svz_counts_N.columns =[i.lower() for i in list(cortex_svz_counts_N)] ## gene expression normalization
    cortex_svz_cellcentroids = pd.read_csv(LocaldataPath)
elif DatasetName=='MERFISH':
    print('The MERFISH dataset is loaded!')
    cortex_svz_counts=pd.read_csv(ExpDataSavePath,header=0,index_col=0)
    cortex_svz_counts_N =cortex_svz_counts.div(cortex_svz_counts.sum(axis=1)+1, axis='rows')*10**4
    cortex_svz_counts_N.columns =[i.lower() for i in list(cortex_svz_counts_N)]
    cortex_svz_cellcentroids = pd.read_excel(LocaldataPath) 
elif DatasetName=='ST_SCC_P2_1' or DatasetName=='ST_SCC_P2_2' or DatasetName=='ST_SCC_P2_3':
    print('The ST_SCC_P2 is loaded!')
    cortex_svz_counts = pd.read_csv(ExpDataSavePath)
    cortex_svz_counts=cortex_svz_counts.T
    cortex_svz_counts_N =cortex_svz_counts.div(cortex_svz_counts.sum(axis=1)+1, axis='rows')*10**4
    cortex_svz_counts_N.columns =[i.lower() for i in list(cortex_svz_counts_N)]
    cortex_svz_cellcentroids = pd.read_csv(LocaldataPath)         
else:
    print('DatasetName input error ')    


gene_pair_label = []
s=open(GroundTrueSave_Dir)#'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
for line in s:
    gene_pair_label.append(line)


gene_pair_index = GetSeprationIndex(SplitIndexFilesSave_Dir)#sys.argv[6]) # read file speration index
s.close()

####generate gene pair
gene_pair_label_array = array(gene_pair_label) 
for i in range(len(gene_pair_index)-1):   
    print (i)
    start_index = gene_pair_index[i]
    end_index = gene_pair_index[i+1]
    x = []
    y = []
    z = []
    for gene_pair in gene_pair_label_array[start_index:end_index]: ## each speration
        separation = gene_pair.split()
    
        ligand,receptor,label = separation[0],separation[1],separation[2]
        y.append(label)
        z.append(ligand+'\t'+receptor)
        cell_LR_expression = np.array(cortex_svz_counts_N[[ligand, receptor]]) # postive sample
        x.append(cell_LR_expression)
        xx = np.array(x)


    save(UnionGenePairExpDataSavePath+'/Nxdata_tf' + str(i) + '.npy', xx)
    save(UnionGenePairExpDataSavePath+'/ydata_tf' + str(i) + '.npy', array(y))
    save(UnionGenePairExpDataSavePath+'/zdata_tf' + str(i) + '.npy', array(z))
    
    
    
    
    
    
    
 
