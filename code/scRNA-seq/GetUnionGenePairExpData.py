'''
功能：生成基因对联合表达数据。
调用：python GetUnionGenePairExpData.py  --UnionGenePairExpDataSavePath  xx  --ExpDataSavePath xx --GeneNameFilesSave_Dir xx --SplitIndexFilesSave_Dir xx  --GroundTrueSave_Dir xx  --IsBoneOrDend True/False 
参数:
--UnionGenePairExpDataSavePath  基因对联合表达数据输出路径
--ExpDataSavePath  数据集所在路径
--GeneNameFilesSave_Dir  基因名称文件所在路径
--SplitIndexFilesSave_Dir  基因对区间索引划分文件所在路径
--GroundTrueSave_Dir 真实网络对所在路径
--IsBoneOrDend  是否为骨髓驱动的巨噬细胞或树突状细胞数据
'''
import pandas as pd
from numpy import *
import json, re,os, sys
from sklearn import preprocessing
import numpy as np
import random
seed_value= 1234
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
import argparse 

parse = argparse .ArgumentParser()
parse.add_argument('--UnionGenePairExpDataSavePath',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/mHSC-GM/UnionGenePariExpData',help='Syndicated gene pair expression data output pathway')
parse.add_argument('--ExpDataSavePath',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/mHSC-GM/NeedDataFile/ExpressionData.csv',help='The path in which the gene expression data is located')
parse.add_argument('--GeneNameFilesSave_Dir',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/mHSC-GM/NeedDataFile/mHSC-GM_geneName_map.txt',help='The path where the gene name file is located')
parse.add_argument('--SplitIndexFilesSave_Dir',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/mHSC-GM/NeedDataFile/split_for_mHSC-GM.txt',help='Delineation of gene index interval files')
parse.add_argument('--GroundTrueSave_Dir',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/mHSC-GM/NeedDataFile/training_formHSC-GM.txt',help='Candidate gene pair list file')
parse.add_argument('--IsBoneOrDend',type=str,default='True',help='Whether it is a local bonemarrow macrophage or dendritic cell dataset')
parse.add_argument('--ExpDataFileisH5orCsv',type=str,default='h5',help='Whether it is an H5 or CSV format file')
args = parse.parse_args()

UnionGenePairExpDataSavePath=args.UnionGenePairExpDataSavePath
ExpDataSavePath=args.ExpDataSavePath
GeneNameFilesSave_Dir=args.GeneNameFilesSave_Dir
SplitIndexFilesSave_Dir=args.SplitIndexFilesSave_Dir
GroundTrueSave_Dir=args.GroundTrueSave_Dir
ExpDataFileisH5orCsv=args.ExpDataFileisH5orCsv
IsBoneOrDend=args.IsBoneOrDend

if not os.path.isdir(UnionGenePairExpDataSavePath):
        os.makedirs(UnionGenePairExpDataSavePath)
        
def GeneNameList(file_path):
    import re
    h={}
    s = open(file_path,'r') #gene symbol ID list of sc RNA-seq
    for line in s:
        search_result = re.search(r'^([^\s]+)\s+([^\s]+)',line)
        h[search_result.group(1).lower()]=search_result.group(2) # h [gene symbol] = gene ID
    s.close()
    return h

def GetSeprationIndex(file_path):
    import numpy as np
    index_list = []
    s = open(file_path, 'r')
    for line in s:
        index_list.append(int(line))
    return (np.array(index_list))

GeneName_List =GeneNameList(GeneNameFilesSave_Dir) # 'sc_gene_list.txt')#

########## 读入基因对信息
gene_pair_label = []
s=open(GroundTrueSave_Dir)#'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
for line in s:
    gene_pair_label.append(line)

#########读取基因对列表索引信息
gene_pair_index = GetSeprationIndex(SplitIndexFilesSave_Dir)#sys.argv[6]) # read file speration index
s.close()

gene_pair_label_array = array(gene_pair_label) 
# ########### 获取基因表达数据
if ExpDataFileisH5orCsv == 'h5':
    store = pd.HDFStore(ExpDataSavePath)#'/home/yey3/sc_process_1/rank_total_gene_rpkm.h5')    # scRNA-seq expression data                        )#
    rpkm = store['/RPKMs']
    store.close()
    print('read h5 sc RNA-seq expression')
elif ExpDataFileisH5orCsv == 'csv':
    
    rpkm= pd.read_csv(ExpDataSavePath,header='infer',index_col=0)
    rpkm = rpkm.T
    gene_id=rpkm.columns.to_list()
    gene_id=[gene_id[i].lower() for i in range(len(gene_id))]
    rpkm.columns=gene_id
    print('read csv sc RNA-seq expression')
else:
    print('Error: ExpDataFileisH5orCsv is not h5 or csv')

#########产生基因对表达矩阵
for i in range(len(gene_pair_index)-1):   #### many sperations
    print (i)
    start_index = gene_pair_index[i]
    end_index = gene_pair_index[i+1]
    x = []
    y = []
    z = []
    for gene_pair in gene_pair_label_array[start_index:end_index]: ## each speration
        separation = gene_pair.split()
        
        x_gene_name,y_gene_name,label = separation[0],separation[1],separation[2]
        y.append(label)
        z.append(x_gene_name+'\t'+y_gene_name)
        if IsBoneOrDend=='True':
           
            cell_LR_expression = np.array(rpkm[[int(GeneName_List[x_gene_name]),int(GeneName_List[y_gene_name])]])
        elif IsBoneOrDend=='False':
          
            cell_LR_expression = np.array(rpkm[[(GeneName_List[x_gene_name]),(GeneName_List[y_gene_name])]])
        else:
             print('IsBoneOrDend is not True or False')  
        cell_LR_expression=log10( cell_LR_expression+10**-4)
        x.append(cell_LR_expression)
      
    if (len(x)>0):
        xx = array(x)[:, :, :]

    save(UnionGenePairExpDataSavePath+'/Nxdata_tf' + str(i) + '.npy', xx)
    save(UnionGenePairExpDataSavePath+'/ydata_tf' + str(i) + '.npy', array(y))
    save(UnionGenePairExpDataSavePath+'/zdata_tf' + str(i) + '.npy', array(z))