
import pandas as pd
from numpy import *
import json, re,os, sys
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from scipy.spatial.distance import pdist, squareform

from pandas import DataFrame
import matplotlib.pyplot as plt
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
parse.add_argument('--TFlength',type=int,default=18,help='The number of transcription factors')
parse.add_argument('--ExpDataSavePath',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/NeedDataFile/bonemarrow_cell.h5',help='The path in which the gene expression data is located')
parse.add_argument('--GeneNameFilesSave_Dir',type=str,help='The path where the gene name file is located')
parse.add_argument('--AdjDataSave_Dir',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/AdjData',help='Adjacency matrix output path')
parse.add_argument('--UnionGenePairExpDataSavePath',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/UnionGenePariExpData',help='The path in which the joint gene pair expression data is located')
parse.add_argument('--IsBoneOrDend',type=str,default='True',help='Whether it is a local bonemarrow macrophage or dendritic cell dataset')
parse.add_argument('--ExpDataFileisH5orCsv',type=str,default='h5',help='Whether it is an H5 or CSV format file')
args = parse.parse_args()

TFlength =args.TFlength # number of data parts divided
UnionGenePairExpDataSavePath = args.UnionGenePairExpDataSavePath
ExpDataSavePath=args.ExpDataSavePath
AdjDataSave_Dir=args.AdjDataSave_Dir
IsBoneOrDend=args.IsBoneOrDend
ExpDataFileisH5orCsv=args.ExpDataFileisH5orCsv
GeneNameFilesSave_Dir=args.GeneNameFilesSave_Dir


num_classes = 2
whole_data_TF = [i for i in range(TFlength )]

if not os.path.isdir(AdjDataSave_Dir):
    os.makedirs(AdjDataSave_Dir)

def read_H5genemaplist():
    sc_gene_list=pd.read_csv(GeneNameFilesSave_Dir,header=None,sep='\t')
    return sc_gene_list[0]

if ExpDataFileisH5orCsv == 'h5':
    store = pd.HDFStore(ExpDataSavePath)#'/home/yey3/sc_process_1/rank_total_gene_rpkm.h5')    # scRNA-seq expression data                        )#
    rpkm = store['/RPKMs']
    store.close()
    print('read h5 sc RNA-seq expression')
    if IsBoneOrDend=='True' :
        genename_list=read_H5genemaplist()
        rpkm.columns=genename_list
        print('read IsBoneOrDend')
    elif IsBoneOrDend=='False':
        genename_list=read_H5genemaplist()
        rpkm=rpkm[genename_list]
        rpkm.columns=genename_list
        print('read Is Not BoneOrDend')
    else:
        print('IsBoneOrDend is not True or False')
elif ExpDataFileisH5orCsv == 'csv':
    
    rpkm= pd.read_csv(ExpDataSavePath,header='infer',index_col=0)
    rpkm = rpkm.T
    gene_id=rpkm.columns.to_list()
    gene_id=[gene_id[i].lower() for i in range(len(gene_id))]
    rpkm.columns=gene_id
    print('read csv sc RNA-seq expression')
else:
    print('Error: ExpDataFileisH5orCsv is not h5 or csv')


def GenesInvolvedInTrainingSet(indel_list,data_path): 
    import random
    import numpy as np
    import pandas as pd

    zzdata = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:#len(h_tf_sc)):
       
        ydata = np.load(data_path+'/ydata_tf' + str(i) + '.npy')
        zdata = np.load(data_path+'/zdata_tf' + str(i) + '.npy')
       
        for k in range(int(len(ydata))):
           
            yydata.append(ydata[k])
            zzdata.append(zdata[k].split())
        count_setx = count_setx + int(len(ydata))
        count_set.append(count_setx)
        print (i,len(ydata)) 
    Gp=pd.DataFrame(zzdata)
    all_train_gene=Gp[0].unique().tolist()+Gp[1].unique().tolist()

    print('len(Gp[1].unique().tolist())',len(Gp[1].unique().tolist()),'len(Gp[0].unique().tolist())',len(Gp[0].unique().tolist()),Gp[0].unique().tolist())
    all_train_gene=list(set(all_train_gene))
    print('len(all_train_gene)',len(all_train_gene))
    # print (np.array(yydata).shape)
    return(all_train_gene,count_set,Gp[0].unique().tolist())

def Findmax(nparr):
    c=nparr.tolist()
    return c.index(max(c))

def counter(Array):
    return Counter(arr_list_x)

def presbar(count): 
    plt.style.use('seaborn-darkgrid')
    plt.bar(list(res.keys()), res.values(), width=0.9, color='g')
    plt.ylabel('Number of tags for sample K neighbors')
    plt.xlabel('f('+str(count+1)+')')
    plt.title('FN('+str(count+1)+')')
    plt.xticks(np.arange(1,pca_reduction_scaled.shape[1]+1))
    plt.show()
    return 
  
def getvector(dict_y,i):
    vectors=[0]*pca_reduction_scaled.shape[1]
    for keys_k,values_v in dict_y:
        vectors[keys_k-1]=values_v
    FN.append(vectors)
    return    

for test_indel in range(1,4): ################## three fold cross validation
    test_TF = [i for i in range (int(np.ceil((test_indel-1)*0.333333*TFlength)),int(np.ceil(test_indel*0.333333*TFlength)))]
    
    train_TF = [i for i in whole_data_TF if i not in test_TF]

    (train_gene,count_set,tfs)= GenesInvolvedInTrainingSet(train_TF,UnionGenePairExpDataSavePath)
    data_train=rpkm[train_gene]
    print(data_train.shape)
  
    Genexp_mat=np.mat(data_train)
    pca=PCA(n_components=0.9,svd_solver='full')
    pca.fit(Genexp_mat)
    pca_reduction=pca.transform(Genexp_mat)
    # pca_reduction=Genexp_mat
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(pca_reduction)
    distances, indices = nbrs.kneighbors(pca_reduction)

    scaler = preprocessing.StandardScaler().fit(pca_reduction)
    pca_reduction_scaled = scaler.transform(pca_reduction)
    print('after reduce data shape',pca_reduction_scaled.shape)
    labels=['factor'+str(x) for x in range(0,pca_reduction_scaled.shape[1])]
    df_one=DataFrame(pca_reduction_scaled)
    df_one['lables']='factor'

    for i in range(pca_reduction_scaled.shape[0]):
        z=Findmax(pca_reduction_scaled[i])
        df_one.loc[i,'lables']='f('+str(z)+')'
        

    maxgene=np.argmax(pca_reduction_scaled,axis=1)+1
    maxgene_list=list(maxgene)
    indices_list=list(range(pca_reduction_scaled.shape[0]))

    dict_x={indices_list[i]:maxgene_list[i] for i in range(pca_reduction_scaled.shape[0])}

    data_x=[]
    list_x=[]
    arr_list_x=[]
    value_list=[]
    FN=[]

    for i in range(pca_reduction_scaled.shape[0]):
        p1 = {key:value for key, value in dict_x.items() if key in indices[i]}
        value_x=p1.values()
        value_list=list(value_x)
        arr_list_x=np.array(value_list)
        res=counter(value_list)
        getvector(res.items(),i)
        # presbar(i)
        # print('FN('+str(i+1)+")=",FN[i])

    X=np.array(FN)
    Y = pdist(X, 'cityblock')


    distance_matrix = squareform(Y)
    distance_matrixs=1/(1+distance_matrix)
    distsim_to_thre = np.where(distance_matrixs>0.5, 1, 0)
    # distance_matrixs=np.where(distance_matrix<10,1,0)
    np.save(AdjDataSave_Dir+'/'+str(test_indel)+'foldadj.npy', distsim_to_thre)
