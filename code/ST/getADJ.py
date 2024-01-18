import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, re,os, sys
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy import sparse
import pickle
import spektral
import scipy.linalg
from pandas import DataFrame
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
parse.add_argument('--TFlength',type=int,default=286,help='The number of transcription factors')
parse.add_argument('--ExpDataSavePath',type=str,default='/home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/cortex_svz_counts.csv',help='The path in which the gene expression data is located')
parse.add_argument('--AdjDataSave_Dir',type=str,default='/home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/Adajdata/seqFISH',help='Adjacency matrix output path')
parse.add_argument('--UnionGenePairExpDataSavePath',type=str,default='/home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_dataGenerate/seqFISH',help='The path in which the joint gene pair expression data is located')
parse.add_argument('--LocaldataPath',type=str,default='/home/dreameryty/.vscode/wyj/Github_for_SFINN/code/ST/ST_data/seqFISH/cortex_svz_cellcentroids.csv',help='Spatial location information data loading path')
parse.add_argument('--DatasetName',type=str,default='seqFISH',help='DatasetName')
args = parse.parse_args()

TFlength =args.TFlength # number of data parts divided
UnionGenePairExpDataSavePath = args.UnionGenePairExpDataSavePath
ExpDataSavePath=args.ExpDataSavePath
AdjDataSave_Dir=args.AdjDataSave_Dir
LocaldataPath=args.LocaldataPath
DatasetName=args.DatasetName

if not os.path.isdir(AdjDataSave_Dir):
    os.makedirs(AdjDataSave_Dir)
    
current_path = os.path.abspath('.')

#加载数据
if DatasetName=='seqFISH':
    print('The seqFISH dataset is loaded!')
    cortex_svz_counts = pd.read_csv(ExpDataSavePath)
    cortex_svz_counts_N =cortex_svz_counts.div(cortex_svz_counts.sum(axis=1)+1, axis='rows')*10**4
    cortex_svz_counts_N.columns =[i.lower() for i in list(cortex_svz_counts_N)] ## gene expression normalization
    cortex_svz_cellcentroids = pd.read_csv(LocaldataPath)
    dataset_views=7
elif DatasetName=='MERFISH':
    print('The MERFISH dataset is loaded!')
    cortex_svz_counts=pd.read_csv(ExpDataSavePath,header=0,index_col=0)
    cortex_svz_counts_N =cortex_svz_counts.div(cortex_svz_counts.sum(axis=1)+1, axis='rows')*10**4
    cortex_svz_counts_N.columns =[i.lower() for i in list(cortex_svz_counts_N)]
    cortex_svz_cellcentroids = pd.read_excel(LocaldataPath)
    dataset_views=3 
elif DatasetName=='ST_SCC_P2_1' or DatasetName=='ST_SCC_P2_2' or DatasetName=='ST_SCC_P2_3':
    print('The ST_SCC_P2 is loaded!')
    cortex_svz_counts = pd.read_csv(ExpDataSavePath)
    cortex_svz_counts=cortex_svz_counts.T
    cortex_svz_counts_N =cortex_svz_counts.div(cortex_svz_counts.sum(axis=1)+1, axis='rows')*10**4
    cortex_svz_counts_N.columns =[i.lower() for i in list(cortex_svz_counts_N)]
    cortex_svz_cellcentroids = pd.read_csv(LocaldataPath)    
    dataset_views=1      
else:
    print('DatasetName input error ')    
    
cell_view_list = []
for view_num in range(dataset_views):
    cell_view = cortex_svz_cellcentroids[cortex_svz_cellcentroids['Field of View']==view_num]
    cell_view_list.append(cell_view)


distance_list_list = []
distance_list_list_2 = []
print ('calculating distance matrix, it takes a while')

for view_num in range(dataset_views):
    print (view_num)
    cell_view = cell_view_list[view_num]
    distance_list = []
    for j in range(cell_view.shape[0]):
        for i in range (cell_view.shape[0]):
            if i!=j:
                distance_list.append(np.linalg.norm(cell_view.iloc[j][['X','Y']]-cell_view.iloc[i][['X','Y']]))
    distance_list_list = distance_list_list + distance_list
    distance_list_list_2.append(distance_list)

# np.save(current_path+'/seqfish_plus/distance_array.npy',np.array(distance_list_list))
###try different distance threshold, so that on average, each cell has x neighbor cells, see Tab. S1 for results

distance_array = np.array(distance_list_list)
distance_matrix_threshold_I_list = []
from sklearn.metrics.pairwise import euclidean_distances
for view_num in range (dataset_views):
    cell_view = cell_view_list[view_num]
    distance_matrix = euclidean_distances(cell_view[['X','Y']], cell_view[['X','Y']])
    distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
  
    for i in range(distance_matrix_threshold_I.shape[0]):
        for j in range(distance_matrix_threshold_I.shape[1]):
            
            distance_matrix_threshold_I[i,j] = distance_matrix[i,j]
        
    distance_matrix_threshold_I_list.append(distance_matrix_threshold_I)
    
if DatasetName=='seqFISH':
    print('Calculate the seqFISH distance matrix')
    whole_distance_matrix_threshold_I = scipy.linalg.block_diag(distance_matrix_threshold_I_list[0],
                                                            distance_matrix_threshold_I_list[1],
                                                            distance_matrix_threshold_I_list[2],
                                                            distance_matrix_threshold_I_list[3],
                                                            distance_matrix_threshold_I_list[4],
                                                            distance_matrix_threshold_I_list[5],
                                                            distance_matrix_threshold_I_list[6])

    distance_matrix_threshold_I_N = np.float32(whole_distance_matrix_threshold_I)
    distance_matrixs=1/(1+distance_matrix_threshold_I_N )
    distance_matrixs=np.where(distance_matrixs>0.5, 1, 0)
elif DatasetName=='MERFISH':
    print('Calculate the MERFISH distance matrix')
    whole_distance_matrix_threshold_I = scipy.linalg.block_diag(distance_matrix_threshold_I_list[0],
                                                            distance_matrix_threshold_I_list[1],
                                                            distance_matrix_threshold_I_list[2])

    distance_matrix_threshold_I_N = np.float32(whole_distance_matrix_threshold_I)
    distance_matrixs=1/(1+distance_matrix_threshold_I_N )
    distance_matrixs=np.where(distance_matrixs>0.5, 1, 0)
elif DatasetName=='ST_SCC_P2_1' or DatasetName=='ST_SCC_P2_2' or DatasetName=='ST_SCC_P2_3':
    print('Calculate the '+DatasetName+' distance matrix')
    whole_distance_matrix_threshold_I = scipy.linalg.block_diag(distance_matrix_threshold_I_list[0])

    distance_matrix_threshold_I_N = np.float32(whole_distance_matrix_threshold_I)
    distance_matrixs=1/(1+distance_matrix_threshold_I_N )
    distance_matrixs=np.where(distance_matrixs>3e-3, 1, 0)



# 4. Set the `tensorflow` pseudo-random generator at a fixed value

# tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)
length_TF =TFlength# number of data parts divided
data_path = UnionGenePairExpDataSavePath
num_classes = 2
whole_data_TF = [i for i in range(length_TF)]
# distance_matrixs=np.load('/home/dreameryty/.vscode/wyj/GCNG/3fold_Inputdata/seqfish/distmat/distance_matrixs.npy')
adj=np.float32(distance_matrixs)
adj=adj.flatten()   
############


def load_data_TF2(indel_list,data_path): # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
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
        for k in range(len(ydata)):
            
            yydata.append(ydata[k])
            zzdata.append(zdata[k].split())
        count_setx = count_setx + len(ydata)
        count_set.append(count_setx)
        print (i,len(ydata))
    Gp=pd.DataFrame(zzdata)
    # print(Gp)
    all_train_gene=Gp[0].unique().tolist()+Gp[1].unique().tolist()

    print('len(Gp[1].unique().tolist())',len(Gp[1].unique().tolist()),'len(Gp[0].unique().tolist())',len(Gp[0].unique().tolist()),Gp[0].unique().tolist())
    all_train_gene=list(set(all_train_gene))
    print('len(all_train_gene)',len(all_train_gene))
    # print (np.array(yydata).shape)
    return(all_train_gene,count_set,Gp[0].unique().tolist())

for test_indel in range(1,4): ################## three fold cross validation
    test_TF = [i for i in range (int(np.ceil((test_indel-1)*0.333333*length_TF)),int(np.ceil(test_indel*0.333333*length_TF)))]
    
    train_TF = [i for i in whole_data_TF if i not in test_TF]
  
    
    (gg_pair,count_set,tfs)= load_data_TF2(train_TF,data_path)
    # print('train_tf',train_TF)

     
    data_train=cortex_svz_counts_N[gg_pair]
    Genexp_mat=np.mat(data_train)
    pca=PCA(n_components=0.9,svd_solver='full')#实例化
    pca.fit(Genexp_mat)#拟合模型
    pca_reduction=pca.transform(Genexp_mat)#获取降维后新数据
    # pca_reduction=Genexp_mat
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(pca_reduction)
    distances, indices = nbrs.kneighbors(pca_reduction)

    scaler = preprocessing.StandardScaler().fit(pca_reduction)
    pca_reduction_scaled = scaler.transform(pca_reduction)
    print('降维后数据大小',pca_reduction_scaled.shape)
    labels=['factor'+str(x) for x in range(0,pca_reduction_scaled.shape[1])]
    df_one=DataFrame(pca_reduction_scaled)
    df_one['lables']='factor'


    def Findmax(nparr):
        c=nparr.tolist()
        return c.index(max(c))

    for i in range(pca_reduction_scaled.shape[0]):
        z=Findmax(pca_reduction_scaled[i])
        df_one.loc[i,'lables']='f('+str(z)+')'
        
    #统计每个细胞K个近邻中标记的数量
    #得到样本标记因子
    maxgene=np.argmax(pca_reduction_scaled,axis=1)+1
    maxgene_list=list(maxgene)
    indices_list=list(range(pca_reduction_scaled.shape[0]))

    dict_x={indices_list[i]:maxgene_list[i] for i in range(pca_reduction_scaled.shape[0])}

    data_x=[]
    list_x=[]
    arr_list_x=[]
    value_list=[]
    FN=[]

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
    #获取向量   
    def getvector(dict_y,i):
        vectors=[0]*pca_reduction_scaled.shape[1]
        for keys_k,values_v in dict_y:
            vectors[keys_k-1]=values_v
        FN.append(vectors)
        return 

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
    # z1 =abs(corr_x_y(x=X, y=X))
    # normalize_corr = np.where(z1 >0.8, 1, 0)
    Y = pdist(X, 'cityblock')

# # #将浓缩矩阵还原得到距离矩阵
    distance_matrix = squareform(Y)
    distance_matrixs=1/(1+distance_matrix)
    distsim_to_thre = np.where(distance_matrixs>0.5, 1, 0)
    # distance_matrixs=np.where(distance_matrix<10,1,0)
    
    
    #加入空间信息
    distsim_to_thre_flatten=distsim_to_thre.flatten()
    
    
    join_mat=[]
    for i in range(distsim_to_thre_flatten.shape[0]):
        if distsim_to_thre_flatten[i]==1 or adj[i]==1:
            join_mat.append(1)
        else:
            join_mat.append(0)
    join_mat=np.array(join_mat)
    join_mat=join_mat.reshape( distsim_to_thre.shape[0], distsim_to_thre.shape[1])
    np.save(AdjDataSave_Dir+'/'+str(test_indel)+'foldadj.npy', join_mat)
    
    #不加空间信息
    # np.save(save_dirs+str(test_indel)+'foldadj.npy', distsim_to_thre)
    