import argparse 
import numpy as np
from numpy import *
import json, re,os, sys
from sklearn import preprocessing
parse = argparse .ArgumentParser()
parse.add_argument('--TFlength',type=int,default=18,help='The number of transcription factors')
parse.add_argument('--TrainDataSaveDir',type=str,help='The path where the training dataset is saved')
parse.add_argument('--TestDataSaveDir',type=str,help='Test the dataset save path')
parse.add_argument('--UnionGenePairExpDataSavePath',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/UnionGenePariExpData',help='Federated gene pair expression data storage pathway')

args = parse.parse_args()
TFlength=args.TFlength
UnionGenePairExpDataSavePath=args.UnionGenePairExpDataSavePath
TrainDataSaveDir=args.TrainDataSaveDir
TestDataSaveDir=args.TestDataSaveDir
Whole_Data_TF = [i for i in range(TFlength)]

if not os.path.isdir(TrainDataSaveDir):
        os.makedirs(TrainDataSaveDir)
if not os.path.isdir(TestDataSaveDir):
        os.makedirs(TestDataSaveDir)
        
def Load_UnionGenePairData(indel_list,UnionGenePairExpDataSavePath): 
    import random
    import numpy as np
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:#len(h_tf_sc)):
        xdata = np.load(UnionGenePairExpDataSavePath+'/Nxdata_tf' + str(i) + '.npy')
        ydata = np.load(UnionGenePairExpDataSavePath+'/ydata_tf' + str(i) + '.npy')
        for k in range(len(ydata)):
            xxdata_list.append(xdata[k,:,:])
            yydata.append(ydata[k])
        count_setx = count_setx + len(ydata)
        count_set.append(count_setx)
        # print (i,len(ydata))
    yydata_array = np.array(yydata)[:,np.newaxis]
    yydata_x = yydata_array.astype('int')
    print (np.array(xxdata_list).shape)
    return((np.array(xxdata_list),yydata_x,count_set))

for test_indel in range(1,4): ################## three fold cross validation
    
    test_TF = [i for i in range (int(np.ceil((test_indel -1)*0.333333*TFlength)),int(np.ceil(test_indel *0.333333*TFlength)))]
    train_TF = [i for i in Whole_Data_TF if i not in test_TF]
    (x_train, y_train,count_set_train) = Load_UnionGenePairData(train_TF,UnionGenePairExpDataSavePath)

    TrainDataSaveDirs=TrainDataSaveDir+'/'+str(test_indel)+'_fold'
    
    if not os.path.isdir(TrainDataSaveDirs):
            os.makedirs(TrainDataSaveDirs)
  
    save(TrainDataSaveDirs+'/x_train_data' +  '.npy', x_train)
    save(TrainDataSaveDirs+'/y_train_data' + '.npy', y_train)
    save(TrainDataSaveDirs+'/count_set_train_data_tf' + '.npy',count_set_train)

    (x_test, y_test,count_set) = Load_UnionGenePairData(test_TF,UnionGenePairExpDataSavePath)
    TestDataSaveDirs=TestDataSaveDir+'/'+str(test_indel)+'_fold'
    
    if not os.path.isdir(TestDataSaveDirs):
        os.makedirs(TestDataSaveDirs)
    save(TestDataSaveDirs+'/x_test_data' +  '.npy', x_test)
    save(TestDataSaveDirs+'/y_test_data' + '.npy', y_test)
    save(TestDataSaveDirs+'/count_set_test_data_tf' + '.npy',count_set)
