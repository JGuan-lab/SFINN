import pandas as pd
from numpy import *
import numpy as np
import json, re,os, sys
import keras
import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.per_process_gpu_memory_fraction = 0.9
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten,Dropout
from tensorflow.keras.optimizers import Adam,SGD
from keras.regularizers import l2

from spektral.layers import GCNConv
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.utils import normalized_laplacian

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
from sklearn.metrics import precision_recall_curve
from scipy import sparse as sp
seed_value= 1234

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.compat.v1.set_random_seed(seed_value)
from scipy import sparse as sp
current_path = os.path.abspath('.')

import argparse 
parse = argparse .ArgumentParser()
parse.add_argument('--TFlength',type=int,default=18,help='The number of transcription factors')
parse.add_argument('--TrainDataLoadDir',type=str)
parse.add_argument('--TestDataLoadDir',type=str)
parse.add_argument('--AdjDataSave_Dir',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/AdjData',help='The number of transcription factors')
parse.add_argument('--UnionGenePairExpDataSavePath',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/bone/UnionGenePariExpData',help='Federated gene pair expression data storage pathway')
parse.add_argument('--ModelResSavePath',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/mesc/ModelRes',help='The path where the model results are saved')
parse.add_argument('--Epochs',type=int,default=200,help='of epoch')
parse.add_argument('--Lr',type=float,default=1e-6,help='learning rate')
parse.add_argument('--L2_Reg',type=float,default=5e-4,help='l2_reg')
parse.add_argument('--Es_Patience',type=int,default=100,help='es_patience')
parse.add_argument('--Batch_Size',type=int,default=512,help='batch_size')
args = parse.parse_args()

# Parameters
TFlength =args.TFlength # number of data parts divided
TrainDataLoadDir=args.TrainDataLoadDir
TestDataLoadDir=args.TestDataLoadDir
UnionGenePairExpDataSavePath = args.UnionGenePairExpDataSavePath
AdjDataSave_Dir=args.AdjDataSave_Dir
ModelResSavePath=args.ModelResSavePath
Epochs=args.Epochs
Es_Patience=args.Es_Patience
Lr=args.Lr
L2_Reg=args.L2_Reg
Batch_Size=args.Batch_Size
Whole_Data_TF = [i for i in range(TFlength)]

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

def Degree_Power(adj, pow):
    """
    Computes \(D^{p}\) from the given adjacency matrix. Useful for computing
    normalised Laplacians.
    :param adj: rank 2 array or sparse matrix
    :param pow: exponent to which elevate the degree matrix
    :return: the exponentiated degree matrix in sparse DIA format
    """
    degrees = np.power(np.array(adj.sum(1)), pow).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(adj):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D

def Self_Connection_Normalized_Adjacency(adj, symmetric=True):
    """
    Normalizes the given adjacency matrix using the degree matrix as either
    \(D~^{-1}A~\) or \(D~^{-1/2}A~D~^{-1/2}\) (symmetric normalization).where A~ = A+I
    :param adj: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized adjacency matrix.
    """
    if sp.issparse(adj):
        I = sp.eye(adj.shape[-1], dtype=adj.dtype)
    else:
        I = np.eye(adj.shape[-1], dtype=adj.dtype)
    A1 = adj  
   
    if symmetric:
        normalized_D = Degree_Power(A1, -0.5)
        output = normalized_D.dot(A1).dot(normalized_D)
    else:
        normalized_D = Degree_Power(A1, -1.)
        output = normalized_D.dot(A1)
    return output




for test_indel in range(1,4): ################## three fold cross validation
    
    adj=np.load(AdjDataSave_Dir+'/'+str(test_indel)+'foldadj.npy')
    adj=np.float32(adj)
    fltr = Self_Connection_Normalized_Adjacency(adj)
    # test_TF = [i for i in range (int(np.ceil((test_indel-1)*0.333333*TFlength)),int(np.ceil(test_indel*0.333333*TFlength)))]
    # train_TF = [i for i in Whole_Data_TF if i not in test_TF]
    # (x_train, y_train,count_set_train) = Load_UnionGenePairData(train_TF,UnionGenePairExpDataSavePath)
    # (x_test, y_test,count_set) = Load_UnionGenePairData(test_TF,UnionGenePairExpDataSavePath)
    
    TrainDataSaveDirs=TrainDataLoadDir+'/'+str(test_indel)+'_fold'
    TestDataSaveDirs=TestDataLoadDir+'/'+str(test_indel)+'_fold'
    x_train=np.load(TrainDataSaveDirs+'/x_train_data' +  '.npy')
    y_train=np.load(TrainDataSaveDirs+'/y_train_data' + '.npy')
    count_set_train=np.load(TrainDataSaveDirs+'/count_set_train_data_tf' + '.npy')
    
    x_test=np.load(TestDataSaveDirs+'/x_test_data' +  '.npy')
    y_test=np.load(TestDataSaveDirs+'/y_test_data' + '.npy')
    count_set=np.load(TestDataSaveDirs+'/count_set_test_data_tf' + '.npy')
    
    print( 'x_train samples',x_train.shape)
    print( 'x_test samples',x_test.shape)

    N = x_train.shape[-2]  # Number of nodes in the graphs
    F = x_train.shape[-1]  # Node features dimensionality
    n_out = y_train.shape[-1]  # Dimension of the target

    # fltr = normalized_laplacian(adj)
    # Model definition
    X_in = Input(shape=(N, F))
  
    A_in = Input(tensor=tf.convert_to_tensor(fltr,dtype='float32'))
   
    # A_in = Input(tensor=sp_matrix_to_sp_tensor(adj))
    nn=Dense(32, input_dim=2,activation='relu')(X_in)
    nn=Dense(32,activation='relu')(nn)
    gcnconv = GCNConv(32,activation='elu',kernel_regularizer=l2(L2_Reg),use_bias=True)([X_in, A_in])
    gcnconv = GCNConv(32,activation='elu',kernel_regularizer=l2(L2_Reg),use_bias=True)([gcnconv, A_in])
    concatenates= keras.layers.Concatenate()([gcnconv,nn])
    flatten = Flatten()(concatenates)
    fc = Dense(512, activation='relu')(flatten)
    output = Dense(n_out, activation='sigmoid')(fc)
    # Build model
    model = Model(inputs=[X_in, A_in], outputs=output)
    optimizer = Adam(lr=Lr)
   
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])
    model.summary()

    save_dirs = os.path.join(ModelResSavePath,str(test_indel)+'_ThreeFold_Saved_SFINNmodels_e'+str(Epochs))
    
   
    early_stopping = EarlyStopping(monitor='val_acc', patience=Es_Patience, verbose=0, mode='auto')
    checkpoint1 = ModelCheckpoint(filepath=save_dirs + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    checkpoint2 = ModelCheckpoint(filepath=save_dirs + '/weights.hdf5', monitor='val_acc', verbose=1,save_best_only=True, mode='auto', period=1)

    callbacks = [checkpoint2, early_stopping]
   
    history = model.fit(x_train,y_train, batch_size=Batch_Size,validation_split=0.2,shuffle=True,epochs=Epochs,callbacks=callbacks)

    # Load best model
    # Save model and weights

    model_name = 'SFINN.h5'
    model_path = os.path.join(save_dirs, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    y_predict = model.predict(x_test)
    np.save(save_dirs + '/end_y_test.npy', y_test)
    np.save(save_dirs + '/end_y_predict.npy', y_predict)
    ################################## plot training process

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.savefig(save_dirs + '/end_result.pdf')

    
    y_testy = y_test
    y_predicty = y_predict
    fig = plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1])
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel('FP')
    plt.ylabel('TP')
    # plt.grid()

    AUC_set = []
    s1 = open(save_dirs + '/DivInteract_Roc.txt', 'w')
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)  # 3068
    for jj in range(len(count_set) - 1):  # len(count_set)-1):
        if count_set[jj] < count_set[jj + 1]:
            print(test_indel, jj, count_set[jj], count_set[jj + 1])
            y_test = y_testy[count_set[jj]:count_set[jj + 1]]
            y_predict = y_predicty[count_set[jj]:count_set[jj + 1]]
            # Score trained model.
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            # Print ROC curve
            plt.plot(fpr, tpr, color='0.5', lw=0.001, alpha=.2)
            auc = np.trapz(tpr, fpr)
            s1.write(str(jj) + '\t' + str(count_set[jj]) + '\t' + str(count_set[jj + 1]) + '\t' + str(auc) + '\n')
            print('AUC:', auc)
            AUC_set.append(auc)

    mean_tpr = np.median(tprs, axis=0)
    mean_tpr[-1] = 1.0
    per_tpr = np.percentile(tprs, [25, 50, 75], axis=0)
    mean_roc = np.trapz(mean_tpr, mean_fpr)
    median_roc=np.median(AUC_set)
    plt.plot(mean_fpr, mean_tpr, 'k', lw=3, label='median ROC')
    plt.title(str('%.4f'%median_roc)+"("+str('%.4f'%mean_roc)+')')
    plt.fill_between(mean_fpr, per_tpr[0, :], per_tpr[2, :], color='g', alpha=.2, label='Quartile')
    plt.plot(mean_fpr, per_tpr[0, :], 'g', lw=3, alpha=.2)
    plt.legend(loc='lower right')
    plt.savefig(save_dirs + '/DivInteract_Roc.pdf')
    del fig

    fig = plt.figure(figsize=(5, 5))
    plt.hist(AUC_set, bins=50)
    plt.savefig(save_dirs + '/DivInteract_Roc_Hist.pdf')
    del fig
    s1.close()

    fig = plt.figure(figsize=(5, 5))
    # plt.plot([0, 1], [0, 1])
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    s= open(save_dirs + '/DivInteract_Pr.txt', 'w')
    precisions=[]
    PR_set=[]
    # mean_fpr = np.linspace(0, 1, 100)  # 3068
    mean_recall=np.linspace(0, 1, 100) 
    for jj in range(len(count_set) - 1):  # len(count_set)-1):
        if count_set[jj] < count_set[jj + 1]:
            print(test_indel, jj, count_set[jj], count_set[jj + 1])
            y_test = y_testy[count_set[jj]:count_set[jj + 1]]
            y_predict = y_predicty[count_set[jj]:count_set[jj + 1]]
            # Score trained model.
            precision,recall,_=precision_recall_curve(y_test,y_predict)
            # precisions[-1][0]=1.0
            precision = np.array(precision)
            recall= np.array(recall)
            
            arrIndex = np.array(recall).argsort()
            precision = precision[arrIndex]
            recall= recall[arrIndex]
            precisions.append(np.interp(mean_recall, recall, precision))
            plt.plot(recall,precision,color='0.5',lw=0.001,alpha=.2)
            aupr=np.trapz(precision,recall)
            
            s.write(str(jj) + '\t' + str(count_set[jj]) + '\t' + str(count_set[jj + 1]) + '\t' + str(aupr) + '\n')
            print('PR:', aupr)
            PR_set.append(aupr)

    mean_precision = np.median(precisions, axis=0)
    per_precision = np.percentile(precisions,[25,50,70],axis=0)
    
    mean_pr = np.trapz(mean_precision,mean_recall)
    median_pr=np.median(PR_set)
    plt.plot(mean_recall, mean_precision, 'k', lw=3, label='median PR')
    plt.title(str('%.4f'%median_pr)+"("+str('%.4f'%mean_pr)+')')
    plt.fill_between(mean_recall, per_precision[0, :], per_precision[2, :], color='g', alpha=.2, label='Quartile')
    plt.plot(mean_recall, per_precision[0, :], 'g', lw=3, alpha=.2)
    plt.legend(loc='lower right')
    plt.savefig(save_dirs + '/DivInteract_Pr.pdf')
    
    del fig
    s.close()
