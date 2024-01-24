import os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
from sklearn.metrics import precision_recall_curve
import argparse 
parse = argparse .ArgumentParser()
parse.add_argument('--TFlength',type=int,default=18,help='The number of transcription factors')
parse.add_argument('--Epochs',type=int,default=200,help='of epoch')
parse.add_argument('--UnionGenePairExpDataSavePath',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/mHSC-GM/UnionGenePariExpData',help='The path in which the joint gene pair expression data is located')
parse.add_argument('--ModelResSave_Dir',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/mHSC-GM/ModelRes',help='The path where the model results are stored')
parse.add_argument('--PrResPicSave_Dir',type=str,default='/home/dreameryty/.vscode/wyj/SFINN/sc-RNAseqData/mHSC-GM/PR_Res',help='The output path of the resulting picture')
args = parse.parse_args()

def load_UnionGenePairData(indel_list,UnionGenePairExpDataSavePath): # cell type specific 
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


Epochs=args.Epochs
TFlength =args.TFlength# number of data parts divided
UnionGenePairExpDataSavePath = args.UnionGenePairExpDataSavePath

metric_type='PR'
PrResPicSave_Dir=args.PrResPicSave_Dir
ModelResSave_Dir=args.ModelResSave_Dir

if not os.path.isdir(PrResPicSave_Dir):
        os.makedirs(PrResPicSave_Dir)
whole_data_TF = [i for i in range(TFlength)]
s= open(PrResPicSave_Dir+'/'+metric_type+'.txt', 'a')
precisions=[]
PR_set=[] 

mean_recall=np.linspace(0, 1, 100) 
for test_indel in range(1,4): ################## three fold cross validation
    
    test_TF = [i for i in range (int(np.ceil((test_indel-1)*0.333333*TFlength)),int(np.ceil(test_indel*0.333333*TFlength)))]
    train_TF = [i for i in whole_data_TF if i not in test_TF]
  
    (x_train, y_train,count_set_train) = load_UnionGenePairData(train_TF,UnionGenePairExpDataSavePath)
    (x_test, y_test,count_set) = load_UnionGenePairData(test_TF,UnionGenePairExpDataSavePath)
    print(x_train.shape, 'x_train samples')
    print(x_test.shape, 'x_test samples')
    # save_dirs = os.path.join(r'C:\wyj\Linux_MyModel\mEsc\PR_result\adam_lre-6_lr5e-7_B512_d128_drop\',str(test_indel)+'_xxjust_two_3fold_YYYY_saved_models_e100')
    save_dirs = os.path.join(ModelResSave_Dir,str(test_indel)+'_ThreeFold_Saved_SFINNmodels_e'+str(Epochs))
 
    # ###########
    # if not os.path.isdir(save_dirs):
    #     os.makedirs(save_dirs)

    y_test=np.load(save_dirs+'/end_y_test.npy')
    y_predict=np.load(save_dirs+'/end_y_predict.npy')

    y_testy = y_test
    y_predicty = y_predict
    fig = plt.figure(figsize=(5, 5))
    # plt.plot([0, 1], [0, 1])
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precison')
  
    for jj in range(len(count_set) - 1):  # len(count_set)-1):
        if count_set[jj] < count_set[jj + 1]:
            print(test_indel, jj, count_set[jj], count_set[jj + 1])
            y_test = y_testy[count_set[jj]:count_set[jj + 1]]
            y_predict = y_predicty[count_set[jj]:count_set[jj + 1]]
            precision,recall,_=precision_recall_curve(y_test,y_predict)
     
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
median_pr=np.mean(PR_set)
plt.plot(mean_recall, mean_precision, 'k', lw=3, label='median PR')
plt.title(str('%.4f'%mean_pr)+"("+str('%.4f'%median_pr)+')')
plt.fill_between(mean_recall, per_precision[0, :], per_precision[2, :], color='c', alpha=.2, label='Quartile')
plt.plot(mean_recall, per_precision[0, :], 'c', lw=3, alpha=.2)
plt.legend(loc='lower right')
plt.savefig(PrResPicSave_Dir+'/'+metric_type+'.pdf')
s.close()
