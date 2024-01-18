'''
功能:输入Groundtrue列表,只保留含有0,1关系的基因对,同时为该候选基因对列表生成区间划分索引列表文件
调用:python GenerateSplitIndexFile.py --GroundTrueFilePath xx --SplitIndexSavePath xx
参数:
--GroundTrueFilePath  GroundTrue数据所在路径
--SplitIndexSavePath  区间划分索引列表保存路径

'''
import pandas as pd
from numpy import *
import json, re,os, sys
import argparse 
parse = argparse .ArgumentParser()
parse.add_argument('--GroundTrueFilePath',type=str)
parse.add_argument('--SplitIndexSavePath',type=str)
args = parse.parse_args()

GroundTrueFilePath=args.GroundTrueFilePath 
SplitIndexSavePath=args.SplitIndexSavePath
file = open(GroundTrueFilePath, "r") 
lines = []
for i in file:
	lines.append(i.split())
file.close()
lines=array(lines)
new = []
for i in lines:
	if(i[2]=='0'or i[2]=='1'):
		new.append(i)

file_write = open(GroundTrueFilePath, "w")
for var in new:
	file_write.writelines(var[0]+'\t'+var[1]+'\t'+var[2])
	file_write.writelines('\n')	
file_write.close()

data=pd.read_csv(GroundTrueFilePath,sep='\t',header=None)
result=data[0].tolist()
result_dic={}
for item_str in result:
    if item_str not in result_dic:
        result_dic[item_str]=1
    else:
        result_dic[item_str]+=1
myDict = result_dic
valueList = myDict.values()
a=list(valueList)
count=0
counts=[0]
for i in range(0,len(a)):
    count=counts[i]+a[i]
    counts.append(count)   
counts= list(map(str, counts))

with open(SplitIndexSavePath,"w") as f:
	for c in counts:
		f.writelines(c+'\n')
				
	file_write.close()