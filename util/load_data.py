import json
import os
import re
import collections
import datetime
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
# import dgl
import csv
import random
import h5py
import numpy as np
class build_dataset():
    """配置参数"""
    def __init__(self,path,tf_num,ways):
        self.act_dict = {
            'Logon': 1,
            'Logoff': 2,
            'http': 3,
            'email': 4,
            'file': 5,
            'Connect': 6,
            'Disconnect': 7
        }
        self.path=path
        self.tf_num=tf_num
        self.data_seq={}
        self.label={}
        self.ways=ways
        # self.get_process_seq()
        if self.ways=="bert":
            self.train_dataset,self.train_label,self.test_dataset,self.test_label=self.get_data_seq(1,0.7)
        elif self.ways=="sup-test":
            # self.simcse_data,self.neg_data=self.get_sup_data_seq()
            self.neg_data=self.get_sup_data_seq()
        elif self.ways=="mlm":
            self.simcse_data=self.get_mlmdata_seq()
        elif self.ways=="mal":
            self.train_data,self.test_data,self.labels=self.get_mal_seq()
        else:
            self.simcse_data=self.get_sup_data_seq()
    
        #self.simcse_data=self.get_sup_data_seq() #这个是为了获得对比训练数据集的
        # self.train_dataset,self.train_label,self.test_dataset,self.test_label=self.get_data_seq(65,0.7)
        # self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
        # self.data_sens,self.graph_list=self.get_data_sens()#(name,pc,此人在这个pc上这个时间段的操作序列【logon-logoff为一句】，标签)
    #获取训练集和测试集。原理是遍历获得的所有语句序列和它的标签，如果序列里标签值有1，就定义为恶意序列。
    def get_data_seq(self,num,ratio):#num的原因是原来考虑做操作序列级别的上下文提取，但是现在还没出最后结果，就先不动
        dd_pos=[]
        dd_neg=[]
        la_pos=[]
        la_neg=[]
        for key in self.data_seq:#1的意思是暂时不考虑多语序结合
            for i in range(len(self.data_seq[key])-num):
                if 1 in self.label[key][i:i+num]:
                    la_pos.append(1)
                    dd_pos.append(self.data_seq[key][i:i+num])
                else:
                    la_neg.append(0)
                    dd_neg.append(self.data_seq[key][i:i+num])
        #从样本里随机采样获取训练集，其余的就是测试集。为了样本数量平衡
        dd_train_pos=random.sample(dd_pos,int(ratio*len(dd_pos)))
        dd_test_pos= [x for x in dd_pos if x not in dd_train_pos]
        dd_train_neg=random.sample(dd_neg,int((1-ratio)*len(dd_neg)))
        dd_test_neg=[x for x in dd_neg if x not in dd_train_neg]
        # return dd_train_pos+dd_train_neg, [1 for _ in range(len(dd_train_pos))]+[0 for _ in range(len(dd_train_neg))],dd_pos+dd_neg,[1 for _ in range(len(dd_pos))]+[0 for _ in range(len(dd_neg))]
        return dd_train_pos+dd_train_neg, [1 for _ in range(len(dd_train_pos))]+[0 for _ in range(len(dd_train_neg))],dd_test_pos+dd_test_neg,[1 for _ in range(len(dd_test_pos))]+[0 for _ in range(len(dd_test_neg))]
    def get_mal_seq(self):
        train_data=[]
        test_data=[]
        test_label=[]
        evidence_info=pd.read_csv("/home/c402/LhL/cross_certification/dataset/r4.2/sens_data_deive.csv")
        evi_json={"1-1":[],"2-1":[],"2-2":[],"2-3":[],"3-1":[],"3-2":[],"3-3":[]}
        for index,item in evidence_info.iterrows():
            if item[0] in evi_json:
                evi_json[item[0]].append(item[1])
            # for key in evi_json:
            in_turn=1
        for key in evi_json:
            train_data.extend(evi_json[key][:int(0.7*len(evi_json[key]))])
            test_data.extend(evi_json[key][int(0.7*len(evi_json[key])):])
            test_label.extend(len(evi_json[key][int(0.7*len(evi_json[key])):])*[1])
        key_list=evi_json.keys()
        neg_info=pd.read_csv("/home/c402/LhL/cross_certification/dataset/r4.2/sens_data_multi_rep1.csv")#这里有问题！！
        name_info=list(set(neg_info["name"].tolist()))
        name_json={}
        for k in name_info:
            name_json[k]=[]
        for index,item in neg_info.iterrows():
            name_json[item["name"]].append(item["sens"])
        for k in name_json:
            train_data.extend(name_json[k][:int(0.7*len(name_json[k]))])
            test_data.extend(evi_json[key][int(0.7*len(evi_json[key])):])
            test_label.extend(len(evi_json[key][int(0.7*len(evi_json[key])):])*[0])
        '''
        neg_info=pd.read_csv("/home/c402/LhL/cross_certification/dataset/r4.2/sens_data_multi_rep2.csv")
        name_info=list(set(neg_info["name"].tolist()))
        name_json={}
        for k in name_info:
            name_json[k]=[]
        for index,item in neg_info.iterrows():
            name_json[item["name"]].append(item["sens"])
        for k in name_json:
            train_data.extend(name_json[k][:int(0.7*len(name_json[k]))])
            test_data.extend(evi_json[key][int(0.7*len(evi_json[key])):])
            test_label.extend(len(evi_json[key][int(0.7*len(evi_json[key])):])*[0])
            
        neg_info=pd.read_csv("/home/c402/LhL/cross_certification/dataset/r4.2/sens_data_multi_rep3.csv")

        name_info=list(set(neg_info["name"].tolist()))
        name_json={}
        for k in name_info:
            name_json[k]=[]
        for index,item in neg_info.iterrows():
            name_json[item["name"]].append(item["sens"])
        for k in name_json:
            train_data.extend(name_json[k][:int(0.7*len(name_json[k]))])
            test_data.extend(evi_json[key][int(0.7*len(evi_json[key])):])
            test_label.extend(len(evi_json[key][int(0.7*len(evi_json[key])):])*[0])
        random.shuffle(train_data)
        '''
        return train_data,test_data,test_label
    def get_mlmdata_seq(self):##增量训练的
        dd_train=[]
        labels=[]
        dd_test=[]
        x=0
        evidence_info=pd.read_csv("/home/lhl/cross_certification/dataset/r4.2/sens_data_deive.csv")
        # evidence_info=pd.read_csv("/home/lhl/cross_certification/dataset/r5.2/sens_data_deive.csv")
        # evi_json={"1-1":[],"2-1":[],"2-2":[],"2-3":[],"3-1":[],"3-2":[],"3-3":[]}
        for index,item in evidence_info.iterrows():
            dd_train.append(item[1])
        '''
        neg_info=pd.read_csv("/home/lhl/cross_certification/dataset/r5.2/sens_data_multirep4.csv")
        neg_sens=[]
        name_info=list(set(neg_info["name"].tolist()))
        name_json={}
        for k in name_info:
            name_json[k]=[]
        for index,item in neg_info.iterrows():
            name_json[item["name"]].append(item["sens"])
        for k in name_json:
            neg_sens.extend(name_json[k][:int(0.7*len(name_json[k]))])
        xxx=len(neg_sens)
        a=0
        dd_train.extend(neg_sens)
        '''
        neg_info=pd.read_csv("/home/c402/LhL/cross_certification/dataset/sens_data_multi_rep1.csv")
        '''
        neg_sens=[]
        name_info=list(set(neg_info["name"].tolist()))
        name_json={}
        for k in name_info:
            name_json[k]=[]
        for index,item in neg_info.iterrows():
            name_json[item["name"]].append(item["sens"])
        for k in name_json:
            neg_sens.extend(name_json[k][:int(0.7*len(name_json[k]))])
        xxx=len(neg_sens)
        a=0
        '''
        neg_sens=neg_info["sens"].tolist()
        dd_train.extend(neg_sens)  
    
        neg_info=pd.read_csv("/home/c402/LhL/cross_certification/dataset/sens_data_multi_rep2.csv")
        neg_sens=neg_info["sens"].tolist()
        dd_train.extend(neg_sens)    
        neg_info=pd.read_csv("/home/c402/LhL/cross_certification/dataset/sens_data_multi_rep3.csv")
        neg_sens=neg_info["sens"].tolist()
        dd_train.extend(neg_sens)
        return dd_train
    def get_sup_data_seq(self):#为了获得对比训练数据集
        dd_pos=[]
        dd_neg=[]
        la_pos=[]
        la_neg=[]
        dd_train=[]
        num=1
        # in_turn=5
        if self.ways=="sup":        
            evidence_info=pd.read_csv("/home/lhl/cross_certification/dataset/r4.2/sens_data_deive.csv")
            evi_json={"1-1":[],"2-1":[],"2-2":[],"2-3":[],"3-1":[],"3-2":[],"3-3":[]}
          #  evi_json={"1-1":[],"2-1":[],"2-2":[],"3-1":[],"3-2":[],"4-1":[]}
            # evi_json={"1-1":[],"2-1":[],"2-2":[],"3-1":[],"3-2":[],"3-3":[]}
            ll=0
            for index,item in evidence_info.iterrows():
                ll+=1
                if item[0] in evi_json:
                    evi_json[item[0]].append(item[1])
            print(ll)
            # for key in evi_json:
            in_turn=1
            for key in evi_json:#证据的前70%是训练集
                evi_json[key]=evi_json[key][:int(0.7*len(evi_json[key]))]
            key_list=evi_json.keys()
            print(len(key_list))
            while(in_turn>0):#现在只搞一遍
                for key in evi_json:
                    for key_neg in key_list:
                        if key_neg!=key:
                            for da in evi_json[key]:
                                # dd_train.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(evi_json[key_neg],1)[0]]])
                                dd_train.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(evi_json[key_neg],1)[0]]])
                in_turn-=1
            neg_info=pd.read_csv("/home/lhl/cross_certification/dataset/r4.2/sens_data_multi.csv")#这里有问题！！
            neg_sens=[]
            name_info=list(set(neg_info["name"].tolist()))
            name_json={}
            print(len(name_info))
            for k in name_info:
                name_json[k]=[]
            ll=0
            for index,item in neg_info.iterrows():
                name_json[item["name"]].append(item["sens"])
            for k in name_json:
                neg_sens.extend(name_json[k][:int(0.7*len(name_json[k]))])
            xxx=5000
            a=0#这里有问题！！！
            while a<xxx:
                for key in evi_json:
                    for da in evi_json[key]:#这就不一样了
                        # dd_train.append([[da],[da],[random.sample(neg_sens,1)[0]]])
                        dd_train.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(neg_sens,1)[0]]])
                    #dd_train.append([[da],[random.sample(evi_json[key],1)[0]],[neg_sens[a]]])#
                        a+=1
                        if a==xxx:
                            break
                #if a==xxx:
                    #break
            '''
            neg_info=pd.read_csv("/home/lhl/cross_certification/dataset/r5.2/sens_data_multirep4.csv")
            neg_sens=[]
            name_info=list(set(neg_info["name"].tolist()))
            print(len(name_info))
            name_json={}
            ll=0
            for k in name_info:
                name_json[k]=[]
                ll+=1
            print(ll)
            for index,item in neg_info.iterrows():
                name_json[item["name"]].append(item["sens"])
            for k in name_json:
                neg_sens.extend(name_json[k][:int(0.7*len(name_json[k]))])
            xxx=5000
            a=0
            # while a<xxx:
            for key in evi_json:
                for da in evi_json[key]:
                        # dd_train.append([[da],[da],[random.sample(neg_sens,1)[0]]])
                        # dd_train.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(neg_sens,1)[0]]])
                    dd_train.append([[da],[random.sample(evi_json[key],1)[0]],[neg_sens[a]]])
                    a+=1
                    if a>=xxx:
                        break
                if a>=xxx:
                    break
            '''
            neg_info=pd.read_csv("/home/lhl/cross_certification/dataset/r4.2/sens_data_multi_rep3.csv")
            neg_sens=[]
            name_info=list(set(neg_info["name"].tolist()))
            name_json={}
            for k in name_info:
                name_json[k]=[]
            ll=0
            for index,item in neg_info.iterrows():
                ll+=1
                name_json[item["name"]].append(item["sens"])
            print(ll)
            for k in name_json:
                neg_sens.extend(name_json[k][:int(0.7*len(name_json[k]))])
            xxx=5000
            a=0
            while a<xxx:
                for key in evi_json:
                    for da in evi_json[key]:
                        # dd_train.append([[da],[da],[random.sample(neg_sens,1)[0]]])
                        dd_train.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(neg_sens,1)[0]]])
                    #dd_train.append([[da],[random.sample(evi_json[key],1)[0]],[neg_sens[a]]])
                        a+=1
                        if a>=xxx:
                            break
                #if a>=xxx:
                    #break

        #从样本里随机采样获取训练集，其余的就是测试集。
        # while(in_turn>=0):
            # for da in dd_pos:#【0】的意思就是列表第一个
                # new_da=[da,random.sample(dd_pos,1)[0],random.sample(dd_neg,1)[0]]
                # dd_train.append(new_da)
                # new_da=[da,da,random.sample(dd_neg,1)[0]]
                # dd_train.append(new_da)
            # in_turn-=1
            # lens=len(dd_train)
            # aa=len(dd_train)
            # while(lens>=int(aa/2)):
                # for da in dd_neg:
                    # new_da=[da,random.sample(dd_neg,1)[0],random.sample(dd_pos,1)[0]]
                    # dd_train.append(new_da)
                # lens-=1
            # '''
            print(len(dd_train))
            return dd_train
        if self.ways=="sup-test":
            evidence_info=pd.read_csv("/home/lhl/cross_certification/dataset/r4.2/sens_data_deive.csv")
            evi_json={"1-1":[],"2-1":[],"2-2":[],"2-3":[],"3-1":[],"3-2":[],"3-3":[]}
            # evi_json={"1-1":[],"2-1":[],"2-2":[],"3-1":[],"3-2":[],"4-1":[]}
            neg_data=[]
            for index,item in evidence_info.iterrows():
                if item[0] in evi_json:
                    evi_json[item[0]].append(item[1])
            neg_info=pd.read_csv("/home/lhl/cross_certification/dataset/r4.2/sens_data_multi_rep1.csv")
            neg_sens=[]
            name_info=list(set(neg_info["name"].tolist()))
            name_json={}
            for k in name_info:
                name_json[k]=[]
            for index,item in neg_info.iterrows():
                name_json[item["name"]].append(item["sens"])
            for k in name_json:
                # neg_sens.append(name_json[k][int(0.7*len(name_json[k])):])
                neg_sens.extend(name_json[k][int(0.7*len(name_json[k])):])
            xxx=1500
            a=0
            #这有问题啊
            while a<xxx:#问题在这里！！！啥啊这是，neg_data里是混的
                for key in evi_json:
                    for da in evi_json[key]:
                        # neg_data.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(neg_sens,1)[0]]])
                  #  neg_data.append([[da],[random.sample(evi_json[key],1)[0]],[neg_sens[a]]])
                        neg_data.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(neg_sens,1)[0]]])
                        a+=1
                        if a==xxx:
                            break
                #if a==xxx:
                 #   break
            neg_info=pd.read_csv("/home/lhl/cross_certification/dataset/r4.2/sens_data_multi_rep2.csv")
            neg_sens=[]
            name_info=list(set(neg_info["name"].tolist()))
            name_json={}
            for k in name_info:
                name_json[k]=[]
            for index,item in neg_info.iterrows():
                name_json[item["name"]].append(item["sens"])
            for k in name_json:
                neg_sens.extend(name_json[k][int(0.7*len(name_json[k])):])
            xxx=1500
            a=0
            # neg_data=[]
            while a<xxx:
                for key in evi_json:
                    for da in evi_json[key]:
                   # neg_data.append([[da],[random.sample(evi_json[key],1)[0]],[neg_sens[a]]])
                        neg_data.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(neg_sens,1)[0]]])
                        a+=1
                        if a==xxx:
                            break
               # if a==xxx:
                #    break
            neg_info=pd.read_csv("/home/lhl/cross_certification/dataset/r4.2/sens_data_multi_rep3.csv")
            neg_sens=[]
            name_info=list(set(neg_info["name"].tolist()))
            name_json={}
            for k in name_info:
                name_json[k]=[]
            for index,item in neg_info.iterrows():
                name_json[item["name"]].append(item["sens"])
            for k in name_json:
                neg_sens.extend(name_json[k][int(0.7*len(name_json[k])):])
            xxx=1500
            a=0
            # neg_data=[]
            while a<xxx:
                for key in evi_json:
                    for da in evi_json[key]:
                            # neg_data.append([[da],[random.sample(neg_sens,1)[0]]])
                        neg_data.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(neg_sens,1)[0]]])
                        a+=1
                        if a==xxx:
                            break
               # if a==xxx:
                #    break
                        # a+=1
                        # if a==xxx:
                            # break
                    # if a==xxx:
                        # break
            # for index,item in evidence_info.iterrows():
                # if item[0] in evi_json:
                    # evi_json[item[0]].append(item[1])
            # for key in evi_json:
            in_turn=2
            xxx=0
            for key in evi_json:
                evi_json[key]=evi_json[key][int(0.7*len(evi_json[key])):]
            key_list=evi_json.keys()
            while(in_turn>0):
                for key in evi_json:
                    for key_neg in key_list:
                        if key_neg!=key:
                            for da in evi_json[key]:
                                # dd_train.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(neg_sens,1)[0]]])
                                # if xxx%2==0:
                                # dd_train.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(evi_json[key_neg],1)[0]]])
                                neg_data.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(evi_json[key_neg],1)[0]]])
                                    # xxx+=1
                in_turn-=1

            # neg_data=random.sample(neg_data,len(dd_train))
            # return dd_train,neg_data
            print(len(neg_data))
            return neg_data
            # return dd_train
        if self.ways=="unsup":
            in_turn=1
            evidence_info=pd.read_csv("/home/lhl/cross_certification/dataset/r5.2/sens_data_deive.csv")
            self.data_seq=[]
            while in_turn>0:
                for index,item in evidence_info.iterrows():
                    self.data_seq.append(item[1])
                in_turn-=1
            evidence_info=pd.read_csv("/home/lhl/cross_certification/dataset/r5.2/sens_data_multirep4.csv")
            for index,item in evidence_info.iterrows():
                self.data_seq.append(item[1])
            
            return self.data_seq[:int(0.7*(len(self.data_seq)))]

    # 获取一个logon、logoff序列的。【文件、邮件、http取关键词前三个】
    def get_process_seq(self):
        x=0
        #加上分类了
        emil=0
        for dir in os.listdir(self.path):
            # dir="r4.2-3"
            self.new_path=os.path.join(self.path,dir)
            for file in os.listdir(self.new_path):
                # x+=1
                # if x>=3:
                    # break
                file_path=os.path.join(self.new_path,file)
                self.data_seq[file[:-3]]=[]
                self.label[file[:-3]]=[]
                with h5py.File(file_path, 'a') as hf:
                    da=hf["seq_data"]
                    la=hf["seq_label"]
                    lal=[]
                    str_sent=''
                    
                    for i in range(len(da)):
                        lal.append(la[i])
                        seq=str(da[i].decode('utf-8'))
                        behaviors=int(seq.split(" ")[0])+1#因为dt.hour函数从0开始，所以这里需要+1，目前发现只保留b+t就行，不给自己找麻烦。
                        if behaviors>=241 and behaviors<=10*24+24:
                            emil+=1
                            print(la[i])
                    #后期可能考虑提取http关键词之类的
                        if behaviors>=145 and behaviors<=168:#file
                        # fre=collections.Counter(re.split(r'[ .//-:;]',seq)[4:])
                        # fre=dict(sorted(fre.items(),key=lambda x: x[1], reverse = True))
                        # chose_words=list(fre.keys())[:self.tf_num]
                        # cho_str=' '.join(chose_words)
                        # print(type(cho_str))
                            #动作时间+内容
                        # newsens=str(behaviors)+" "+cho_str+";"
                            newsens=str(behaviors)+" "
                            str_sent+=newsens
                        elif behaviors>=121 and behaviors<=144:#邮件
                        # fre=collections.Counter(re.split(r'[ .//-:;]',seq)[3:])
                        # fre=dict(sorted(fre.items(),key=lambda x: x[1], reverse = True))
                        # chose_words=list(fre.keys())[:self.tf_num]
                        # cho_str=' '.join(chose_words)
                            #动作时间+内容
                        # newsens=str(behaviors)+" "+str(cho_str)+";"
                            newsens=str(behaviors)+" "
                            str_sent+=newsens
                        elif behaviors>=49 and behaviors<=96:#device
                        # newsens=str(behaviors)+";"
                            newsens=str(behaviors)+" "
                            str_sent+=newsens
                        elif behaviors>=1 and behaviors<=48:#logon
                        # newsens=str(behaviors)+";"
                            newsens=str(behaviors)+" "
                            if behaviors>=25 and behaviors<=48:#如果是logoff操作
                                str_sent+=newsens
                                self.data_seq[file[:-3]].append(str_sent)
                                str_sent=""
                            # ff=re.split('[ ;]',str_sent)
                                #print(len(ff))
                                if 1 in lal:
                                    self.label[file[:-3]].append(1)#给每个序列打标，如果序列里有恶意操作，就是恶意序列
                                elif 2 in lal:
                                    self.label[file[:-3]].append(2)
                                elif 3 in lal:
                                    self.label[file[:-3]].append(3)
                                elif 4 in lal:
                                    self.label[file[:-3]].append(4)
                                else:
                                    self.label[file[:-3]].append(0)
                                lal=[]#记录每个序列的情况
                            else:
                                str_sent+=newsens          
                        elif behaviors>=97 and behaviors<=120:#http
                        # fre=collections.Counter(re.split(r'[ .//-:;]',seq)[3:])
                        # fre=dict(sorted(fre.items(),key=lambda x: x[1], reverse = True))
                        # url=time_info["sens"].split()[2]
                        # chose_words=list(fre.keys())[:self.tf_num]
                        # cho_str=' '.join(chose_words)
                            #动作时间+内容
                            newsens=str(behaviors)+" "
                        # newsens=str(behaviors)+" "+str(cho_str)+";"
                            str_sent+=newsens
                        else:
                            newsens=str(behaviors)+" "
                        # newsens=str(behaviors)+" "+str(cho_str)+";"
                            str_sent+=newsens
        #写入数据，必要时很有用
        csv_filename="/home/lhl/cross_certification/dataset/r5.2/sens_data_multi.csv"   
        attr_names=["name","sens","label"]
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=attr_names)
            writer.writeheader()
            for key in self.data_seq:
                for da,la in zip(self.data_seq[key],self.label[key]):
                    dd={"name":key,"sens":da,"label":la}
                    writer.writerow(dd)
        print("finish!!")
        
      #获取图的，暂时没用了。。。如果建图再参考          
   #-----搁置，目前不知道能干啥-----
    def get_graph_data_sens(self):#以一次logon and log off为序列 #这个是为了得到微调数据集的
        data_list=[]
        graph_list=[]
        for class_path in os.listdir(self.path):
            pp=os.path.join(self.path,class_path)
            for file in os.listdir(pp):
                path=os.path.join(pp,file)
                gs={}
                node=[[],[]]#读出的第一个节点为主节点
                weights=[]#目前的weights策略：操作个数为weights
                with open(path,"r") as f:
                    json_data=json.load(f)
                    str_sent=""
                    num=0
                    for key in json_data:#pc级别
                        if num!=0:
                            node[0].append(0)
                            node[1].append(num)
                        num+=1
                        key_list=[]
                        pc_info=json_data[key]
                        gs[key]=[]
                        wnum=0
                        for time_info in pc_info:#timestamp级别
                            wnum+=1
                            if time_info["type"]=="email":
                                fre=collections.Counter(time_info["sens"].split()[3:])
                                fre=dict(sorted(fre.items(),key=lambda x: x[1], reverse = True))
                                chose_words=list(fre.keys())[:self.tf_num]
                                dt=datetime.datetime.fromtimestamp(time_info["timestamp"])
                                hour=dt.hour
                                act_word=self.act_dict["email"]+hour
                                cho_str=' '.join(chose_words)
                            #动作时间+内容
                                newsens=str(act_word)+" "+cho_str+";"
                            elif time_info["type"]=="device":
                                fre=re.split('[ ;]',time_info["sens"])
                                dt=datetime.datetime.fromtimestamp(time_info["timestamp"])
                                hour=dt.hour
                                act_word=self.act_dict[fre[2]]+hour
                                newsens=str(act_word)+";"
                            elif time_info["type"]=="logon":
                                fre=re.split('[ ;]',time_info["sens"])
                                dt=datetime.datetime.fromtimestamp(time_info["timestamp"])
                                hour=dt.hour
                                act_word=self.act_dict[fre[2]]+hour
                                newsens=str(act_word)+";" 
                                if fre[2]=="Logoff":
                                    str_sent+=newsens
                                    ff=re.split('[ ;]',str_sent)
                                #print(len(ff))
                                    key_list.append(str_sent)
                                    str_sent=""          
                            elif time_info["type"]=="http":
                                fre=collections.Counter(time_info["sens"].split()[3:])
                                fre=dict(sorted(fre.items(),key=lambda x: x[1], reverse = True))
                                url=time_info["sens"].split()[2]
                                chose_words=list(fre.keys())[:self.tf_num]
                                cho_str=' '.join(chose_words)
                                dt=datetime.datetime.fromtimestamp(time_info["timestamp"])
                                hour=dt.hour
                                act_word=self.act_dict["http"]+hour
                            #动作时间+内容
                                newsens=str(act_word)+" "+cho_str+";"
                            str_sent+=newsens
                        if num!=1:
                            weights.append(wnum)
                        if str_sent!="":
                            ff=re.split('[ ;]',str_sent)
                            key_list.append(str_sent)
                        if class_path=="normal":
                            labels=0
                        else:
                            labels=int(pp[pp.find("-")+1:])
                        data_list.append({"name":file[:-6],"pc":key,"data":key_list,"label":labels})
                        gs[key]=key_list
                    total=sum(weights)
                    weights=[(x / total) for x in weights]
                    #node=dgl.graph(node)
                    #g1.ndata['h']=torch.tensor(gs)
                    gs_input_ids=[]
                    gs_attention_masks=[]
                    for key in gs:
                        cnt=0
                        input_ids=[]
                        attention_masks=[]
                        for da in gs[key]:#时间序列
                            encoded_dict=self.tokenizer.encode_plus(da,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 32,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                            truncation=True
                    )
                            cnt+=1
                            # Add the encoded sentence to the list.    
                            input_ids.append(encoded_dict['input_ids'])
                            # And its attention mask (simply differentiates padding from non-padding).
                            attention_masks.append(encoded_dict['attention_mask'])
                        while cnt<=47:
                            encoded_dict=self.tokenizer.encode_plus('',                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 32,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                            truncation=True
                    )
                             # Add the encoded sentence to the list.    
                            input_ids.append(encoded_dict['input_ids'])
                            # And its attention mask (simply differentiates padding from non-padding).
                            attention_masks.append(encoded_dict['attention_mask'])
                            cnt+=1
                        input_ids = torch.cat(input_ids, dim=0)
                        attention_masks = torch.cat(attention_masks, dim=0)
                        gs_input_ids.append(input_ids)
                        gs_attention_masks.append(attention_masks)
                    gs_input_ids= torch.stack(gs_input_ids,dim=0)
                    gs_attention_masks=torch.stack(gs_attention_masks,dim=0)
                    if node==[[],[]]:
                        g = dgl.graph([], num_nodes=1)
                    else:
                        g=dgl.graph((node[0],node[1]),num_nodes=num)
                    #aa=np.array(gs_input_ids)
                    g.ndata['input_ids']= gs_input_ids
                    g.ndata['attention_masks']=gs_attention_masks
                    g.edata['w'] = torch.tensor(weights)
                    graph_list.append({"graph":g,"label":labels})                    
        return data_list,graph_list
    #-----搁置，目前不知道能干啥-----
    def get_graph_train_test(self):#这个是为了得到图分类的数据集的
        all_group={0:[],1:[],2:[],3:[],4:[],5:[]}
        train_label=[]
        train_graph=[]
        train_weights=[]
        test_graph=[]
        test_weights=[]
        test_label=[]
        for da in self.graph_list:
            all_group[da["label"]].append(da["graph"])
        train_dataset=[]
        test_dataset=[]
        for key in all_group:
            choos_num=np.random.randint(0,len(all_group[key]),size=int(7*len(all_group[key])/10))
            #train_data=random.sample(all_group[key],int(7*len(all_group[key])/10))
            #test_data=list(set(all_group[key]).difference(set(train_data)))
            for i in range(len(all_group[key])):
                if i in choos_num:
                    da=all_group[key][i]
                    train_label.append(key)
                    train_dataset.append(da)
                else:
                    da=all_group[key][i]
                    test_label.append(key)
                    test_dataset.append(da)
        return train_dataset,train_label,test_dataset,test_label
                
                
class load_dataset(Dataset):
    def __init__(self, dataset,labels,ways):
        self.data=dataset
        self.labels=labels
        self.max_length=128
        self.ways=ways
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
    def __getitem__(self, index):#语句，node weights
        aa=[]
        #每个data[index]是一个长操作语序，对每个语序分词一下
        if self.ways=="bert":
            aa=self.data[index]
            # for i in range(len(self.data[index])):#目前只有一个，就先这样
            inputs = self.tokenizer(
                self.data[index],
                add_special_tokens=True,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
                # inputs=inputs.reshape(self.batch_size,-1)
                # print(inputs['input_ids'].shape)
                # print(inputs['input_ids'])
            # aa.append(inputs)
            return inputs
        elif self.ways=="unsup":#无监督simcse直接返回值，不需要label
            # for i in range(len(self.data[index])):##这里有问题
            xx=[self.data[index],self.data[index]]
            inputs = self.tokenizer( #(1,512)
                xx,
                add_special_tokens=True,
                padding='max_length',
                max_length=self.max_length,
                truncation=True, 
                return_tensors='pt'
            )
            # print(inputs['input_ids'].shape)
            return inputs
        elif self.ways=="sup":#有监督simcse直接返回值，不需要label
            # for i in range(len(self.data[index])):
            # print(self.data[index])
            inputs = self.tokenizer( #(3,256)
                    [self.data[index][0][0],self.data[index][1][0],self.data[index][2][0]],
                add_special_tokens=True,
                padding='max_length',
                max_length=self.max_length,
                truncation=True, 
                return_tensors='pt'
            )
            return inputs
        elif self.ways=="sup-test":
            inputs = self.tokenizer( #(3,256)
                    [self.data[index][0][0],self.data[index][1][0]],
                add_special_tokens=True,
                padding='max_length',
                max_length=self.max_length,
                truncation=True, 
                return_tensors='pt'
            )
            # print(inputs['input_ids'].shape)
            return inputs    
        # return aa,self.labels[index]
        #return aa
    def __len__(self):
        return len(self.data)
class TestDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def text2id(self, text):
        text_ids = self.tokenizer(text, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
        return text_ids

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.text2id(self.data[index][0]), self.text2id(self.data[index][1]), int(self.data[index][2])##这里不能是这样
    
                            
                            
                            
                            
                            
        
                            
                            
                            
                
                