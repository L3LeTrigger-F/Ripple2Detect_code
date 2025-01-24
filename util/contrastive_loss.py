from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from load_data import build_dataset,load_dataset
import torch
import random
import logging
from sentence_transformers import LoggingHandler, util
import pandas as pd
from sentence_transformers.losses import CoSENTLoss
from mlm import MLMModel
# Define the model. Either from scratch of by loading a pre-trained model
def get_train_dataset():
    dd_pos=[]
    dd_neg=[]
    la_pos=[]
    la_neg=[]
    dd_train=[]
    num=1
        # in_turn=5      
    evidence_info=pd.read_csv("/home/lhl/cross_certification/dataset/r5.2/sens_data_deive.csv")
   # evi_json={"1-1":[],"2-1":[],"2-2":[],"2-3":[],"3-1":[],"3-2":[],"3-3":[]}
    evi_json={"1-1":[],"2-1":[],"2-2":[],"3-1":[],"3-2":[],"4-1":[]}
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
                        #dd_train.append(InputExample(texts=[da,random.sample(evi_json[key],1)[0]],label=1.0))
                        #dd_train.append(InputExample(texts=[da,random.sample(evi_json[key_neg],1)[0]],label=0.0))
                        dd_train.append(InputExample(texts=[da,random.sample(evi_json[key],1)[0], random.sample(evi_json[key_neg],1)[0]]))
        in_turn-=1
    neg_info=pd.read_csv("/home/lhl/cross_certification/dataset/r5.2/sens_data_multirep4.csv")#这里有问题！！
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
                #dd_train.append([da,random.sample(evi_json[key],1)[0],1])
                #dd_train.append([da,random.sample(neg_sens,1)[0],0])
                #dd_train.append(InputExample(texts=[da,random.sample(evi_json[key],1)[0]],label=1.0))
                #dd_train.append(InputExample(texts=[da,random.sample(neg_sens,1)[0]],label=0.0))
                dd_train.append(InputExample(texts=[da,random.sample(evi_json[key],1)[0], random.sample(evi_json[key_neg],1)[0]]))
                    #dd_train.append([[da],[random.sample(evi_json[key],1)[0]],[neg_sens[a]]])#
                a+=1
                if a>=xxx:
                    break
    return dd_train
def get_eval_dataset():
    evidence_info=pd.read_csv("/home/lhl/cross_certification/dataset/r5.2/sens_data_deive.csv")
    #evi_json={"1-1":[],"2-1":[],"2-2":[],"2-3":[],"3-1":[],"3-2":[],"3-3":[]}
    evi_json={"1-1":[],"2-1":[],"2-2":[],"3-1":[],"3-2":[],"4-1":[]}
    neg_data=[]
    for index,item in evidence_info.iterrows():
        if item[0] in evi_json:
            evi_json[item[0]].append(item[1])
    neg_info=pd.read_csv("/home/lhl/cross_certification/dataset/r5.2/sens_data_multirep4.csv")
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
               # neg_data.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(neg_sens,1)[0]]])
                #neg_data.append(InputExample(texts=[da,random.sample(evi_json[key],1)[0], random.sample(neg_sens,1)[0]]))
                #neg_data.append([da,random.sample(evi_json[key],1)[0],1.0])
                #neg_data.append([da,random.sample(neg_sens,1)[0],0.0])
                a+=1
                if a==xxx:
                    break
                #if a==xxx:
                 #   break
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
                        #neg_data.append([[da],[random.sample(evi_json[key],1)[0]],[random.sample(evi_json[key_neg],1)[0]]])
                        #neg_data.append(InputExample(texts=[da,random.sample(evi_json[key],1)[0], random.sample(evi_json[key_neg],1)[0]]))
                        neg_data.append([da,random.sample(evi_json[key],1)[0],1.0])
                        neg_data.append([da,random.sample(evi_json[key_neg],1)[0],0.0])
                                    # xxx+=1
        in_turn-=1

            # neg_data=random.sample(neg_data,len(dd_train))
            # return dd_train,neg_data
    print(len(neg_data))
    return neg_data
device='cuda:0' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer("/home/lhl/cross_certification/bert-base-uncased")
model.to(device)
train_data = get_train_dataset()
test_data = get_eval_dataset()
# Define your train dataset, the dataloader and the train loss
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=16)
#train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
#test_dataloader= DataLoader(test_dataset, shuffle=True, batch_size=32)
#train_loss = losses.TripletLoss(model=model)
sens1=[]
sens2=[]
labels=[]
for da in test_data:
    sens1.append(da[0])
    sens2.append(da[1])
    labels.append(da[2])
from sentence_transformers.evaluation import BinaryClassificationEvaluator
evaluators=BinaryClassificationEvaluator(sens1,sens2,labels)
#train_loss = losses.CosineSimilarityLoss(model)
train_loss=CoSENTLoss(model)
#evaluator = evaluation.EmbeddingSimilarityEvaluator()
# Tune the model

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)


model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluators,
            epochs=10,
            evaluation_steps=100, 
            warmup_steps=100,
            output_path="/home/lhl/cross_certification/cosnet_record_r5.2")
#### Just some code to print debug information to stdout
