#按照pc读取句子。用bert学
from transformers import BertModel,BertTokenizer,BertForMaskedLM
import json
import argparse
from util.load_data import build_dataset,load_dataset
from torch import distributed as dist
from model import simcse_lstm_Models
from util.train_unsup import TrainUnsupSimcse,SimcseUnsupModel
from sklearn.metrics.pairwise import cosine_similarity
from util.train_sup import TrainSupSimcse,SimcseSupModel,TrainSentenceBert
import torch
from torch.utils.data import DataLoader
import numpy as np
from util.cons_ex import svm_train
from util.mlm import MLMModel
import util
from sklearn.metrics import roc_auc_score,roc_curve
from util.mlm import TrainMLMModel
import torch.nn.parallel
parser = argparse.ArgumentParser()
#参数
parser.add_argument('--path',type=str,help="file path")
parser.add_argument('--tf_num',type=int,help='tf number')
parser.add_argument('--model_path',type=int,help='model path')
parser.add_argument('--if_dropout',type=bool,help='if use SimCSE')
parser.add_argument('--data_process_mode',type=str,help="data process way")
def cal_score(predictions,labels):
    assert len(predictions) == len(labels)
    # predictions = [1 if p > 0.5 else 0 for p in predictions]
    # 计算准确率
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    total = len(labels)
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    accuracy = correct / total
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    FPR=fp/(tn+fp)
    auc = roc_auc_score(y_true=labels, y_score=predictions)
    # AUC=roc_auc_score(labels, predictions)
    return accuracy, precision, recall, f1_score,FPR,auc
def evaluate_model(model,dataloader,dataloader1,dataloader2,device,f1_best):
    modes="aa"
    if modes=="simcse-sup":
        model=SimcseSupModel('bert-base-uncased',0.3)
        model.load_state_dict(torch.load("/home/lhl/cross_certification/trained_model/simcse-sup/new_train_model160.pth"))
    elif modes=="bert-MLM":
        model=SimcseSupModel('bert-base-uncased',0.3)
        model.load_state_dict(torch.load("/home/lhl/cross_certification/trained_model/simcse-sup/new_train_model160.pth"))        
    model.eval()  # 设置模型为评估模式
    true_labels = []
    predicted_labels = []
    need_to_save=[]
    model=model.to(device)
    cosine_list=[]
    with torch.no_grad():
        for batch in dataloader1:
            ori_input_ids=batch.get('input_ids').to(device)#(64,3,128)
            ori_attention_mask=batch.get('attention_mask').to(device)
            ori_token_type_ids=batch.get('token_type_ids').to(device)
            # aa=torch.index_select(ori_input_ids,1,torch.tensor([0]).to(device)).reshape(64,-1)
            out_ori,_ = model(torch.index_select(ori_input_ids,1,torch.tensor([0]).to(device)).reshape(64,-1),torch.index_select(ori_attention_mask,1,torch.tensor([0]).to(device)).reshape(64,-1), torch.index_select(ori_token_type_ids,1,torch.tensor([0]).to(device)).reshape(64,-1))
            out_pos,_ = model(torch.index_select(ori_input_ids,1,torch.tensor([1]).to(device)).reshape(64,-1),torch.index_select(ori_attention_mask,1,torch.tensor([1]).to(device)).reshape(64,-1), torch.index_select(ori_token_type_ids,1,torch.tensor([1]).to(device)).reshape(64,-1))
            out_neg,_ = model(torch.index_select(ori_input_ids,1,torch.tensor([2]).to(device)).reshape(64,-1),torch.index_select(ori_attention_mask,1,torch.tensor([2]).to(device)).reshape(64,-1),torch.index_select(ori_token_type_ids,1,torch.tensor([2]).to(device)).reshape(64,-1))
            res1=cosine_similarity(out_ori.cpu(),out_pos.cpu())
        #    for a,b,c in zip(out_ori,out_pos,out_neg):
            res2=cosine_similarity(out_ori.cpu(),out_neg.cpu())
            true_labels.extend(64*64*[1])
            true_labels.extend(64*64*[0])
            for da in res1:
                cosine_list.extend(list(da))
            for da in res2:
                cosine_list.extend(list(da))
        min_va=min(cosine_list)
        max_va=max(cosine_list)
        max_f1=0
        max_acc=0
        max_pre=0
        max_rec=0
        min_fpr=0
        max_auc=0
        # ther=min_va+0.01
        ther=0.975
        max_va=0.975
        while ther<=max_va:
            predicted_labels=[]
            for r in cosine_list:
                if r>ther:
                    predicted_labels.append(1)
                else:
                    predicted_labels.append(0)
            accuracy, precision, recall, f1_score,FPR,AUC=cal_score(predicted_labels,true_labels)
            if f1_score>max_f1:
                max_f1=f1_score
                max_acc=accuracy
                max_pre=precision
                max_rec=recall
                min_fpr=FPR
                max_auc=AUC
            ther+=0.01
            '''
            c=0
            for r in res1:
                if r[c]>0.97:
                    predicted_labels.append(1)
                else:
                    predicted_labels.append(0)
                need_to_save.append(float(r[c]))
                c+=1
                
            c=0
            for r in res2:
                if r[c]>0.97:
                    predicted_labels.append(1)
                else:
                    predicted_labels.append(0)
                c+=1
        '''
        '''
        for batch in dataloader2:
            ori_input_ids=batch.get('input_ids').to(device)#(64,3,128)
            ori_attention_mask=batch.get('attention_mask').to(device)
            ori_token_type_ids=batch.get('token_type_ids').to(device)
            out_ori,_ = model(torch.index_select(ori_input_ids,1,torch.tensor([0]).to(device)).reshape(64,-1),torch.index_select(ori_attention_mask,1,torch.tensor([0]).to(device)).reshape(64,-1), torch.index_select(ori_token_type_ids,1,torch.tensor([0]).to(device)).reshape(64,-1))        
            out_neg,_ = model(torch.index_select(ori_input_ids,1,torch.tensor([1]).to(device)).reshape(64,-1),torch.index_select(ori_attention_mask,1,torch.tensor([1]).to(device)).reshape(64,-1),torch.index_select(ori_token_type_ids,1,torch.tensor([1]).to(device)).reshape(64,-1))
            
            for a,b,c in zip(out_ori,out_pos,out_neg):
                res2=cosine_similarity(out_ori.cpu(),out_neg.cpu())
                true_labels.extend(64*[0])
                c=0
                for r in res2:
                    if r[c]>0.99:
                        predicted_labels.append(1)
                    else:
                        predicted_labels.append(0)
                    c+=1
        '''
        print("acc:",max_acc,"pre:",max_pre,"rec:",max_rec,"F1:",max_f1,"FPR:",min_fpr,"AUC:",max_auc)
    return f1_best
def train(train_data,train_loader,test_loader,modes,if_dropout):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cuda:1"
    if modes=="bert":#bert训练好后，再训练一个classification。哦吼就改这个吧
        # model=simcse_lstm_Models("bert-base-uncased","bert","train",64,30,2,1)
        model=SimcseSupModel('/home/lhl/cross_certification/bert-base-uncased',0.3)
        model.load_state_dict(torch.load("/home/lhl/cross_certification/trained_model/simcse-sup/new_train_model160.pth"))
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
        criterion=torch.nn.BCELoss().to(device)
        epoch=50
        #这个就是分类的train #先改好fgm？
        for epo in range(epoch):
            run_loss=[]
            pred_la=[]
            for batch,(x,label) in enumerate(train_loader):
                la= label.to(device)
                optimizer.zero_grad()
                preds,loss=model(x,la,device)
                pred_la.extend(preds.cpu().numpy())
                run_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            if epo%5==0:
                print('%d/50 ,train_loss: %.5f' % (epo,np.mean(run_loss)))
                evaluate_model(model,train_loader,device)
                evaluate_model(model,test_loader,device)
    if modes=="simcse":#无监督simcse，训练过程是微调，单独写。
        train_model=TrainUnsupSimcse("/home/lhl/cross_certification/trained_model/r5.2/bert_mlm/simcse_model40.pth","/home/lhl/cross_certification/trained_model/simcse-sup")
        train_model.train(train_loader,test_loader)
    if modes=="simcse-sup":
        train_model=TrainSupSimcse("/home/lhl/cross_certification/bert-base-uncased","/home/lhl/cross_certification/trained_model/simcse-unsup")
        train_model.train(train_loader,test_loader)
    if modes=="mlm":
        train_model=TrainMLMModel("/home/lhl/cross_certification/bert-base-uncased","/home/lhl/cross_certification/trained_model/bert_mlm")
        train_model.train(train_loader)
    if modes=="sentence-bert":
        train_model=TrainSentenceBert("/home/lhl/cross_certification/bert-base-uncased",train_loader,test_loader,"/home/lhl/cross_certification/trained_model/sentence_bert")
        train_model.train()
    #改变下游分类模型的classification，暂时用不上
    if modes=="classification":
        model=simcse_lstm_Models("princeton-nlp/sup-simcse-roberta-base","classification","train",768,200,2,1)
        #模型根据自己需求改吧
        model=SimcseSupModel("/home/lhl/cross_certification/bert-base-uncased",0.3)
        model.load_state_dict(torch.load("/home/lhl/cross_certification/trained_model/bert_MLM/model40.pth"))
        model=model.to(device)
        epoch=50
        mlp=torch.nn.Linear(768,2)
        mlp=mlp.to(device)
        run_loss=[]
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion=torch.nn.CrossEntropyLoss().to(device)
        for epo in range(epoch):
            for batch,(x,label) in enumerate(train_loader):
                for key in x[0]:
                    x[0][key]=x[0][key].reshape(16,-1)#这里有个batch_size要改
                x[0]=x[0].to(device)
                la= label.to(device)
                optimizer.zero_grad()
                aa=model(x[0]['input_ids'],x[0]['attention_mask'],x[0]['token_type_ids'])#64,768
                preds=mlp(aa)
                loss=criterion(preds,la)
                # pred_la.extend(preds.cpu().numpy())
                run_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            if epo%5==0:
                print('%d/50 ,train_loss: %.5f' % (epo,np.mean(run_loss)))
                evaluate_model(model,mlp,train_loader,device)
                evaluate_model(model,mlp,test_loader,device)
if __name__=='__main__':
    args=parser.parse_args()
    path="/home/lhl/cross_certification/dataset/r4.2/mal_sec"
    modes="sup"
    tf_num=5
    dataset = build_dataset(path,tf_num,"sup")
    da_test=build_dataset(path,tf_num,"sup-test")
    train_data=[]
    data_process="sup"
    if data_process=="bert":#bert是为了直接做classification
    #其他训练时需要
        train_dataset=load_dataset(dataset.simcse_data,data_process,"bert")
        # test_dataset=load_dataset(dataset.test_dataset,dataset.test_label,"bert")
    elif data_process=="unsup":
        train_dataset=load_dataset(dataset.simcse_data,data_process,"unsup")
        #不知道放啥先填这个
        test_dataset=load_dataset(da_test.neg_data,data_process,"sup")
    elif data_process=="sup":
        train_dataset=load_dataset(dataset.simcse_data,data_process,"sup")
        #不知道放啥先填这个
        test_dataset2=load_dataset(da_test.neg_data,data_process,"sup")
        #test_dataset1=load_dataset(da_test.simcse_data,data_process,"sup")
    elif data_process=="sup-test":
        train_dataset=load_dataset(dataset.simcse_data,data_process,"sup")
        #不知道放啥先填这个
        test_dataset2=load_dataset(da_test.neg_data,data_process,"sup")
     
    #只有小样本需要
    #graph等等再看吧，目前用不上
    #train_loader=GraphDataLoader(train_dataset,batch_size=2,drop_last=False,shuffle=True)
    #test_loader=GraphDataLoader(test_dataset,batch_size=2,drop_last=False,shuffle=True)
    train_loader=DataLoader(train_dataset,batch_size=128,drop_last=True,shuffle=True)
    # train_loader=DataLoader(cons_dataset,batch_size=16,drop_last=True,shuffle=True)
   # test_loader1=DataLoader(test_dataset1,batch_size=64,drop_last=True,shuffle=True)
    test_loader2=DataLoader(test_dataset2,batch_size=128,drop_last=True,shuffle=True)
    if modes=="sup-test":
        # model=SimcseSupModel('bert-base-uncased',0.3)
        # model.load_state_dict(torch.load("/home/lhl/cross_certification/trained_model/r5.2/simcsefinetune15.pth"))
        model = MLMModel(pretrained_bert_path="/home/lhl/cross_certification/bert-base-uncased", drop_out=0.3).to("cuda:0")
        model.load_state_dict(torch.load("/home/lhl/cross_certification/new_train/r4.2/base_model.pth"))
        # evaluate_model(model,train_loader,test_loader2,test_loader2, torch.device("cuda:0"),0)
        evaluate_model(model,test_loader2,train_loader,train_loader, torch.device("cuda:0"),0)
    # train(train_data,train_loader,test_loader1,test_loader2,"mlm",True)
    train(train_data,train_loader,train_loader,"simcse-sup",True)
    