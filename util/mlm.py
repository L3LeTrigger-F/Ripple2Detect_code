import torch
#from util.early_stopping import EarlyStopping
from util.early_stopping import EarlyStopping
from transformers import BertModel, BertConfig,BertForMaskedLM
import torch.nn.functional as F
# from loguru import logger
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
class MLMModel(nn.Module):
    def __init__(self, pretrained_bert_path, drop_out) -> None:
        super(MLMModel, self).__init__()

        self.pretrained_bert_path = pretrained_bert_path
        config = BertConfig.from_pretrained(self.pretrained_bert_path)
        config.attention_probs_dropout_prob = drop_out
        config.hidden_dropout_prob = drop_out
        self.bert = BertForMaskedLM.from_pretrained(self.pretrained_bert_path, config=config)
        # self.bertl.load_state_dict(torch.load("/home/c402/LhL/cross_certification/trained_model/bert_MLM/model5.pth"))
        # self.bert=BertModel.from_pretrained(self.pretrained_bert_path, config=config)
    def forward(self, input_ids, attention_mask, token_type_ids, pooling="cls"):
        out = self.bert(input_ids, attention_mask, token_type_ids,labels=input_ids,output_hidden_states=True)
        return out.hidden_states[-1][:, 0,:],out.loss
class TrainMLMModel:
    def __init__(self, pretrained_model_path, model_save_path) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.pretrained_model_path = pretrained_model_path
        self.model_save_path = model_save_path
        self.best_loss = 1e8
        self.lr = 0.0001
        self.dropout = 0.3
        self.model = MLMModel(pretrained_bert_path=self.pretrained_model_path, drop_out=self.dropout).to(self.device)
        # self.model.load_state_dict(torch.load("/home/c402/LhL/cross_certification/trained_model/simcse-sup/simcse_model5.pth"))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.early_stopping=EarlyStopping(patience=3, verbose=True)
    def train(self, train_dataloader):
        self.model.train()
        for batch in range(100):
            loss_list=[]
            for batch_idx, source in enumerate(tqdm(train_dataloader), start=1):
                #这里这个0存疑
                real_batch_num = source.get('input_ids').shape[0]#32
                input_ids = source.get('input_ids').view(real_batch_num, -1).to(self.device)
                attention_mask = source.get('attention_mask').view(real_batch_num, -1).to(self.device)
                token_type_ids =source.get('token_type_ids').view(real_batch_num, -1).to(self.device)
                # out,MLM_loss = self.model(input_ids, attention_mask, token_type_ids) 
                # MLM_loss = self.model(input_ids, attention_mask, token_type_ids)     #(16,768)    
                cls,MLM_loss= self.model(input_ids, attention_mask, token_type_ids)    
                loss_list.append(MLM_loss.cpu().detach().numpy())
                self.optimizer.zero_grad()
                MLM_loss.backward()
                self.optimizer.step()
            print('loss:',np.mean(loss_list))
            self.early_stopping(np.mean(loss_list), self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            # if batch%5==0:
                # name='/home/c402/LhL/cross_certification/trained_model/r5.2/bert_mlm/simcse_model'+str(batch)+".pth"
                # name='/home/lhl/cross_certification/new_train/r4.2/base_model_'+str(batch)+".pth"
                # torch.save(self.model.state_dict(),name)
                # evaluate_model(self.model,train_dataloader,test_loader1,test_loader2,self.device)