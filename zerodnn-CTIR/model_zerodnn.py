import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ZERODNN(nn.Module):
    def __init__(self, config):
        super(ZERODNN, self).__init__()
        
        self.s_cnum = config['s_cnum']
        self.u_cnum = config['u_cnum']
        self.all_cnum = self.s_cnum + self.u_cnum 
        self.emb_len = config['emb_len']
        self.st_len = config['st_len']
        self.K = 300
        self.L = 128 
        
        self.batch_size = config['batch_size']
  
        self.linear = nn.Linear(self.K, self.L,bias = True) 
        self.mean = nn.AvgPool1d(self.st_len)
        self.max = nn.MaxPool1d(self.st_len)
        self.softmax = nn.Softmax()
        self.in_linear = nn.Linear(self.K, self.L,bias = False) 
        
        self.cossim = nn.CosineSimilarity(eps=1e-6)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.testmode = config['test_mode']
        
    def forward(self,is_train,  seen_n, utter, intents, embedding,all_emb):
        all_emb=torch.Tensor(all_emb)
        all_emb = all_emb.to(torch.int64)
        if (embedding.nelement() != 0): 
            self.word_embedding = nn.Embedding.from_pretrained(embedding)
        utter = self.word_embedding(utter)      
        intents = self.word_embedding(intents)
        all_intents = self.word_embedding(all_emb)
        all_intents = torch.mean(all_intents,1)
        intents = torch.mean(intents,2)
        utter = utter.transpose(1,2) 
        utter_mean = self.mean(utter) 
        utter_encoder = F.tanh(self.linear(utter_mean.permute(0,2,1))) 
        
        intents = intents.transpose(1,2)
        class_num = list(intents.shape)
        
        int_encoders = [F.tanh(self.in_linear(intents[:,:,i])) for i in range(class_num[2])]
        int_encoders = torch.stack(int_encoders)
        
        
        all_int_encoders = [F.tanh(self.in_linear(i)) for i in all_intents]
        all_int_encoders = torch.stack(all_int_encoders)
         
        sim = [self.cossim(utter_encoder.squeeze(1), yi) for yi in int_encoders]
        sim = torch.stack(sim)
        sim = sim.transpose(0,1)
        y_pred = [self.softmax(r) for r in sim]
        y_pred = torch.stack(y_pred)
        if is_train:
            seen_logits,unseen_logits = y_pred.split(seen_n,1)
        
            seen_logits=torch.sum(seen_logits,dim=1,keepdim=True)/seen_n
            unseen_logits=torch.sum(unseen_logits,dim=1,keepdim=True)/(y_pred.size()[1]-seen_n)
            ifunseen_y_pred = torch.cat([seen_logits,unseen_logits],1)
        else:
            ifunseen_y_pred=0
        return y_pred, ifunseen_y_pred, utter_encoder.squeeze(1)
      
    def loss(self, y_pred, y_true): 
        loss = self.criterion(y_pred, y_true)
        return loss
