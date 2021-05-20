from model_zerodnn import ZERODNN
from random import *
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import input_data
import sys
import torch
import numpy as np
import time
import pickle
import torch.optim as optim
import os
import random

def setup_seed(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True

seed = 2021
setup_seed(seed)
#choosedataset = 'CLINC'
#choosedataset = 'SNIP'
choosedataset = sys.argv[1]
if choosedataset not in ['SNIP', 'CLINC']:
    print('the input argv[1] is', sys.argv[1])
    print('argv[1] have to be selected in [SNIP, CLINC].')
    assert 0 == 1
 

# ================================== data setting =============================
dataSetting = {}
#dataSetting['test_mode'] = 0
# 0: ZSID; 1: GZSID
if sys.argv[2] == 'ZSID':
    dataSetting['test_mode'] = 0
elif sys.argv[2] == 'GZSID':
    dataSetting['test_mode'] = 1
else:
    print('argv[1] have to be selected in [ZSID, GZSID].')


dataSetting['training_prob'] = 0.7
dataSetting['test_intrain_prob'] = 0.3
dataSetting['dataset'] = choosedataset

if choosedataset == 'CLINC':
    dataSetting['data_prefix'] = '../data/nlu_data/'
    dataSetting['dataset_name'] = 'dataCLINC150.txt'
    dataSetting['add_dataset_name'] = 'clinc_unseen_label_name.txt'
    dataSetting['wordvec_name'] = '60000_glove.840B.300d.txt'
if choosedataset == 'SNIP':
    dataSetting['data_prefix'] = '../data/nlu_data/'
    dataSetting['dataset_name'] = 'dataSNIP.txt'
    dataSetting['add_dataset_name'] = 'snips_unseen_label_name.txt'
    dataSetting['wordvec_name'] = 'wiki.en.vec'

#=====================================load w2v ================================

# only seen in training process

data = input_data.read_datasets_gen(dataSetting)
x_tr = torch.from_numpy(data['x_tr'])
x_te = torch.from_numpy(data['x_te'])
y_tr = torch.from_numpy(data['y_tr'])
y_tr_id = torch.from_numpy(data['y_tr'])
y_te_id = torch.from_numpy(data['y_te'])
y_ind = torch.from_numpy(data['s_label'])
s_len = torch.from_numpy(data['s_len'])# number of training examples 
embedding = torch.from_numpy(data['embedding'])
u_len = torch.from_numpy(data['u_len'])# number of testing examples 

s_cnum = np.unique(data['y_tr']).shape[0]
u_cnum = np.unique(data['y_te']).shape[0]
y_emb_tr = data['y_emb_tr']
y_emb_te = data['y_emb_te']
vocab_size, word_emb_size = data['embedding'].shape

#============================ cut train data in batch =========================

config = {}
sample_num, max_time = data['x_tr'].shape
test_sample_num, sen = data['x_te'].shape
config['sample_num'] = sample_num #sample number of training data
config['test_sample_num'] = test_sample_num 
config['batch_size'] = 128 
if choosedataset == 'SNIP':
    config['learning_rate'] = 0.01
if choosedataset == 'CLINC':
    config['learning_rate'] = 0.001
#0.01 for SNIP; 0.001 for CLINC
config['seen_class'] = data['seen_class']
config['unseen_class'] = data['unseen_class']
config['emb_len'] = word_emb_size
config['s_cnum'] = s_cnum # seen class num
config['u_cnum'] = u_cnum #unseen class num
config['st_len'] = max_time
config['num_epochs'] = 50 
config['model_name'] = 'ZERO-DNN'
config['dataset'] = choosedataset
config['report'] = True
config['dropout'] = 0.5
config['alpha'] = 0.05
#1 for SNIP ZSID; 0.0125 for SNIP GZSID; 0.05 for both CLINC ZSID&GZSID
config['ckpt_dir'] = './saved_models/' 
config['test_mode'] = dataSetting['test_mode']
config['experiment_time']= time.strftime('%y%m%d%I%M%S')
batch_num = int(config['sample_num'] / config['batch_size'] + 1)
test_batch_num = int(config['test_sample_num'] / config['test_sample_num'] + 1)
y_emb = torch.from_numpy(np.tile(y_emb_tr, (config['batch_size'], 1, 1)))
all_emb = y_emb_te
y_emb_te = torch.from_numpy(np.tile(y_emb_te,(config['test_sample_num'], 1, 1)))
zerodnn = ZERODNN(config) 
optimizer = optim.Adam(zerodnn.parameters(), lr = config['learning_rate'])

def generate_batch(n, batch_size):
    batch_index = random.sample(range(n), batch_size)
    return batch_index

def evaluate_test(data, config, seen_n, unseen_n):
    
    test_batch_num = int(config['test_sample_num'] / config['test_sample_num'])
    total_unseen_pred = np.array([], dtype=np.int64)
    total_y_test = np.array([], dtype=np.int64)

    with torch.no_grad():
        test_feature_list=[]
        test_y =[]
        for batch in range(test_batch_num):
            batch_index = generate_batch(config['test_sample_num'], config['test_sample_num'])
            batch_x = x_te[batch_index]
            batch_y_id = y_te_id[batch_index]
            y_pred, _, test_features = zerodnn(0, seen_n, batch_x, y_emb_te, embedding, all_emb)
            
            test_feature_list.append(test_features)
            test_y.append(batch_y_id)
            y_pred_id = torch.argmax(y_pred, dim=1)
            total_unseen_pred = np.concatenate((total_unseen_pred,  y_pred_id))
            total_y_test = np.concatenate((total_y_test, batch_y_id))

        print('        '+config['model_name']+" "+ config['dataset']+" ZStest Perfomance        ")
        acc = accuracy_score(total_y_test, total_unseen_pred)
        print (classification_report(total_y_test, total_unseen_pred, digits=4))
        logclasses = precision_recall_fscore_support(total_y_test, total_unseen_pred)
    return acc, logclasses

#===================================train=====================================

zerodnn.train()
i = 0
avg_acc = 0.0
test_avg_acc = 0.0
log=[]
logForClasses=[]
best_acc = 0
tr_best_acc = 0
tr_min_loss = np.inf
curr_step = 0
overall_train_time = 0.0
overall_test_time = 0.0
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
filename=config['ckpt_dir']+'mode'+str(config['test_mode'])+'_'+\
        config['dataset']+'_'+config['model_name']+'_'+config['experiment_time']+'.pkl'
if choosedataset == 'SNIP':
    seen_n = 5
    unseen_n = 2
elif choosedataset == 'CLINC':
    seen_n = 50
    unseen_n = 10
  
for epoch in range(config['num_epochs']):
    avg_acc = 0.0
    scheduler.step()
    epoch_time = time.time()      
    print("==================epoch ", epoch, "======================")
    for batch in range(batch_num):
        batch_index = generate_batch(config['sample_num'], config['batch_size'])
        batch_x = x_tr[batch_index]
        batch_y_id = y_tr_id[batch_index]
        batch_len = s_len[batch_index]
        batch_ind = y_ind[batch_index]
        batch_y_id_ifunseen = torch.LongTensor(len(batch_y_id))
        for i in range(len(batch_y_id_ifunseen)):
            if batch_y_id[i]<seen_n:

                batch_y_id_ifunseen[i]=int(0)
            else:
                batch_y_id_ifunseen[i]=int(1)
        y_pred ,ifunseen_pred,_= zerodnn.forward(1,seen_n,batch_x, y_emb, embedding,all_emb)


        #============================  MT =======================================
        
        loss1 = zerodnn.loss(y_pred, batch_y_id)
        loss2 = zerodnn.loss(ifunseen_pred, batch_y_id_ifunseen)
        loss = loss1+config['alpha']*loss2

        y_pred_id = torch.argmax(y_pred, dim = 1)
        acc = accuracy_score(y_pred_id, batch_y_id)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_acc += acc 
            
    train_time = time.time() - epoch_time
    overall_train_time += train_time

    #============================  test =======================================
    
    print("===========================test====================================")
    cur_acc, logC = evaluate_test(data, config, seen_n, unseen_n)
    logForClasses.append(logC)

    print("-----epoch : ", epoch, "/", config['num_epochs'], " Loss: ", loss.item(), \
                  " Acc:", round((avg_acc / batch_num), 4), "TestACC:", round(cur_acc, 6), \
                  "Train_time: ", round(train_time, 4), "overall train time: ", \
                  round(overall_train_time, 4), '-----')
    log.append(dict(epoch=epoch, loss=loss.item(), acc_tr=round((avg_acc / batch_num), 8), acc_te=round(cur_acc,8)))
    
    # early stop
    if (avg_acc / batch_num >= tr_best_acc and loss.item() <= tr_min_loss) or \
    (abs(avg_acc / batch_num - tr_best_acc) < 0.005 and loss.item() <= tr_min_loss):
        tr_best_acc = avg_acc / batch_num
        tr_min_loss = loss.item()
        if config['report']:
            torch.save(zerodnn.state_dict(), config['ckpt_dir'] + 'best_model'+config['experiment_time']+'.pth')
            if cur_acc > best_acc:
                # save model
                best_acc = cur_acc
                config['best_epoch']=epoch
                config['best_acc']=best_acc
            print("cur_acc", cur_acc)
            print("best_acc", best_acc)
            curr_step = 0
    else:
        curr_step += 1
        if curr_step > 5:
            print('curr_step: ', curr_step)
            if curr_step == 10:
                print('Early stop!')                    
                print("Overall training time", overall_train_time)
                print("Overall testing time", overall_test_time)
                # output log
                if config['report']:

                    config['overall_train_time'] = overall_train_time
                    config['overall_test_time'] = overall_test_time
                    '''pickle.dump([config,data['sc_dict'],data['uc_dict'],\
                                 log,logForClasses],open(filename, 'wb')) ''' 
                                      
                break
