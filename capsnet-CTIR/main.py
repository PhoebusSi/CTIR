import os
import random
import time
import datetime
starttime = datetime.datetime.now()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tool
import math
import os
import sys

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

setup_seed(2021)

from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import input_data
import model_torch as model


def setting(data):
    vocab_size, word_emb_size = data['embedding'].shape
    sample_num, max_time = data['x_tr'].shape
    test_num = data['x_te'].shape[0]
    s_cnum = np.unique(data['y_tr']).shape[0]
    u_cnum = np.unique(data['y_te']).shape[0]
    config = {}
    config['keep_prob'] = 0.8 # embedding dropout keep rate
    config['hidden_size'] = 300# embedding vector size
    # 64 for SNIPS; 256 for CLINC
    config['batch_size'] = 64
    config['vocab_size'] = vocab_size # vocab size of word vectors (10,895)
    config['num_epochs'] = 50 # number of epochs
    config['max_time'] = max_time
    config['sample_num'] = sample_num #sample number of training data
    config['test_num'] = test_num #number of test data
    config['s_cnum'] = s_cnum # seen class num
    config['u_cnum'] = u_cnum #unseen class num
    config['word_emb_size'] = word_emb_size # embedding size of word vectors (300)
    # 20 for SNIPS; 60 for CLINC
    config['d_a'] = 20 # self-attention weight hidden units number
    # 10 for SNIPS; 30 for CLINC
    config['output_atoms'] = 10 #capsule output atoms
    config['r'] = 3 #self-attention weight hops
    config['num_routing'] = 2 #capsule routing num
    config['alpha'] = 0.0001 # coefficient of self-attention loss
    # 0.5 for SNIPS, 0.05 for CLINC
    config['lambda_'] = 0.5 # coefficient of the SUID loss in Multi-task loss
    config['margin'] = 1.0 # ranking loss margin
    # 0.0001 for SNIPS, 0.001 for CLINC
    config['learning_rate'] = 0.0001
    config['sim_scale'] = 4 #sim scale
    config['nlayers'] = 2 # default for bilstm
    config['ckpt_dir'] = './saved_models/' #check point dir
    return config

def get_sim(data):
    """
    get unseen and seen categories similarity by the embeddings of labels
    """
    s = normalize(data['all_class_vec'])
    u = normalize(data['uc_vec'])
    sim = tool.compute_label_sim(u, s, config['sim_scale'])
    return sim

def compute_sim_with_SS( sc_intents, uc_intents):
    """
    get unseen and seen categories similarity by SIMILARITY SCORER(Cosine Distances)
    """
    cossim = nn.CosineSimilarity(eps=1e-6)
    sc_intents = torch.from_numpy(sc_intents)
    uc_intents = torch.from_numpy(uc_intents)
    feature_len = sc_intents.shape[1]
    sim = [cossim(xi.view(1,feature_len), yi.view(1,feature_len)) for xi in sc_intents for yi in uc_intents]
    sim = torch.stack(sim)
    sim = sim.view(len(sc_intents),len(uc_intents))
    softmax = nn.Softmax(dim=1)
    return sim.float() 
def evaluate_test(data, config, lstm,embedding,avg_p,best_acc):
    # zero-shot testing state
    x_te = data['x_te']
    y_te_id = data['y_te']
    u_len = data['u_len']
    y_ind = data['s_label']
    # get unseen and seen categories similarity through SS
    sim_ori = compute_sim_with_SS(avg_p[-len(data['uc_vec']):],avg_p)

    total_unseen_pred = np.array([], dtype=np.int64)
    total_y_test = np.array([], dtype=np.int64)
    batch_size  = config['test_num']
    test_batch = int(math.ceil(config['test_num'] / float(batch_size)))
    with torch.no_grad():
        test_batch_f_plot=[]
        test_batch_y_plot=[]
        for i in range(test_batch):
            begin_index = i * batch_size
            end_index = min((i + 1) * batch_size, config['test_num'])
            batch_te_original = x_te[begin_index : end_index]
            batch_len = u_len[begin_index : end_index]
            batch_test = y_te_id[begin_index: end_index]
            batch_len = torch.from_numpy(batch_len)

            # sort by descending order for pack_padded_sequence
            batch_len, perm_idx = batch_len.sort(0, descending=True)
            batch_te = batch_te_original[perm_idx]
            batch_test = batch_test[perm_idx]
            batch_te = torch.from_numpy(batch_te)

            lstm(batch_te, batch_len, embedding)
            attentions, seen_logits, seen_votes, seen_weights_c , test_feature = lstm.attention, lstm.logits, \
                                                                  lstm.votes, lstm.weights_c, lstm.sentence_embedding
            test_avg_att_features = torch.mean(test_feature,1)
            test_batch_f_plot.append(test_avg_att_features)
            test_batch_y_plot.append(batch_test)
            
            sim = np.expand_dims(sim_ori,0)

            sim =  np.tile(sim, [seen_votes.shape[1],1,1])
            sim = np.expand_dims(sim, 0)
            sim = np.tile(sim, [seen_votes.shape[0],1,1,1])
            seen_weights_c = np.tile(np.expand_dims(seen_weights_c, -1), [1,1,1, config['output_atoms']])
            mul = np.multiply(seen_votes, seen_weights_c)

            print(sim.shape,mul.shape)
            # transformation-based model
            unseen_votes = np.matmul(sim, mul)
            

            # routing unseen classes
            torch_unseen_votes = unseen_votes
            u_activations, u_weights_c = update_unseen_routing(torch_unseen_votes, config, 3)
            unseen_logits = torch.norm(u_activations, dim=-1)
            te_logits = unseen_logits
            te_batch_pred = np.argmax(te_logits, 1)
            total_unseen_pred = np.concatenate((total_unseen_pred, te_batch_pred))
            total_y_test = np.concatenate((total_y_test, batch_test))
            print ("           zero-shot intent detection test set performance        ")
            acc = accuracy_score(total_y_test, total_unseen_pred)
            print (classification_report(total_y_test, total_unseen_pred, digits=4))
    
    if acc > best_acc:
        best_acc = acc
    return acc, best_acc

def generate_batch(n, batch_size):
    batch_index = random.sample(range(n), batch_size)
    return batch_index

def _squash(input_tensor):
    norm = torch.norm(input_tensor, dim=2, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (0.5 + norm_squared))


def update_unseen_routing(votes, config, num_routing=3):
    votes_t_shape = [3, 0, 1, 2]
    r_t_shape = [1, 2, 3, 0]
    votes_trans = votes.permute(votes_t_shape)
    num_dims = 4
    input_dim = config['r']
    output_dim = config['u_cnum']
    input_shape = votes.shape
    logit_shape = np.stack([input_shape[0], input_dim, output_dim])
    logits = torch.zeros(logit_shape[0], logit_shape[1], logit_shape[2])
    activations = []


    for iteration in range(num_routing):
        route = F.softmax(logits, dim=2)
        preactivate_unrolled = route * votes_trans
        preact_trans = preactivate_unrolled.permute(r_t_shape)

        # delete bias to fit for unseen classes
        preactivate = torch.sum(preact_trans, dim=1)
        activation = _squash(preactivate)
        # activations = activations.write(i, activation)
        activations.append(activation)
        # distances: [batch, input_dim, output_dim]
        act_3d = torch.unsqueeze(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = act_3d.repeat(tile_shape)
        distances = torch.sum(votes * act_replicated, dim=3)
        logits = logits + distances

    return activations[num_routing-1], route

def sort_batch(batch_x, batch_y, batch_len, batch_ind):
    batch_len_new = torch.from_numpy(batch_len)
    batch_len_new, perm_idx = batch_len_new.sort(0, descending=True)
    batch_x_new = batch_x[perm_idx]
    batch_y_new = batch_y[perm_idx]
    batch_ind_new = batch_ind[perm_idx]

    return torch.from_numpy(batch_x_new), torch.from_numpy(batch_y_new), \
           batch_len_new, torch.from_numpy(batch_ind_new)

if __name__ == "__main__":
    #dataset = 'SNIP'
    #dataset = 'CLINC'
    dataset = sys.argv[1]
    if dataset not in ['SNIP', 'CLINC']:
        print('the input argv[1] is', sys.argv[1])
        print('argv[1] have to be selected in [SNIP, CLINC].')
        assert 0 == 1

    # ================================== data setting =============================
    dataSetting={}
    # 0: ZSID; 1: GZSID
    if sys.argv[2] == 'ZSID':
        dataSetting['test_mode'] = 0
    elif sys.argv[2] == 'GZSID':
        dataSetting['test_mode'] = 1
    else:
        print('argv[1] have to be selected in [ZSID, GZSID].')

    dataSetting['training_prob']=0.7
    dataSetting['test_intrain_prob']=0.3
    dataSetting['dataset']=dataset

    if dataset == 'CLINC':
        dataSetting['data_prefix']='../data/nlu_data/'
        dataSetting['dataset_name']='dataCLINC150.txt'
        dataSetting['add_dataset_name']='clinc_unseen_label_name.txt'
        dataSetting['wordvec_name']='60000_glove.840B.300d.txt'
    if dataset == 'SNIP':
        dataSetting['data_prefix']='../data/nlu_data/'
        dataSetting['dataset_name']='dataSNIP.txt'
        dataSetting['add_dataset_name']='snips_unseen_label_name.txt'
        dataSetting['wordvec_name']='wiki.en.vec'
        # load data

    data = input_data.read_datasets_gen(dataSetting)
    x_tr = data['x_tr']
    y_tr = data['y_tr']
    y_tr_id = data['y_tr']
    y_te_id = data['y_te']
    y_ind = data['s_label']
    s_len = data['s_len']
    embedding = data['embedding']

    x_te = data['x_te']
    u_len = data['u_len']
    # load settings
    config = setting(data)
    if dataset == 'CLINC':
        config['learning_rate'] = 0.001
    elif dataset == 'SNIP':
        config['learning_rate'] = 0.0001

    lambda_ = config['lambda_']
    # Training cycle
    batch_num = int(config['sample_num'] / config['batch_size'])
    overall_train_time = 0.0
    overall_test_time = 0.0

    lstm = model.CapsuleNetwork(config)
    optimizer = optim.Adam(lstm.parameters(), lr=config['learning_rate'])
    if os.path.exists(config['ckpt_dir'] + 'best_model.pth'):
        print("Restoring weights from previously trained rnn model.")
        lstm.load_state_dict(torch.load(config['ckpt_dir'] + 'best_model.pth' ))
    else:
        print('Initializing Variables')
        if not os.path.exists(config['ckpt_dir']):
            os.mkdir(config['ckpt_dir'])

    best_acc = 0
    if dataset == 'SNIP':
        seen_n = 5
        unseen_n = 2 
    elif dataset == 'CLINC':
        seen_n = 50
        unseen_n = 10 

    def collectSamples4classes(y_id, features_tensor):
        """
        Collect samples in different categories( for SIMILARITY SCORER)
        """
        features = features_tensor.detach().numpy()
        features_collected = np.zeros((seen_n+unseen_n,features.shape[1]))
        for i in range(seen_n + unseen_n):
            num = 0
            for j in range(len(y_id)):
                if y_id[j] == i:
                    features_collected[i]=features_collected[i]+features[j]
                    num = num+1
        if not num:
            features_collected[i] = np.zeros((1,features.shape[1]))
        else:
            features_collected[i] = features_collected[i]/num
        return features_collected
    
    
    for epoch in range(config['num_epochs']):
        lstm.train()
        avg_acc = 0.0;
        epoch_time = time.time()
        all_avg_features=[]
        batch_f_plot=[]
        batch_y_plot=[]
        for batch in range(batch_num):

            batch_index = generate_batch(config['sample_num'], config['batch_size'])

            batch_x = x_tr[batch_index]
            batch_y_id = y_tr_id[batch_index]
            batch_len = s_len[batch_index]
            batch_ind = y_ind[batch_index]

            # sort by descending order for pack_padded_sequence
            batch_x, batch_y_id, batch_len, batch_ind = sort_batch(batch_x, batch_y_id, batch_len, batch_ind)
            batch_y_id_SUID = torch.LongTensor(len(batch_y_id))
           
            # get the SUID labels
            for i in range(len(batch_y_id_SUID)):
                if batch_y_id[i] < seen_n:

                    batch_y_id_SUID[i]=int(0)
                else:
                    batch_y_id_SUID[i]=int(1)
            

            features = lstm.forward(batch_x, batch_len,torch.from_numpy(embedding))
            avg_att_features = torch.mean(features,1)
            avg_features = collectSamples4classes(batch_y_id,avg_att_features)
            all_avg_features.append(avg_features)
            

            batch_f_plot.append(avg_att_features)
            batch_y_plot.append(batch_y_id)

            batch_y_ind_SUID = torch.from_numpy(input_data.get_label(batch_y_id_SUID))
            
            # loss of MT 
            loss_val = lstm.loss(lambda_, batch_ind,seen_n,batch_y_ind_SUID)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            clone_logits = lstm.logits.detach().clone()
            clone_SUID_logits = lstm.SUID_logits.detach().clone()
            tr_batch_pred = np.argmax(clone_logits, 1)
            SUID_tr_batch_pred = np.argmax(clone_SUID_logits, 1)
            acc = accuracy_score(batch_y_id, tr_batch_pred)
            SUID_acc = accuracy_score(batch_y_id_SUID, SUID_tr_batch_pred)
            avg_acc += acc
        
        avg_p = np.mean(np.array(all_avg_features),0)

        train_time = time.time() - epoch_time
        overall_train_time += train_time
        print( "------epoch : ", epoch, " Loss: ", loss_val.item(), " Acc:", round((avg_acc / batch_num), 4), " Train time: ", \
                                round(train_time, 4), "--------")

        endtime = datetime.datetime.now()
        print("TRANINGTIME!", (endtime - starttime).seconds)
        lstm.eval()
        cur_acc,best_acc = evaluate_test(data, config, lstm,torch.from_numpy(embedding),avg_p,best_acc)
        if cur_acc > best_acc:
            # save model
            best_acc = cur_acc
            torch.save(lstm.state_dict(), config['ckpt_dir'] + 'best_model.pth')
        print("cur_acc", cur_acc)
        print("best_acc", best_acc)
        test_time = time.time() - epoch_time
        overall_test_time += test_time
        print("Testing time", round(test_time, 4))

    print("Overall training time", overall_train_time)
    print("Overall testing time", overall_test_time)

