""" input data preprocess.
"""
import numpy as np
import tool
from gensim.models.keyedvectors import KeyedVectors
import math
from collections import Counter
from random import *

def load_w2v(file_name):
    """ load w2v model
        input: model file name
        output: w2v model
    """
    w2v = KeyedVectors.load_word2vec_format(file_name, binary=False)
    return w2v

def process_label(intents, w2v,class_id_startpoint=0):
    """ pre process class labels
        input: class label file name, w2v model
        output: class dict and label vectors
    """
    class_dict = {}
    label_vec = []
    class_id = class_id_startpoint
    for line in intents:
        # check whether all the words in w2v dict
        line=line[0]
        label = line.split(' ')
        for w in label:
            if not w in w2v.vocab:
                print( "not in w2v dict", w)

        # compute label vec
        label_sum = np.sum([w2v[w] for w in label if w in w2v.vocab], axis = 0)
        label_vec.append(label_sum)
        # store class names => index
        class_dict[' '.join(label)] = class_id
        class_id = class_id + 1
    return class_dict, np.asarray(label_vec)

def load_vec(file_path, w2v, in_max_len):
    """ load input data
        input:
            file_path: input data file
            w2v: word2vec model
            max_len: max length of sentence
        output:
            input_x: input sentence word ids
            input_y: input label ids
            s_len: input sentence length
            max_len: max length of sentence
    """
    input_x = [] # input sentence word ids
    input_y = [] # input label ids
    s_len = [] # input sentence length
    class_dict=[] 
    max_len = 0

    for line in open(file_path,'rb'):
        arr =str(line.strip(),'utf-8')
        #arr = line.strip().split('\t')
        arr = arr.split('\t')
        label = [w for w in arr[0].split(' ')]
        question = [w for w in arr[1].split(' ')]
        if len(label)>1:
            label=[' '.join(label)]
        if not label in class_dict:
            class_dict.append(label)
            

        # trans words into indexes
        x_arr = []
        for w in question:
            if w in w2v.vocab:
                x_arr.append(w2v.vocab[w].index)
        s_l = len(x_arr)
        if s_l < 1:
            continue
        if in_max_len == 0: # can be specific max len
            if s_l > max_len:
                max_len = s_l
        
        input_x.append(np.asarray(x_arr))
        input_y.append(np.asarray(label))
        s_len.append(s_l)

    # add paddings
    max_len = max(in_max_len, max_len)
    x_padding = []
    for i in range(len(input_x)):
        if (max_len < s_len[i]):
            x_padding.append(input_x[i][0:max_len])
            continue
        tmp = np.append(input_x[i], np.zeros((max_len - s_len[i],), dtype=np.int64))
        x_padding.append(tmp)

    x_padding = np.asarray(x_padding)    
    input_y = np.asarray(input_y)
    s_len = np.asarray(s_len)


    return x_padding, input_y, class_dict, s_len, max_len

def get_label(Ybase):
    sample_num = Ybase.shape[0]
    labels = np.unique(Ybase)
    class_num = labels.shape[0]
    # get label index
    ind = np.zeros((sample_num, class_num), dtype=np.float32)
    for i in range(class_num):
        ind[np.hstack(Ybase == labels[i]), i] = 1;
    return ind



def label_emb(y_tr, w2v):
    
    y_tr_emb = []
   
    for w in y_tr:
        if w in w2v.vocab:
            y_tr_emb.append(w2v.vocab[w].index)
  
    return y_tr_emb


def read_datasets_gen(dataSetting):

    data_path = dataSetting['data_prefix'] + dataSetting['dataset_name']
    add_data_path = dataSetting['data_prefix'] + dataSetting['add_dataset_name']
    word2vec_path = dataSetting['data_prefix'] + dataSetting['wordvec_name']

    print ("------------------read datasets begin-------------------")
    data = {}

    # load word2vec model
    print ("------------------load word2vec begin-------------------")
    w2v = load_w2v(word2vec_path)
    print ("------------------load word2vec end---------------------")

    # load normalized word embeddings

    embedding = w2v.syn0
    data['embedding'] = embedding
    #norm_embedding = tool.norm_matrix(embedding)
    max_len = 0
    x, y, class_set, s_lens, max_len = load_vec(data_path, w2v, max_len)
    add_x, add_y, add_class_set, add_s_lens, add_max_len = load_vec(add_data_path, w2v, max_len)
    # split training set and test set
    label_len=len(class_set)    
    no_class_tr = math.ceil(label_len*dataSetting['training_prob'])
    seen_class = class_set[0:no_class_tr]
    unseen_class = class_set[no_class_tr:]
    if dataSetting['dataset']=='SNIP':
        seen_class=[['search creative work'], ['search screening event'], ['play music'], ['get weather'], ['book restaurant']]
        unseen_class=[['add to playlist'], ['rate book']]
    elif dataSetting['dataset']=='CLINC':
        unseen_class = [[ 'cancel reservation'],['freeze account'],[ 'current location'],['how old are you'],[ 'what is your name'],['reset settings'],[ 'travel alert'],[ 'bill due'],[ 'exchange rate'],[ 'shopping list']]
        seen_class=[['jump start'],['where are you from'],['meaning of life'],['what are your hobbies'],[ 'who do you work for'],['find phone'],[ 'whisper mode'],['translate'],[ 'definition'],['next song'],[ 'account blocked'],[ 'who made you'],['pin change'],[ 'insurance'],[ 'next holiday'],[ 'make call'],[ 'insurance change'],[ 'schedule meeting'],[ 'restaurant suggestion'],['roll dice'],[ 'play music'],['repeat'],[ 'calendar update'],[ 'todo list'],[ 'flip coin'],[ 'calories'],[ 'interest rate'],[ 'ingredient substitution'],[ 'plug type'],[ 'book hotel'],['alarm'],[ 'taxes'],[ 'expiration date'],[ 'schedule maintenance'],[ 'lost luggage'],[ 'car rental'],[ 'book flight'],[ 'international visa'],[ 'reminder'], [ 'share location'],[ 'what song'],[ 'update playlist'],[ 'change language'],[ 'min payment'],[ 'direct deposit'],['confirm reservation'],['weather'],[ 'tell joke'],['spending history'],[ 'change user name']]

   
    ind_te = []
    # add unseen classes samples into Test set
    if dataSetting['test_mode']==0:
        for i in range(len(unseen_class)):
            ind_te.extend([indx for indx, j in enumerate(y) if j == unseen_class[i][0]])
    # shuffle unseen classes samples and add test_intrain_prob% of them into Test set
    elif dataSetting['test_mode']==1:
        for i in range(len(unseen_class)):
            ind_te_tmp=[indx for indx, j in enumerate(y) if j == unseen_class[i][0]]
            np.random.shuffle(ind_te_tmp)
            no_sample_temp=int(len(ind_te_tmp)*dataSetting['test_intrain_prob'])
            ind_te_temp=ind_te_tmp[0:no_sample_temp-1]
            ind_te.extend(ind_te_temp)
    

    
    ind_tr = []
    ind_add_tr = []
    #the test_mode contral the mode how select the train_class and test_class!
    if dataSetting['test_mode']==0:
        for i in range(len(seen_class)):
            temp = [indx for indx, j in enumerate(y) if j == seen_class[i][0]]
            ind_tr.extend(temp) 
        #add the unseen label names into training set
        for i in range(len(unseen_class)):
            add_temp = [indx for indx, j in enumerate(add_y) if j == unseen_class[i][0]]
            ind_add_tr.extend(add_temp)
    elif dataSetting['test_mode']==1:
    # split samples(0.7vs0.3) with seen class into trainingset and test set
        for i in range(len(seen_class)):
            ind_temp = [indx for indx, j in enumerate(y) if j == seen_class[i][0]]
            np.random.shuffle(ind_temp)
            no_sample_temp=int(len(ind_temp)*dataSetting['test_intrain_prob'])
            ind_te_temp=ind_temp[0:no_sample_temp-1]
            ind_tr_temp=ind_temp[no_sample_temp:]
            ind_te.extend(ind_te_temp)
            ind_tr.extend(ind_tr_temp)
        #add the unseen label names(0.3) into training set    
        for i in range(len(unseen_class)):
            ind_add_temp =  [indx for indx, j in enumerate(add_y) if j == unseen_class[i][0]]
            np.random.shuffle(ind_add_temp)
            no_sample_add_temp=int(len(ind_add_temp)*dataSetting['test_intrain_prob'])
            ind_tr_add_temp=ind_add_temp[no_sample_add_temp:]
            ind_add_tr.extend(ind_tr_add_temp) 
   
    x_tr=x[ind_tr,:]
    y_tr=y[ind_tr,:]
    s_len=s_lens[ind_tr]
    x_tr_add = add_x[ind_add_tr,:]
    y_tr_add = add_y[ind_add_tr,:]
    s_len_add = add_s_lens[ind_add_tr]
    x_tr=np.concatenate((x_tr,x_tr_add),axis=0)
    s_len=np.concatenate((s_len,s_len_add),axis=0)
    y_tr=np.concatenate((y_tr,y_tr_add),axis=0)
    resort_inds =[x for x in range(len(x_tr))]
    np.random.shuffle(resort_inds)
    x_tr=x_tr[resort_inds,:]
    y_tr=y_tr[resort_inds,:]
    s_len=s_len[resort_inds]
    
    x_te=x[ind_te,:]
    y_te=y[ind_te,:]
    u_len=s_lens[ind_te]
    resort_inds = [x for x in range(len(x_te))]
    np.random.shuffle(resort_inds)
    x_te=x_te[resort_inds,:]
    y_te=y_te[resort_inds,:]
    u_len=u_len[resort_inds]
    # pre process seen and unseen labels    
    class_id_startpoint=0
    sc_dict, sc_vec = process_label(seen_class, w2v,class_id_startpoint)

    if dataSetting['test_mode']== 0:
        uc_dict, uc_vec = process_label(unseen_class, w2v,class_id_startpoint)
    elif dataSetting['test_mode']==1:
        uc_dict, uc_vec = process_label(unseen_class, w2v,class_id_startpoint+len(sc_dict))
        uc_dict=dict(sc_dict,**uc_dict)
        uc_vec = np.concatenate([sc_vec,uc_vec],axis=0)

    all_dict={}
    num = 0
    for i in dict(sc_dict,**uc_dict).keys():
        all_dict[i]=num
        num=num+1

    y_tr=np.ndarray.tolist(y_tr[:,0])
    y_tr=np.asarray([all_dict[i] for i in y_tr])   
    y_te=np.ndarray.tolist(y_te[:,0])
    y_te=np.asarray([uc_dict[i] for i in y_te])   
    
    class_emb=np.concatenate((np.asarray(seen_class),np.asarray(unseen_class)), axis=None)
    
    y_emb_tr = np.asarray(label_emb(np.asarray(class_emb),w2v))

    if dataSetting['test_mode']== 0:
        y_emb_te = np.asarray(label_emb(np.asarray(unseen_class).squeeze(1),w2v))
    elif dataSetting['test_mode']==1:
        y_emb_te = np.asarray(label_emb(np.asarray(class_emb),w2v))
    class_y_emb =  np.asarray(label_emb(np.asarray(class_emb),w2v)) 
    
    

    data['x_tr'] = x_tr
    data['y_tr'] = y_tr
    data['y_emb_tr'] = y_emb_tr 
    data['y_emb_te'] = y_emb_te
    data['class_emb'] = class_y_emb
    data['s_len'] = s_len # number of training examples (9881)
    data['sc_vec'] = sc_vec
    data['sc_dict'] = sc_dict

    data['x_te'] = x_te
    data['y_te'] = y_te

    data['u_len'] = u_len # number of testing examples (3901)
    data['uc_vec'] = uc_vec
    data['uc_dict'] = uc_dict
    
    if dataSetting['test_mode']== 0:
        data['all_class_vec'] =  np.concatenate([sc_vec,uc_vec],axis=0) 
    elif dataSetting['test_mode']==1:
        data['all_class_vec'] =  uc_vec

    data['max_len'] = max_len

    ind = get_label(data['y_tr'])
    data['s_label'] = ind # [0.0, 0.0, ..., 1.0, ..., 0.0]
    
    data['seen_class']=' '.join(list(tool.flatten(seen_class)))
    data['unseen_class']=' '.join(list(tool.flatten(unseen_class)))
    

    print ("------------------read datasets end---------------------")
    return data
