'''
date: 2021/2/24
author: @流氓兔23333
content: 短文本语义匹配预处理
'''

import pandas as pd
import numpy as np
import os, warnings, pickle
warnings.filterwarnings('ignore')
from tqdm import tqdm


data_path = './data_raw/'
save_path = './temp_results/'


# load data
# header=-1 表示读取无表头
text_pairs =  pd.read_csv(data_path+'data_train.tsv', sep='\t', header=-1)
text_pairs = text_pairs.rename(columns={0:'text_A', 1:'text_B', 2:'label'})
text_pairs.head()

text_new = pd.DataFrame({'text_A':['天知道我会发现一个什么样的大秘密',
        '明天见', '我和女朋友分手了','我把猫打了一顿', '我和客人用普通话交流'], 
        'text_B':['我发现了秘密', '明天我就要离开了',
         '女朋友离我而去','猫被我打了一顿', '客人和我用普通话交流'], 
        'label':[1, 0, 0, 1, 1]})
data_pairs_A, data_pairs_B, label = make_data(text_new, padding_len=20)
text_A, text_B = data_pairs_A[1], data_pairs_B[1]
text_A, text_B = torch.LongTensor(text_A), torch.LongTensor(text_B)


import jieba
from gensim import corpora
def make_data(text_pairs, padding_len=20):
    '''
    (text_A, text_B, label)
    (len_text_A, len_text_B)
    '''
    text_A, text_B, label = [], [], []
    len_text_A, len_text_B = [], []
    masks_A, masks_B = [], []

    for i in tqdm(range(len(text_pairs))):
        try:  
            words_A = jieba.lcut(text_pairs.iloc[i, 0])
            words_B = jieba.lcut(text_pairs.iloc[i, 1])
        except:
            continue
        
        # jieba 分词, 长截断
        text_A.append(words_A[:padding_len])
        text_B.append(words_B[:padding_len])
        # 记录文本长度
        len_text_A.append(len(text_A[-1]))
        len_text_B.append(len(text_B[-1]))
        # padding
        if len_text_A[-1] < padding_len:
            text_A[-1] = text_A[-1] + ['pad']*(padding_len-len_text_A[-1])
        if len_text_B[-1] < padding_len:
            text_B[-1] = text_B[-1] + ['pad']*(padding_len-len_text_B[-1])
        # masks
        masks_A.append([1]*len_text_A[-1] + [0]*(padding_len-len_text_A[-1]))
        masks_B.append([1]*len_text_B[-1] + [0]*(padding_len-len_text_B[-1]))

        # 记录label
        label.append(text_pairs.iloc[i, 2])
            
    from gensim import corpora
    text_all = np.concatenate((text_A, text_B), axis=0)
    dictionary = corpora.Dictionary(text_all)
    word2id = dictionary.token2id
    id2word = dictionary.id2token
    pickle.dump(word2id, open(save_path+'word2id.pkl', 'wb'))
    pickle.dump(id2word, open(save_path+'id2word.pkl', 'wb'))

    text_id_A = [[word2id[word] for word in seq] for seq in text_A]
    text_id_B = [[word2id[word] for word in seq] for seq in text_B]

    return (text_A, text_id_A, masks_A, len_text_A), (text_B, text_id_B, masks_B, len_text_B), label


# (text, text_id, masks, len)
data_pairs_A, data_pairs_B, label = make_data(text_pairs, padding_len=20)
pickle.dump((data_pairs_A, data_pairs_B, label), open(save_path+'data_pairs_all.pkl', 'wb'))


# train test split
data_pairs_all = pickle.load(open(save_path+'data_pairs_all.pkl', 'rb'))

def fun_train_test_split(data_pairs_all, random_seed=666):
    from sklearn.model_selection import train_test_split

    data_pairs_A, data_pairs_B, label = data_pairs_all[0], data_pairs_all[1], data_pairs_all[2]

    text_train, text_val, textid_train, textid_val , mask_train, mask_val, len_train, len_val \
        = train_test_split(data_pairs_A[0], data_pairs_A[1], data_pairs_A[2],data_pairs_A[3],
                                                test_size=0.4, random_state=random_seed)
    data_pairs_A_train = (text_train, textid_train, mask_train, len_train)
    data_pairs_A_val = (text_val, textid_val, mask_val, len_val)

    text_train, text_val, textid_train, textid_val , mask_train, mask_val, len_train, len_val \
        = train_test_split(data_pairs_B[0], data_pairs_B[1], data_pairs_B[2],data_pairs_B[3],
                                                test_size=0.4, random_state=random_seed)
    data_pairs_B_train = (text_train, textid_train, mask_train, len_train)
    data_pairs_B_val = (text_val, textid_val, mask_val, len_val)

    label_train, label_val = train_test_split(label, test_size=0.4, random_state=random_seed)

    return (data_pairs_A_train, data_pairs_B_train, label_train), (data_pairs_A_val, data_pairs_B_val, label_val)

data_pairs_train, data_pairs_val = fun_train_test_split(data_pairs_all, random_seed=666)
pickle.dump((data_pairs_train, data_pairs_val), open(save_path+'data_pairs_train_test_split.pkl', 'wb'))


# =============================================================
# data_pairs_train -> (pairs_A, pairs_B, label)
# pairs_A -> (text, textid, masks, len)
data_pairs_train, data_pairs_val = pickle.load(open(save_path+'data_pairs_train_test_split.pkl', 'rb'))

import torch
import torch.utils.data as Data

fun_type_transform = lambda x: torch.LongTensor(x)

batch_size = 128
# [textid_A, textid_B, masks_A, maska_B, len_A, len_B, label]
dataset_train = Data.TensorDataset(fun_type_transform(data_pairs_train[0][1]), 
                                   fun_type_transform(data_pairs_train[1][1]), 
                                   fun_type_transform(data_pairs_train[0][2]), 
                                   fun_type_transform(data_pairs_train[1][2]),
                                   fun_type_transform(data_pairs_train[0][3]), 
                                   fun_type_transform(data_pairs_train[1][3]), 
                                   fun_type_transform(data_pairs_train[2]))
loader_train = Data.DataLoader(dataset_train, batch_size, True)

dataset_val = Data.TensorDataset(fun_type_transform(data_pairs_val[0][1]), 
                                 fun_type_transform(data_pairs_val[1][1]), 
                                 fun_type_transform(data_pairs_val[0][2]), 
                                 fun_type_transform(data_pairs_val[1][2]),
                                 fun_type_transform(data_pairs_val[0][3]), 
                                 fun_type_transform(data_pairs_val[1][3]), 
                                 fun_type_transform(data_pairs_val[2]))
loader_val = Data.DataLoader(dataset_val, len(data_pairs_val[2]), True)












