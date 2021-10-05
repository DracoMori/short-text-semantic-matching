'''
date: 2021/2/24
author: @流氓兔23333
content: 短文本语义匹配模型构建， bert base chinese 使用
'''

data_path = './data_raw/'
save_path = './temp_results/'


import os
import csv
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.utils import shuffle as reset
from tqdm import tqdm
import transformers
from transformers import *
import torch.nn.functional as F
import sklearn.metrics as ms

# model_name = 'hfl/chinese-xlnet-base'
model_name = 'bert-base-chinese'
cache_dir = 'D:/model_pretrain/nlp'
output_model = './models/model.pth'
best_score = 0
batch_size = 32

class CustomDataset(Data.Dataset):
    def __init__(self, data, maxlen, with_labels=True, model_name='bert-base-chinese', cache_dir=cache_dir):
        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)  
        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent_A = str(self.data.loc[index, 'text_A'])
        sent_B = str(self.data.loc[index, 'text_B'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent_A, sent_B, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,       # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def distribution_plot_TextLength(mark_lens):
    # 评论长度分布图：用于确定 过长文本 and 分词的 max_len
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    sns.distplot(mark_lens, fit=stats.norm, color='g')  # 正太概率密度 / 核密度估计图
    plt.tick_params(labelsize=15)
    plt.show()

def train_test_split(data_df, test_size=0.2, shuffle=True, random_state=None):
    if shuffle:
        data_df = reset(data_df, random_state=random_state)
	
    train = data_df[int(len(data_df)*test_size):].reset_index(drop = True)
    test  = data_df[:int(len(data_df)*test_size)].reset_index(drop = True)

    return train, test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

set_seed(1) # Set all seeds to make results reproducible

# header=-1 表示读取无表头
text_pairs =  pd.read_csv(data_path+'data_train.tsv', sep='\t', header=-1)
text_pairs = text_pairs.rename(columns={0:'text_A', 1:'text_B', 2:'label'})
text_pairs = text_pairs.dropna(axis=0)
# text_pairs.to_csv(data_path+'simtext.csv')

text_pairs.isnull().sum()

textA_len = [len(x) for x in text_pairs['text_A']]
textB_len = [len(x) for x in text_pairs['text_B']]
max([np.array(textA_len).max(), np.array(textB_len).max()])
distribution_plot_TextLength(textA_len)
distribution_plot_TextLength(textB_len)



train_df, val_df = train_test_split(text_pairs, test_size=0.2, shuffle=True, random_state=1)

print("Reading training data...")
dataset_trn = CustomDataset(train_df, maxlen=40, model_name=model_name)
loader_trn = Data.DataLoader(dataset_trn, batch_size=batch_size, shuffle=True)
dataset_val = CustomDataset(val_df, maxlen=40, model_name=model_name)
loader_val = Data.DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=True)

_, batch = next(enumerate(loader_trn))
batch[0]


class MyModel(nn.Module):
    def __init__(self, freeze_bert=False, model_name='bert-base-chinese', hidden_size=768):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, output_hidden_states=True, return_dict=True)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = True

        self.fc = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(hidden_size, 1, bias=False),
            nn.Sigmoid()
        )
  
    def forward(self, input, attn_masks, token_type):
        # bert_out (last_hidden_states, pool, all_hidden_states)
        # last_hidden_states [bs, seq_len, hid_size=768]
        # pool [bs, hid_size=768]
        # all_hidden_states (embedding, 各层的hidden_states, ....)
        bert_out = self.bert(input, token_type_ids=token_type, attention_mask=attn_masks) 
        output = self.fc(bert_out[1])
        return output

def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path+'model_bert.pth')
    print('The best model has been saved')


def train_eval(model, criterion, optimizer, loader_trn, loader_val, epochs=2):
    try:
        checkpoint = torch.load(save_path+'model_bert.pth', map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('-----Continue Training-----')
    except:
        print('No Pretrained model!')
        print('-----Training-----')
    
    model = model.to(device)
    loss_his = []
    for epoch in tqdm(range(epochs)):
        model.train()
        print('eopch: %d/%d'% (epoch, epochs))
        for _, batch in enumerate(loader_trn):
            batch = [x.to(device) for x in batch]
            output = model(batch[0], batch[1], batch[2])

            loss = criterion(output, batch[-1].unsqueeze(1).float())
            loss_his.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 1 == 0:
            print(loss.item())
            eval(model, optimizer, loader_val)
    
    return loss_his


best_score = 0.0
def eval(model, optimizer, loader_val):
    model.eval()
    best_score = 0
    batch = next(enumerate(loader_val))
    batch = [x.to(device) for x in batch]

    with torch.no_grad():
        output = model(batch[0], batch[1], batch[2])
        label_ids = batch[-1].to('cpu').numpy()

        _, prediction = torch.max(F.softmax(output, dim=1), 1)
        pred_val = prediction.data.numpy().squeeze()
        acc_val = ms.accuracy_score(label_ids, pred_val)

    print("Validation Accuracy: {}".format(acc_val))
    global best_score
    if best_score < acc_val:
        best_score = acc_val
        save(model, optimizer)



model = MyModel()
criterion  = nn.BCELoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# [token_ids, attn_masks, token_type_ids, label]

loss_his = train_eval(model, criterion, optimizer, loader_trn, loader_val, epochs=2)




