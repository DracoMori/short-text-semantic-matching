'''
date: 2021/3/4
author: @流氓兔23333
content: 短文本语义匹配模型构建， siamsesCNN
'''

import pandas as pd
import numpy as np
import os, warnings, pickle
warnings.filterwarnings('ignore')
from tqdm import tqdm

data_path = './data_raw/'
save_path = './temp_results/'


word2id = pickle.load(open(save_path+'word2id.pkl', 'rb'))
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



import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics as ms


class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hid_size, tar_size):
        super(SiameseLSTM, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hid_size)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hid_size*2*4, tar_size)
    
    
    def forward(self, text_A, text_B):
        batch_size = text_A.shape[0]
        embedding_A = self.word_embedding(text_A)
        embedding_B = self.word_embedding(text_B)
        # embedding_A = [batch_size, seq_len, embed_size]
        # lstm_A = [batch_size, seq_len, hid_size]
        lstm_A, _ = self.lstm(embedding_A)
        # lstm_A.size(1) = 20
        out_max = F.max_pool1d(input=lstm_A.transpose(1,2), kernel_size=lstm_A.size(1)).squeeze(2)
        out_avg = F.avg_pool1d(lstm_A.transpose(1,2), lstm_A.size(1)).squeeze(2)
        fea_A = torch.cat([out_max, out_avg], 1)  # [bs, hid_size*2]

        lstm_B, _ = self.lstm(embedding_B)
        out_max = F.max_pool1d(lstm_B.transpose(1,2), lstm_B.size(1)).squeeze(2)
        out_avg = F.avg_pool1d(lstm_B.transpose(1,2), lstm_B.size(1)).squeeze(2)
        fea_B = torch.cat([out_max, out_avg], 1)  # [bs, hid_size*2]


        # [bs, filters*2*4]
        fea_all = torch.cat([fea_A, fea_B, torch.abs(fea_A-fea_B), fea_A.mul(fea_B)], 1)
        out = self.dropout(fea_all)
        output = self.fc(out)

        return output


def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path+'model_siameseLSTM.pth')
    print('The best model has been saved')


def train_eval(model, criterion, optimizer, train_loader, val_loader, epochs=2):
    try:
        checkpoint = torch.load(save_path+'model_siameseLSTM.pth', map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('-----Continue Training-----')
    except:
        print('No Pretrained model!')
        print('-----Training-----')
    
    _, batch = next(enumerate(loader_train))
    for epoch in range(epochs):
        model.train()
        print('eopch: %d/%d'% (epoch, epochs))
        for i, batch in enumerate(tqdm(train_loader)):
            logits = model(batch[0], batch[1])
            loss = criterion(logits, batch[-1])
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 1 == 0:
            print(loss.item())
            eval(model, optimizer, val_loader)


def eval(model, optimizer, validation_dataloader):
    model.eval()
    best_score = 0
    _, batch = next(enumerate(validation_dataloader))
    with torch.no_grad():
        output = model(batch[0], batch[1])
        label_ids = batch[-1].numpy()

        _, prediction = torch.max(F.softmax(output, dim=1), 1)
        pred_val = prediction.data.numpy().squeeze()
        acc_val = ms.accuracy_score(label_ids, pred_val)

    print("Validation Accuracy: {}".format(acc_val))
    global best_score
    if best_score < acc_val:
        best_score = acc_val
        save(model, optimizer)



tar_size = 2
hid_size = 64
vocab_size = len(word2id)
embed_size = 2 * int(np.floor(np.power(vocab_size, 0.25)))

model = SiameseLSTM(embed_size=embed_size, vocab_size=vocab_size, hid_size=hid_size, tar_size=tar_size)
criterion  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


train_eval(model, criterion, optimizer, loader_train, loader_val, epochs=20)


model.eval()
with torch.no_grad():
    _, batch = next(enumerate(loader_val))
    output = model(batch[0][index], batch[1][index])
    _, prediction = torch.max(F.softmax(output, dim=1), 1)
    pred_val = prediction.data.numpy().squeeze()
    acc_test = ms.accuracy_score(batch[-1][index].numpy(), pred_val)
    print(acc_test)


import random
index = list(range(40000))
random.shuffle(index)

model.eval()
# with torch.no_grad():
output = model(text_A, text_B)
_, prediction = torch.max(F.softmax(output, dim=1), 1)
pred_val = prediction.data.numpy().squeeze()
print(pred_val)



