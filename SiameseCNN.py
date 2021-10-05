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


class SiameseCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, tar_size):
        super(SiameseCNN, self).__init__()
        self.num_filters = 128
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.conv = nn.Conv2d(1, self.num_filters, 2, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.num_filters*2*4, tar_size)
    
    def conv_and_pool(self, out_emb, conv, _max=True):
        # x = [batch_size, channel, seq_len-filter_size[0] , 1] -> [batch_size, channel, seq_len-filter_size[0]]
        x = F.relu(conv(out_emb)).squeeze(3)
        # x = [batch_size, channel, 1 , 1] -> x = [batch_size, channel]
        if _max:
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
        else:
            x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, text_A, text_B):
        batch_size = text_A.shape[0]
        embedding_A = self.word_embedding(text_A)
        embedding_B = self.word_embedding(text_B)
        # embedding_A = [batch_size, channel=1, seq_len, embed_size]
        embedding_A = embedding_A.unsqueeze(1)
        embedding_B = embedding_B.unsqueeze(1) # add channel(=1)
        # out = [batch_size, num_filters]
        out_max = self.conv_and_pool(embedding_A, self.conv)
        out_mean = self.conv_and_pool(embedding_A, self.conv, _max=False)
        fea_A = torch.cat([out_max, out_mean], 1)  # [bs, filters*2]

        out_max = self.conv_and_pool(embedding_B, self.conv)
        out_mean = self.conv_and_pool(embedding_B, self.conv, _max=False)
        fea_B = torch.cat([out_max, out_mean], 1)  # [bs, filters*2]

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
    }, save_path+'model_siameseCNN.pth')
    print('The best model has been saved')

def train_eval(model, criterion, optimizer, train_loader, val_loader, epochs=2):

    checkpoint = torch.load(save_path+'model.pth', map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print('-----Training-----')
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
vocab_size = len(word2id)
embed_size = 2 * int(np.floor(np.power(vocab_size, 0.25)))

model = SiameseCNN(embed_size=embed_size, vocab_size=vocab_size, tar_size=tar_size)
criterion  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


train_eval(model, criterion, optimizer, loader_train, loader_val, epochs=100)



loss_his = []
EPOCH = 100
_, batch = next(enumerate(loader_val))
for epoch in range(EPOCH):
    model = model.train()
    print('epoch: (%d/%d)'%(epoch+1, EPOCH))
    for step, (textid_A, textid_B, _, _, _, _, label) in enumerate(tqdm(loader_train)):
        out_put = model(textid_A, textid_B)
        loss = criterion(out_put, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_his.append(loss.item())
    
    if epoch % 1 == 0:
        model = model.eval()
        output = model(batch[0], batch[1])
        _, prediction = torch.max(F.softmax(output, dim=1), 1)
        pred_val = prediction.data.numpy().squeeze()
        acc_test = ms.accuracy_score(batch[-1].numpy(), pred_val)
        print('test_acc: %s             train loss: %f' % (loss.item(), acc_test))


