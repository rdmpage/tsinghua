#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Name: Promoter_Transformer_main

Created on Friday July 15 2022

Author: Ruohan Ren
"""


import numpy as np
import math
import torch

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x=np.load("all.npy")


K_MER=3

x1=[]

for i in range (0,len(x)):
    k=K_MER
    x2=[]
    for j in range (0,len(x[i])-k+1):
        x2.append(x[i][j:j+k])
    x1.append(x2)

X=x1


y=np.load("all_y.npy")
y=y.astype(np.float32)
for i in range (0,len(y)):
    y[i]=math.log(y[i], 2)



class mydata(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):

        label = self.labels[idx]
        text = self.text[idx]
        
        return self.text[idx],self.labels[idx]

dataset=mydata(X,y)


word_count = Counter()


for i in range (0, len(dataset)):
    for word in dataset.text[i]:
        word_count[word] = word_count[word]+1

#print(word_count)
#print(dict(word_count))

vocab = dict(word_count)

idx_to_word = list(vocab.keys())
print(idx_to_word[0])  
word_to_idx = {word:i for i, word in enumerate(idx_to_word)}
print(word_to_idx)

#print(len(word_to_idx))
#print(dataset.text[0])
#print(dataset.text[0][1])

'''
for i in range (0,len(dataset)):
    for j in range (0, len(dataset.text[i])):
        onehot=[]
        for k in range (0,len(word_to_idx)):
            if(k==word_to_idx[dataset.text[i][j]]):
                onehot.append(1)
            else:
                onehot.append(0)
        dataset.text[i][j]=onehot
'''

for i in range (0,len(dataset)):
    for j in range (0, len(dataset.text[i])):
        dataset.text[i][j]=word_to_idx[dataset.text[i][j]]

#print(dataset.text[0])
#print(dataset.labels)

dataset.text = np.array(dataset.text)
dataset.text = dataset.text.astype(np.int)
dataset.text = torch.from_numpy(dataset.text).to(device)
dataset.labels = torch.from_numpy(dataset.labels).to(device)

#print(dataset.text[0])
#print(dataset.labels)
#print(dataset[1])

train_dataset=[]
for i in range(0,10000):
    train_dataset.append(dataset[i])

test_dataset=[]
for i in range(10000,10800):
    test_dataset.append(dataset[i])

valid_dataset=[]
for i in range(10800,11884):
    valid_dataset.append(dataset[i])

mutation_dataset=[]
for i in range(11884,61885):
    mutation_dataset.append(dataset[i])

#print(train_dataset[0:2])
#print(len(test_dataset))


#print(train_dataset.labels)
#print(test_dataset[0])

#print(dataset[0:2])
#print(train_dataset[0:2])

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0
                          )


valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0
                          )


test_loader = DataLoader(dataset=test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0
                          )

mutation_loader = DataLoader(dataset=mutation_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0
                          )
#print(list(test_loader))


class Transformer(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
        super().__init__()
        

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=200, nhead=10)

        self.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        self.flatten=nn.Flatten(0,-1)

        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(48*embedding_dim, output_dim)
        
        '''
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        '''

    def forward(self, text):
        
        
        embedded = self.embedding(text)

        '''
        for i in range(len(embedded)):
            for j in range(len(embedded[i])):
                if((i+1)%2==0):
                    embedded[i][j]=embedded[i][j]+math.sin((i+1)/(10000**(2*j/200)))
                if((i+1)%2==1):
                    embedded[i][j]=embedded[i][j]+math.cos((i+1)/(10000**(2*j/200)))
        '''

        output = self.TransformerEncoder(embedded)

        #print(len(output))
        final = self.dropout(output.flatten())

        return self.fc(final)
        

        '''
        embedded = self.embedding(text)
        output, hidden = self.lstm(embedded)
        return self.fc(output[-1,:])
        '''


INPUT_DIM = len(word_to_idx)
EMBEDDING_DIM = 200
HIDDEN_DIM = 200
OUTPUT_DIM = 1

model = Transformer(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
if torch.cuda.is_available():
    model = model.cuda()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

print(next(model.parameters()).device)

import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.0001)

criterion = nn.MSELoss()
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    
    model.train()
    
    for batch_x, batch_y in iterator:
        
        optimizer.zero_grad()
                
        predictions = model(torch.squeeze(batch_x))
        #print(batch_x)
        
        loss = criterion(predictions, batch_y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        x1=[]
        x2=[]
        for batch_x, batch_y in iterator:
            #print(torch.squeeze(batch_x))

            predictions = model(torch.squeeze(batch_x))
            
            loss = criterion(predictions, batch_y)
            x1.append(predictions[0].cpu())
            x2.append(batch_y[0].cpu())
            #print(predictions)
            #print(batch_y)
    

            epoch_loss += loss.item()
            
        #print(x1)
        
        #print(np.array(x1))
        #print(x2)
        
        print(np.corrcoef(x1,x2))

    return epoch_loss / len(iterator)
#, np.corrcoef(x1,x2)[0][1]

def evaluate2(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        x1=[]
        x2=[]
        for batch_x, batch_y in iterator:
            #print(torch.squeeze(batch_x))

            predictions = model(torch.squeeze(batch_x))
            
            loss = criterion(predictions, batch_y)
            x1.append(predictions[0].cpu())
            x2.append(batch_y[0].cpu())
            #print(predictions[0])
            #print(batch_y[0])
    

            epoch_loss += loss.item()
            
        #print(x1)
        #print(x2)
        
        x1=np.array(x1)
        f=open("mutation_prediction.txt","w")
        for i in range(len(x1)):
            print(x1[i],file=f)
        

        #print(np.corrcoef(x1,x2))

    return epoch_loss / len(iterator)

N_EPOCHS = 100

best_valid_loss = float('inf')


#for batch_x, batch_y in train_loader:
#    print(batch_x)


for epoch in range(N_EPOCHS):

    
    train_loss = train(model, train_loader, optimizer, criterion)

    valid_loss = evaluate(model, valid_loader, criterion)

    test_loss = evaluate(model, test_loader, criterion)
    
    #, valid_corre
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'final_model.pt')
        evaluate2(model, mutation_loader, criterion)
    
    print(f'Epoch: {epoch+1:02}')
    #print(f'\tTrain Loss: {train_loss:.3f}')
    #print(f'\t Val. Loss: {valid_loss:.3f} ' )

#test_loss, test_corre = evaluate(model, test_loader, criterion)
#print(test_loss, test_corre)








