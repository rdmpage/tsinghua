#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Name: kmer2vec

Created on Friday July 15 2022

Author: Ruohan Ren
"""


f=open("seqdump.txt","r")

if f.mode=="r":
    contents=f.read()

text=list(contents)

name=[[]for i in range(30000)]
dna=[[]for i in range(30000)]
n=0
nn=0    
i=0

while text[i]!='!':
    if text[i]=='>':
        m=1
        nn=nn+1
    else:
        m=0
    if m==1:
        j=i+1
        
        while text[j]!='\n':
            name[nn].append(text[j])
            j=j+1
            
        i=j
        m=0
    else:
        j=i
        
        while text[j]!='>' and text[j]!='!':
            if text[j]=='\n':
                
                j=j+1;
                
            else:
                dna[nn].append(text[j])
                j=j+1
        i=j-1
    i=i+1
    

for i in range(1,nn+1):
    dna[i].append('!')


kmer=4
  
dna2=[[]for i in range(30000)]

for i in range(1,nn+1):
    for j in range(0,len(dna[i])-kmer):
        for k in range(j,j+kmer):
            dna2[i].append(dna[i][k])
        dna2[i].append(' ')
   
sentence=[[]for i in range(30000)]

for i in range(1,nn+1):
    sentence[i]="".join(dna2[i])


f=open("5.txt","w")


for i in range(1,nn+1):
    f.write(sentence[i])
    

from gensim.models import word2vec

import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  
sentences = word2vec.Text8Corpus('5.txt')  
model = word2vec.Word2Vec(sentences, sg=1,  window=24,  min_count=1,  negative=5, sample=0.001, hs=0, workers=1,epochs=1,vector_size=100,seed=0)  
model.save('5_txt_word2vec.model') 

model = word2vec.Word2Vec.load('5_txt_word2vec.model')


import numpy as np


str=[[]for i in range(30000)]
for i in range(1,nn+1):
    str[i]=sentence[i].split( )



v=[[]for i in range(30000)]
vc=[[]for i in range(30000)]
for i in range(1,nn+1):
    v[i]=0
for i in range(1,nn+1):
    vc[i]=0
for ii in range(1,nn+1):
    for i in range(0,len(str[ii])):
        if( str[ii][i] in model.wv.key_to_index.keys ()):
            v[ii]=v[ii]+model.wv[str[ii][i]]/len(str[ii])
    for i in range(0,len(str[ii])):
        if( str[ii][i] in model.wv.key_to_index.keys ()):
            vc[ii]=vc[ii]+(model.wv[str[ii][i]]-v[ii])*(model.wv[str[ii][i]]-v[ii])/len(str[ii])



vv=[[]for i in range(30000)]
vv1=[[]for i in range(30000)]
vv2=[[]for i in range(30000)]

for i in range(1,nn+1):
    vv1[i]=np.array(v[i])
    vv2[i]=np.array(vc[i])
    vv[i]=np.hstack((vv1[i],vv2[i]))


sim=[[]for i in range(30000)]

l=2
for i in range(1,nn):
    for j in range(l,nn+1):
        #sim[i].append(1-(np.matmul(vv[i],vv[j]))/(np.linalg.norm(vv[i])*np.linalg.norm(vv[j])))
        sim[i].append(np.linalg.norm(vv[i] - vv[j]))
    l=l+1


for i in range(1,nn):   
    for j in range(0,i+1):
        sim[i].insert(0,-1)

 
f=open("results.txt","w")
for i in range(1,nn+1):
    print("[",i,"] #","".join(name[i]),file=(f))
    
print("[",file=(f),end="")
for i in range(1,nn+1):
    print(i," ",file=(f),end="")
print("]",file=(f))



print("[","1","] ",file=(f))
for i in range(2,nn+1):
    print("[",i,"] ",file=(f),end="")
    for j in range(1,i):    
        print (format(sim[j][i], '0.10f')," ", file=(f),end="")        
    f.write("\n")






