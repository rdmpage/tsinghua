"""
Name: Promoter_Transformer_mutation

Created on Friday July 15 2022

Author: Ruohan Ren
"""
f = open ("PnisA.txt","r")

if f.mode=="r":
    contents=f.read()

text=list(contents)

x1=text

import numpy as np

xx=[]
xx=np.array(xx)

#print("".join(x1))

xx=np.append(xx,"".join(x1))

for i in range(50000):   
    
    f = open ("PnisA.txt","r")

    if f.mode=="r":
        contents=f.read()

    text=list(contents)

    a=list('agtttgttagatacaatgatttcgttcgaaggaactacaaaataaattat') #PnisA sequence
    #print(a)
    a=text
    for j in range (0,5):
        r1=np.random.randint(1,51,1)[0]
        print(r1)
        r2=np.random.randint(1,5,1)[0]
        print(r2)
        if (r2==1):
            a[r1-1]='a'
        if (r2==2):
            a[r1-1]='g'
        if (r2==3):
            a[r1-1]='c'
        if (r2==4):
            a[r1-1]='t'

    #print("".join(a))
    xx=np.append(xx,"".join(a))

#print(xx)
np.save('mutation.npy',xx)