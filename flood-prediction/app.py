import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x=pd.read_csv("./tamilnadu.csv")
y=pd.read_csv("./tamilnadu.csv")

y1=list(x["YEAR"])
x1=list(x["Oct-Dec"])
z1=list(x["OCT"])
w1=list(x["NOV"])

flood=[]
Nov=[]
sub=[]


for i in range(0,len(x1)):
    if x1[i]>500:
        flood.append('1')
    else:
        flood.append('0')


for k in range(0,len(x1)):
    Nov.append(z1[k]/3)

for k in range(0,len(x1)):
    sub.append(abs(w1[k]-z1[k]))

df = pd.DataFrame({'flood':flood})
df1=pd.DataFrame({'per_10_days':Nov})

x["flood"]=flood
x["avgnov"]=Nov
x["sub"]=sub


x.to_csv("out.csv")
print((x))

import scipy 
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


X = x.iloc[:,[16,20,21]].values
y1=x.iloc[:,19].values

(X_train, X_test, Y_train, Y_test) = train_test_split(X, y1, random_state=0)

Lr=LogisticRegression()

Lr.fit(X,y1)
print(Lr.score(X,y1))

q1=575
w1=830
e1=760

q2=0
w2=0
e2=0


l=[[q1,w1,e1],[q2,w2,e2],[50,300,205]]

f1=Lr.predict(l)


for i in range(len(f1)):
    
    if len(x1) == 0:
        print('')

    if (int(f1[i])==1):
        print(f1[i],"- possibility of  severe flood")
    else:
        print(f1[i],"- no chance of severe flood")
