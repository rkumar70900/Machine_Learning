# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:19:40 2020

@author: rkuma
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
#from statsmodels.graphics.mosaicplot import mosaic

claim_history = pd.read_csv("C:\\Users\\rkuma\\OneDrive\\Documents\\Courses\\Semester 2\\Machine Learning\\Machine_Learning\\Assignment_3\\claim_history.csv")

data = claim_history[['CAR_TYPE','OCCUPATION','EDUCATION','CAR_USE']]

x = data[['CAR_TYPE','OCCUPATION','EDUCATION']].dropna()
y = data[['CAR_USE']].dropna()

iter_split = StratifiedShuffleSplit(n_splits = 5,test_size = 0.25,train_size = 0.75,random_state = 60616)

iter_split.get_n_splits(x,y)

for train_index, test_index in iter_split.split(x,y):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
my_tab = pd.crosstab(index=y_train['CAR_USE'],columns='count')
my_tab.columns=['Count']
my_tab.index = ['Private','Commercial']
my_tab['Proportion'] = my_tab['Count']/my_tab['Count'].sum()
my_tab['count'] = y_train['CAR_USE'].value_counts().index

#1.aTrain data frequency counts and proportion
train_count = list(y_train['CAR_USE'].value_counts())
labels = list(y['CAR_USE'].value_counts().index)
train_freq_table = pd.DataFrame(list(zip(labels,train_count)),columns=['Use','Count'])
train_freq_table['Proportion'] = list(train_freq_table['Count']/train_freq_table['Count'].sum())

#1.bTest data frequency counts and proportion
test_count = list(y_test['CAR_USE'].value_counts())
test_freq_table = pd.DataFrame(list(zip(labels,test_count)),columns=['Use','Count'])
test_freq_table['Proportion'] = list(test_freq_table['Count']/test_freq_table['Count'].sum())

#1.cProbability train
a=train_freq_table.set_index('Use')
b=test_freq_table.set_index('Use')
prob_train=a.loc[['Commercial'],['Count']]/(a.loc[['Commercial'],['Count']]+b.loc[['Commercial'],['Count']])

#1.d Probability test
prob_test = b.loc[['Private'],['Count']]/(b.loc[['Private'],['Count']]+a.loc[['Private'],['Count']])



#Contingency Table
cont_tab=pd.crosstab(data['CAR_TYPE'],data['CAR_USE'])
cont_tab['Total'] = cont_tab['Commercial'] + cont_tab['Private']
cont_tab = cont_tab.append(cont_tab.agg(['sum']))
#entropy calcualtion
import scipy
c=cont_tab.loc[['sum'],['Commercial']]
t=cont_tab.loc[['sum'],['Total']]
p=cont_tab.loc[['sum'],['Private']]
#2.a Entropy of root node
ent = scipy.stats.entropy([c.Commercial/t.Total,p.Private/t.Total], base=2)


#2.b
index = ['Commercial','Private','Total']
def red_entropy(c,p):
    co = []
    pr = []
    for x in c:
        co.append(x)
    for y in p:
        pr.append(y)
    com = cont_tab.loc[co].sum()
    pri = cont_tab.loc[pr].sum() 
    cont_tab1 = pd.DataFrame(list(zip(com,pri)),columns=[" ".join(co)," ".join(pr)])
    cont_tab1.index = [x for x in index]
    cont_tab1 = cont_tab1.T
    cont_tab1 = cont_tab1.append(cont_tab1.agg(['sum']))
    a = cont_tab1.loc[" ".join(co),index[0]]
    b = cont_tab1.loc[" ".join(co),index[1]]
    d = cont_tab1.loc[" ".join(co),index[2]]
    e = cont_tab1.loc[" ".join(pr),index[0]]
    f = cont_tab1.loc[" ".join(pr),index[1]]
    g = cont_tab1.loc[" ".join(pr),index[2]]
    st = cont_tab1.loc['sum',index[2]]
    ent1 = scipy.stats.entropy([a/d,b/d],base=2)
    ent2 = scipy.stats.entropy([e/g,f/g],base=2)
    split_ent = ((d/st)*ent1+(g/st)*ent2)
    return ent-split_ent


red_entropy(['Van','Panel Truck','Pickup'],['Minivan','SUV','Sports Car'])   


#commercial = [['Panel Truck'],['Minivan','Panel Truck'],['Minivan','SUV','Sports Car'],['Minivan','Panel Truck','Pickup','SUV'],['Minivan','Panel Truck','Pickup','SUV']]
#private = 

from itertools import combinations

comm = ['Minivan','Panel Truck','Pickup','SUV','Sports Car','Van']
#priv = ['Minivan','Panel Truck','Pickup','SUV','Sports Car','Van']

#comm = ['Low','Medium','High']
#priv = ['Low','Medium','High']
#list1_permutations = permutations(comm,1)

all_combinations = []
for comb in combinations(comm,4):
    all_combinations.append(comb)
    

def combinations_(u,i):
    c =[]
    for comb in combinations(u,i):
        c.append(comb)
    return c
    

com_list=[]
pri_list=[]
tot=6
for i in range(1,4,1):
    com_list.append(combinations_(comm,i))
    pri_list.append(combinations_(comm,tot-i))


    























