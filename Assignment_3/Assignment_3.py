# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:19:40 2020

@author: rkuma
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import combinations
import scipy
import sklearn.metrics as metrics

#from statsmodels.graphics.mosaicplot import mosaic

claim_history = pd.read_csv("C:\\Users\\rkuma\\OneDrive\\Documents\\Courses\\Semester 2\\Machine Learning\\Machine_Learning\\Assignment_3\\claim_history.csv")

data = claim_history[['CAR_TYPE','OCCUPATION','EDUCATION','CAR_USE']]

x = data[['CAR_TYPE','OCCUPATION','EDUCATION']].dropna()
y = data[['CAR_USE']].dropna()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,train_size = 0.75,random_state = 60616,stratify=y)

    
my_tab = pd.crosstab(index=y_train['CAR_USE'],columns='count')
my_tab.columns=['Count']
my_tab.index = ['Private','Commercial']
my_tab['Proportion'] = my_tab['Count']/my_tab['Count'].sum()
#my_tab['count'] = y_train['CAR_USE'].value_counts().index

#1.aTrain data frequency counts and proportion
train_count = list(y_train['CAR_USE'].value_counts())
labels = list(y['CAR_USE'].value_counts().index)
train_freq_table = pd.DataFrame(list(zip(labels,train_count)),columns=['Car_Use','Count'])
train_freq_table['Proportion'] = list(train_freq_table['Count']/train_freq_table['Count'].sum())
a=train_freq_table.set_index('Car_Use')
print("-"*50)
print("Question 1.a")
print("-"*50)
print(a)

#1.bTest data frequency counts and proportion
test_count = list(y_test['CAR_USE'].value_counts())
test_freq_table = pd.DataFrame(list(zip(labels,test_count)),columns=['Car_Use','Count'])
test_freq_table['Proportion'] = list(test_freq_table['Count']/test_freq_table['Count'].sum())
b=test_freq_table.set_index('Car_Use')
print("-"*50)
print("Question 1.b")
print("-"*50)
print(b)

#1.cProbability train
prob_train=a.loc[['Commercial'],['Count']]/(a.loc[['Commercial'],['Count']]+b.loc[['Commercial'],['Count']])
print("-"*50)
print("Question 1.c")
print("-"*50)
print(prob_train)

#1.d Probability test
prob_test = b.loc[['Private'],['Count']]/(b.loc[['Private'],['Count']]+a.loc[['Private'],['Count']])
print("-"*50)
print("Question 1.d")
print("-"*50)
print(prob_test)

data2 = x_train.iloc[:,:]
data2['CAR_USE'] = y_train.loc[:,'CAR_USE']
#Contingency Table
cont_tab=pd.crosstab(data2['CAR_TYPE'],data2['CAR_USE'])
cont_tab['Total'] = cont_tab['Commercial'] + cont_tab['Private']
cont_tab = cont_tab.append(cont_tab.agg(['sum']))
#entropy calcualtion
c=cont_tab.loc[['sum'],['Commercial']]
t=cont_tab.loc[['sum'],['Total']]
p=cont_tab.loc[['sum'],['Private']]
#2.a Entropy of root node
ent = scipy.stats.entropy([c.Commercial/t.Total,p.Private/t.Total], base=2)
print("-"*50)
print("Question 2.a")
print("-"*50)
print(ent)
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
    values = [ent1,ent2,split_ent]
    return values

#Splitting parent node 
comm = ['Minivan','Panel Truck','Pickup','SUV','Sports Car','Van']

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
        
df = pd.DataFrame(columns=['Commercial','Private','Red_entropy'])

for j in range(6):
    re_15 = ent-red_entropy(com_list[0][j],pri_list[0][-(j+1)])[2]
    df = df.append({'Commercial':com_list[0][j],'Private':pri_list[0][-(j+1)],'Red_entropy':re_15},ignore_index=True)

for k in range(15):
    re_24 = ent-red_entropy(com_list[1][k],pri_list[1][-(k+1)])[2]
    df = df.append({'Commercial':com_list[1][k],'Private':pri_list[1][-(k+1)],'Red_entropy':re_24},ignore_index=True)
    
for l in range(10):
    re_33 = ent-red_entropy(com_list[2][l],pri_list[2][-(l+1)])[2]
    df = df.append({'Private':com_list[2][l],'Commercial':pri_list[2][-(l+1)],'Red_entropy':re_33},ignore_index=True)

#2.b split criterion of first layer is maximum reduction in entropy
sc = df.loc[df['Red_entropy'] == float(df.loc[:,'Red_entropy'].max()),['Commercial','Private']]
print("-"*50)
print("Question 2.b")
print("-"*50)
print("Predictor Name: ",data.columns[0])
print("Left Child: ",list(sc['Commercial']))
print("Right Child: ",list(sc['Private']))
#2.c entropy of split of first layer
se = red_entropy(df.iloc[28,0],df.iloc[28,1])[2]
print("-"*50)
print("Question 2.c")
print("-"*50)
print("Entropy of split of First Layer: ",se)
#child node 1
com_split_1 = data2[data2['CAR_TYPE'].isin(['Panel Truck', 'Pickup','Van'])].reset_index().drop(['index'],axis=1)
#child node 2
pri_split_2 = data2[data2['CAR_TYPE'].isin(['Minivan', 'SUV','Sports Car'])].reset_index().drop(['index'],axis=1)

#splitting child node 1
cont_tab2=pd.crosstab(com_split_1['OCCUPATION'],com_split_1['CAR_USE'])
cont_tab2['Total'] = cont_tab2['Commercial'] + cont_tab2['Private']
cont_tab2 = cont_tab2.append(cont_tab2.agg(['sum']))
c1=cont_tab2.loc[['sum'],['Commercial']]
t1=cont_tab2.loc[['sum'],['Total']]
p1=cont_tab2.loc[['sum'],['Private']]

ent_c1 = scipy.stats.entropy([c1.Commercial/t1.Total,p1.Private/t1.Total], base=2)

def red_entropy_c1(c,p):
    co = []
    pr = []
    for x in c:
        co.append(x)
    for y in p:
        pr.append(y)
    com = cont_tab2.loc[co].sum()
    pri = cont_tab2.loc[pr].sum() 
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
    values = [ent1,ent2,split_ent]
    return values

s1_c1 = ['Blue Collar','Clerical','Doctor','Home Maker','Lawyer','Manager','Professional','Student','Unknown']

c1_list = []
p1_list = []
tot = 9
for i in range(1,5,1):
    c1_list.append(combinations_(s1_c1,i))
    p1_list.append(combinations_(s1_c1,tot-i))

df2 = pd.DataFrame(columns=['Commercial','Private','Red_entropy'])

for x in range(9):
    re_18 = ent_c1-red_entropy_c1(c1_list[0][x],p1_list[0][-(x+1)])[2]
    df2 = df2.append({'Commercial':c1_list[0][x],'Private':p1_list[0][-(x+1)],'Red_entropy':re_18},ignore_index=True)

for y in range(36):
    re_27 = ent_c1-red_entropy_c1(c1_list[1][y],p1_list[1][-(y+1)])[2]
    df2 = df2.append({'Commercial':c1_list[1][y],'Private':p1_list[1][-(y+1)],'Red_entropy':re_27},ignore_index=True)
    
for z in range(84):
    re_36 = ent_c1-red_entropy_c1(c1_list[2][z],p1_list[2][-(z+1)])[2]
    df2 = df2.append({'Commercial':c1_list[2][z],'Private':p1_list[2][-(z+1)],'Red_entropy':re_36},ignore_index=True)

for w in range(126):
    re_45 = ent_c1-red_entropy_c1(c1_list[3][w],p1_list[3][-(w+1)])[2]
    df2 = df2.append({'Commercial':c1_list[3][w],'Private':p1_list[3][-(w+1)],'Red_entropy':re_45},ignore_index=True)

# split criterion of child node 1 maximum reduction in entropy
c1_c = df2.loc[df2['Red_entropy'] == float(df2.loc[:,'Red_entropy'].max()),['Commercial','Private']]

#maximum entropy 
c1_e = float(df2.loc[:,'Red_entropy'].max())
#leaf node 1
com_split_3 = com_split_1[com_split_1['OCCUPATION'].isin(['Blue Collar','Student'])].reset_index().drop(['index'],axis=1)

#entropy of leaf node 1
cont_tab4=pd.crosstab(com_split_3['OCCUPATION'],com_split_3['CAR_USE'])
cont_tab4['Total'] = cont_tab4['Commercial'] + cont_tab4['Private']
cont_tab4 = cont_tab4.append(cont_tab4.agg(['sum']))
c3=cont_tab4.loc[['sum'],['Commercial']]
t3=cont_tab4.loc[['sum'],['Total']]
p3=cont_tab4.loc[['sum'],['Private']]
ent_c3 = scipy.stats.entropy([c3.Commercial/t3.Total,p3.Private/t3.Total], base=2)

#leaf node 2
pri_split_4 = com_split_1[com_split_1['OCCUPATION'].isin(['Clerical','Doctor','Home Maker','Lawyer','Manager','Professional','Unknown'])].reset_index().drop(['index'],axis=1)
#entropy of leaf node 2
cont_tab5=pd.crosstab(pri_split_4['OCCUPATION'],pri_split_4['CAR_USE'])
cont_tab5['Total'] = cont_tab5['Commercial'] + cont_tab5['Private']
cont_tab5 = cont_tab5.append(cont_tab5.agg(['sum']))
c4=cont_tab5.loc[['sum'],['Commercial']]
t4=cont_tab5.loc[['sum'],['Total']]
p4=cont_tab5.loc[['sum'],['Private']]
ent_c4 = scipy.stats.entropy([c4.Commercial/t4.Total,p4.Private/t4.Total], base=2)

#splitting child node 2
cont_tab3=pd.crosstab(pri_split_2['EDUCATION'],pri_split_2['CAR_USE'])
cont_tab3['Total'] = cont_tab3['Commercial'] + cont_tab3['Private']
cont_tab3 = cont_tab3.append(cont_tab3.agg(['sum']))

c2=cont_tab3.loc[['sum'],['Commercial']]
t2=cont_tab3.loc[['sum'],['Total']]
p2=cont_tab3.loc[['sum'],['Private']]

ent_c2 = scipy.stats.entropy([c2.Commercial/t2.Total,p2.Private/t2.Total], base=2)

def red_entropy_c2(c,p):
    co = []
    pr = []
    for x in c:
        co.append(x)
    for y in p:
        pr.append(y)
    com = cont_tab3.loc[co].sum()
    pri = cont_tab3.loc[pr].sum() 
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
    values = [ent1,ent2,split_ent]
    return values

s1_c2 = ['Below High School' , 'High School' , 'Bachelors' , 'Masters' , 'Doctors']

c2_list = []
p2_list = []
tot = 5
for i in range(1,3,1):
    c2_list.append(combinations_(s1_c2,i))
    p2_list.append(combinations_(s1_c2,tot-i))

df3 = pd.DataFrame(columns=['Commercial','Private','Red_entropy'])

for x in range(5):
    re_14 = ent_c2-red_entropy_c2(c2_list[0][x],p2_list[0][-(x+1)])[2]
    df3 = df3.append({'Commercial':c2_list[0][x],'Private':p2_list[0][-(x+1)],'Red_entropy':re_14},ignore_index=True)

for y in range(10):
    re_23 = ent_c2-red_entropy_c2(c2_list[1][y],p2_list[1][-(y+1)])[2]
    df3 = df3.append({'Commercial':c2_list[1][y],'Private':p2_list[1][-(y+1)],'Red_entropy':re_23},ignore_index=True)
    
# split criterion of child node 2 maximum reduction in entropy
c2_c = df3.loc[df3['Red_entropy'] == float(df3.loc[:,'Red_entropy'].max()),['Commercial','Private']]

#maximum reduction entropy 
c2_e = float(df3.loc[:,'Red_entropy'].max())
#leaf node 3
com_split_5 = pri_split_2[pri_split_2['EDUCATION'].isin(['Masters'])].reset_index().drop(['index'],axis=1)
#entropy of leaf node 3
cont_tab6=pd.crosstab(com_split_5['EDUCATION'],com_split_5['CAR_USE'])
cont_tab6['Total'] = cont_tab6['Commercial'] + cont_tab6['Private']
cont_tab6 = cont_tab6.append(cont_tab6.agg(['sum']))
c5=cont_tab6.loc[['sum'],['Commercial']]
t5=cont_tab6.loc[['sum'],['Total']]
p5=cont_tab6.loc[['sum'],['Private']]
ent_c5 = scipy.stats.entropy([c5.Commercial/t5.Total,p5.Private/t5.Total], base=2)
#leaf node 4
pri_split_6 = pri_split_2[pri_split_2['EDUCATION'].isin(['Below High School' , 'High School' , 'Bachelors' ,'Doctors'])].reset_index().drop(['index'],axis=1)
#entropy of leaf node 4
cont_tab7=pd.crosstab(pri_split_6['OCCUPATION'],pri_split_6['CAR_USE'])
cont_tab7['Total'] = cont_tab7['Commercial'] + cont_tab7['Private']
cont_tab7 = cont_tab7.append(cont_tab7.agg(['sum']))
c6=cont_tab7.loc[['sum'],['Commercial']]
t6=cont_tab7.loc[['sum'],['Total']]
p6=cont_tab7.loc[['sum'],['Private']]
ent_c6 = scipy.stats.entropy([c6.Commercial/t6.Total,p6.Private/t6.Total], base=2)


l1_counts=com_split_3['CAR_USE'].value_counts()
l2_counts=pri_split_4['CAR_USE'].value_counts()
l3_counts=com_split_5['CAR_USE'].value_counts()
l4_counts=pri_split_6['CAR_USE'].value_counts()

leaf_node_1 = {'Occupation':list(c1_c['Commercial']),'Car_Type':list(sc['Commercial']),'Commercial':l1_counts['Commercial'],'Private':l1_counts['Private']}
leaf_node_2 = {'Occupation':list(c1_c['Private']),'Car_Type':list(sc['Commercial']),'Commercial':l2_counts['Commercial'],'Private':l2_counts['Private']}
leaf_node_3 = {'Education':list(c2_c['Commercial']),'Car_Type':list(sc['Private']),'Commercial':l3_counts['Commercial'],'Private':l3_counts['Private']}
leaf_node_4 = {'Education':list(c2_c['Private']),'Car_Type':list(sc['Private']),'Commercial':l4_counts['Commercial'],'Private':l4_counts['Private']}

d = pd.DataFrame(columns=['Entropy','No of Observations','% of Commercial','Predicted Class'])

entropy = []

entropy.append(list(ent_c3))
entropy.append(list(ent_c4))
entropy.append(list(ent_c5))
entropy.append(list(ent_c6))

n_obs = []

n_obs.append(len(com_split_3))
n_obs.append(len(pri_split_4))
n_obs.append(len(com_split_5))
n_obs.append(len(pri_split_6))

com_obs = []

com_obs.append((round((l1_counts['Commercial']/n_obs[0])*100)))
com_obs.append((round((l2_counts['Commercial']/n_obs[1])*100)))
com_obs.append((round((l3_counts['Commercial']/n_obs[2])*100)))
com_obs.append((round((l4_counts['Commercial']/n_obs[3])*100)))

pred_class = []

for i in range(len(com_obs)):
    if(com_obs[i]>50.0):
        pred_class.append('Commercial')
    else:
        pred_class.append('Private')

com_counts = []

com_counts.append(l1_counts['Commercial'])
com_counts.append(l2_counts['Commercial'])
com_counts.append(l3_counts['Commercial'])
com_counts.append(l4_counts['Commercial'])

pri_counts = []

pri_counts.append(l1_counts['Private'])
pri_counts.append(l2_counts['Private'])
pri_counts.append(l3_counts['Private'])
pri_counts.append(l4_counts['Private'])


index = ['Leaf Node 1','Leaf Node 2','Leaf Node 3','Leaf Node 4']

d['Index'] = index
d['Entropy'] = entropy
d['No of Observations'] = n_obs
d['% of Commercial'] = com_obs
d['Predicted Class'] = pred_class
d['Commercial'] = com_counts
d['Private'] = pri_counts
d.set_index('Index')

print("-"*50)
print("Question 2.e")
print("-"*50)
print(d)

com_split_3['Predicted_Class'] = pred_class[0]
pri_split_4['Predicted_class'] = pred_class[1]
com_split_5['Predicted_Class'] = pred_class[2]
pri_split_6['Predicted_Class'] = pred_class[3]

#2f
def predict_cat(data):
    if data['CAR_TYPE'] in ('Panel Truck', 'Pickup', 'Van'):
        if data['OCCUPATION'] in ('Doctor','Lawyer'):
            return [0.84,0.16]
        else:
            return [0.64,0.46]
    else:
        if data['EDUCATION'] in ('High School','Bachelors'):
            return [0,1]
        else:
            return [0.24,0.76]
    
def decision_tree(data):
    out_data = np.ndarray(shape=(len(data), 2), dtype=float)
    count = 0
    for index, row in data.iterrows():
        probability = predict_cat(data=row)
        out_data[count] = probability
        count += 1
    return out_data

pred_prob_train = decision_tree(data=x_train)
pred_prob_train = pred_prob_train[:, 0]
pred_prob_train = list(pred_prob_train)

thres = x_train['CAR_USE'].value_counts()['Commercial']/len(x_train)

y_train['Pred_prob'] = pred_prob_train

pred_train = []
for i in range(len(y_train)):
    if y_train.iloc[i,1] > thres:
        pred_train.append('Commercial')
    else:
        pred_train.append('Private')

y_train['Predicted'] = pred_train
        
mis_class_train = []

for i in range(len(y_train)):
    if y_train.iloc[i,0] == y_train.iloc[i,2]:
        mis_class_train.append(0)
    else:
        mis_class_train.append(1)

y_train['miss Classified'] = mis_class_train


FalsePositive,TruePositive, Threshold = metrics.roc_curve(y_train['CAR_USE'],y_train['Pred_prob'], pos_label='Commercial')

cutoff = np.where(Threshold > 1.0, np.nan, Threshold)
plt.plot(cutoff, TruePositive, marker = 'o', label = 'True Positive', color = 'blue', linestyle = 'solid')
plt.plot(cutoff, FalsePositive, marker = 'o',label = 'False Positive',color = 'red', linestyle = 'solid')
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True)
print("-"*50)
print("Question 2.f")
print("-"*50)
print("KS Statistic: 0.47")
print("event probability cutoff value 0.62")
plt.show()


#KS Statistic 0.65-0.18 = 0.47 
#event probability cutoff value 0.62

eve_prob_cutoff = 0.65

#Question 3
pred_prob = decision_tree(data=x_test)
pred_prob = pred_prob[:, 0]
pred_prob = list(pred_prob)

y_test['Pred_prob'] = pred_prob

pred = []
for i in range(len(y_test)):
    if y_test.iloc[i,1] > thres:
        pred.append('Commercial')
    else:
        pred.append('Private')

y_test['Predicted'] = pred
        
mis_class = []

for i in range(len(y_test)):
    if y_test.iloc[i,0] == y_test.iloc[i,2]:
        mis_class.append(0)
    else:
        mis_class.append(1)

y_test['miss Classified'] = mis_class
#3.a
#metrics.confusion_matrix(y_test, y_pred_class)
accuracy = metrics.accuracy_score(y_test['CAR_USE'], y_test['Predicted'])
missclass = 1-accuracy
print("-"*50)
print("Question 3.a")
print("-"*50)
print("MissClassification Rate: ",missclass)

#3.b
pred_KS = []
for i in range(len(y_test)):
    if y_test.iloc[i,1] > eve_prob_cutoff:
        pred_KS.append('Commercial')
    else:
        pred_KS.append('Private')
y_test.loc[:,'KS'] = pred_KS
accuracy_KS = metrics.accuracy_score(y_test['CAR_USE'], pred_KS)
missclass_KS = 1-accuracy_KS
print("-"*50)
print("Question 3.b")
print("-"*50)
print("MissClassification Rate with Kolmogorov-Smirnov event probability cutoff value as Threshold: ",missclass_KS)

#3.c
y_comm = 1.0 * np.isin(y_test['CAR_USE'], ['Commercial'])
MSE = metrics.mean_squared_error(y_comm, y_test['Pred_prob'])
RMSE = np.sqrt(MSE)
print("-"*50)
print("Question 3.c")
print("-"*50)
print("Root Mean Squared Error for Test Partition: ",RMSE)

#Grouping Probabilities as commercial and private
com_prob = []
pri_prob = []
for i in range(len(y_test)):
    if(y_test.iloc[i,0]=='Commercial'):
        com_prob.append(y_test.iloc[i,1])
    else:
        pri_prob.append(y_test.iloc[i,1])
#Counting concordant, discordant and tie pairs
con = 0
dis = 0
tie = 0
for i in com_prob:
    for j in pri_prob:
        if(i>j):
            con+=1
        elif(i<j):
            dis+=1
        else:
            tie+=1
#3.d
print("-"*50)
print("Question 3.d")
print("-"*50)
AUC = 0.5 + 0.5*(con-dis)/(con+dis+tie)
print("Area Under Curve in Test Partition: ",AUC)
#3.e
print("-"*50)
print("Question 3.e")
print("-"*50)
GINI = (con-dis)/(con+dis+tie)
print("GINI Coefficient in the Test Partition: ",GINI) 
#3.f 
print("-"*50)
print("Question 3.f")
print("-"*50)        
GKG = (con-dis)/(con+dis)
print("Goodman-Kruskal Gamma statistic in the Test partition: ",GKG)
#3.g
OneMinusSpecificity = np.append([0], FalsePositive)
Sensitivity = np.append([0], TruePositive)
OneMinusSpecificity = np.append(OneMinusSpecificity, [1])
Sensitivity = np.append(Sensitivity, [1])
#ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.axis("equal")
print("-"*50)
print("Question 3.g")
print("-"*50) 
plt.show()
















