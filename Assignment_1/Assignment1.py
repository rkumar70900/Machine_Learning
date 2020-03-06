# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:44:49 2020

@author: rkuma
"""

import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
import numpy as np
import seaborn as sns
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier as kNN

file = "NormalSample.csv"
file2 = "Fraud.csv"
#CSV to DataFrame
NormalSample_df = pd.read_csv(file)
Fraud_df = pd.read_csv(file2)
x=NormalSample_df['x']
print(x)
#Total number of elements in the column 'x'
N = x.count()
lis = x.values.tolist()
lis.sort()
#Median and Median Location
med=st.median(lis)
medloc = lis.index(med)
leftlis = lis[0:medloc]
rightlis = lis[medloc+1:]
#quartile 1 and quartile 3 values
quartile1 = st.median(leftlis)
quartile3 = st.median(rightlis)
#Calculating the IQR Value
IQR = quartile3-quartile1
bw = 2*IQR*(N**(-1/3))
#Question 1.a
print("----------------------------------------------")
print("---------------------question 1.a-------------")
print("----------------------------------------------")
binwidth = np.round(bw,2)
print("Binwidth: ",binwidth)
#question 1.b
print("---------------------------------------")
print("-----------question 1.b----------------")
print("---------------------------------------")
print("Minimum Value: ",min(x))
print("Maximum Value: ",NormalSample_df['x'].max())
#question 1.c
print("-------------------------------------------")
print("-----------------question 1.c--------------")
print("-------------------------------------------")
a = np.floor(min(x))
b = np.ceil(x.max())
print("a: ",a)
print("b: ",b)
#question 1.d
#Histogram Density Estimation Funtion
def hist_den_est(h):
    p = []
    mid = [a+(h/2)]
    while True:
        u = [(x[i]-mid[-1])/h for i in range(len(x))]
        w = [1 if -1/2<i<=1/2 else 0 for i in u]
        p.append(sum(w)/(len(x)*h))
        if mid[-1] >= b:
            break
        else:
            mid.append(mid[-1]+h)
    return p[:-1],mid[:-1]
print("-----------------------------------")
print("-----------question 1.d------------")
print("-----------------------------------")
print("-density estimates,midpoint values-")
den_val,mid_val = hist_den_est(0.25)
dm = zip(den_val,mid_val)
print(list(dm))
#number of bins bin width = 0.25
nobin_1d = int((b-a)/0.25)
plt.figure(figsize=(15,15))
sns.distplot(x,kde=False,bins=nobin_1d)
plt.show()
#question 1.e
print("--------------------------------------")
print("------------question 1.e--------------")
print("--------------------------------------")
print("--density estimates,midpoint values---")
den_val,mid_val = hist_den_est(0.5)
dm=zip(den_val,mid_val)
print(list(dm))
#number of bins bin width = 0.5
nobin_1e = int((b-a)/0.5)
plt.figure(figsize=(15,15))
sns.distplot(x,kde=False,bins=nobin_1e)
plt.show()
#question 1.f
print("---------------------------------------")
print("----------question 1.f-----------------")
print("---------------------------------------")
print("--density estimates,midpoint values----")
den_val,mid_val = hist_den_est(1)
dm=zip(den_val,mid_val)
print(list(dm))
#number of bins bin width = 1
nobin_1f = int((b-a)/1)
plt.figure(figsize=(15,15))
sns.distplot(x,kde=False,bins=nobin_1f)
plt.show()
#question 1.g
print("-------------------------------------")
print("----------question 1.g---------------")
print("-------------------------------------")
print("--density estimates,midpoint values--")
den_val,mid_val = hist_den_est(2)
dm=zip(den_val,mid_val)
print(list(dm))
#number of bins bin width = 2
nobin_1g= int((b-a)/2)
plt.figure(figsize=(15,15))
sns.distplot(x,kde=False,bins=nobin_1g)
plt.show()

#Five number summary and IQR Whiskers
def Five_Number_Summary(f):
    Q1,Median,Q3 = np.percentile(f,[25,50,75])
    minimum = min(x)
    maximum = x.max()
    return Median,Q1,Q3,minimum,maximum
def IQR_Whiskers(g):
    Median,Q1,Q3,Minimum,Maximum = Five_Number_Summary(g)
    low_whisker = Q1 - (1.5*IQR)
    up_whisker = Q3 + (1.5*IQR)
    return low_whisker,up_whisker

#question 2.a
print("-------------------------------------------------")
print("---------------------question 2.a----------------")
print("-------------------------------------------------")
print("Five Number Summary")
print(dict(list(zip(['Median','Q1','Q3','Minimum','Maximum'],Five_Number_Summary(x)))))
print(Five_Number_Summary(x))
print("IQR Whiskers")
print(dict(list(zip(['Lower Whisker','Upper Whisker'],IQR_Whiskers(x)))))

#Grouping Data into 0 and 1
def grouping(d,group):
    global group_0_data
    group_0_data = []
    global group_1_data
    group_1_data= []
    i=0
    while i<len(group):
        if group[i] == 0:
            group_0_data.append(d.iloc[i,2])
        else:
            group_1_data.append(d.iloc[i,2])
        i+=1
    return pd.Series(group_0_data).describe(),pd.Series(group_1_data).describe()
#question 2.b
print("--------------------------------------")
print("----------------question 2.b----------")
print("--------------------------------------")
print(grouping(NormalSample_df,NormalSample_df['group']))
#1.5 IQR Whiskers
print("Group 0 Whiskers") 
print(dict(list(zip(['Lower Whisker','Upper Whisker'],IQR_Whiskers(group_0_data)))))
print("group 1 Whiskers")
print(dict(list(zip(['Lower Whisker','Upper Whisker'],IQR_Whiskers(group_1_data)))))

#question 2.c
print("-----------------------------------")
print("------------question 2.c-----------")
print("-----------------------------------")
plt.figure(figsize=[15,15])
sns.boxplot(data=x)
plt.show()

#question 2.d
print("--------------------------------------------")
print("-----------------question 2.d---------------")
print("--------------------------------------------")
group_data = pd.DataFrame(list(zip(group_0_data,group_1_data)),columns=['Group_0','Group_1'])
plt.figure(figsize=[15,15])
sns.boxplot(data=group_data)
plt.show()
#function outliers
def outliers(data):
    i=0
    outlie = []
    while i<len(data):
        if data[i]<quartile1 - (1.5*IQR) or data[i]>quartile3 + (1.5*IQR):
            outlie.append(data[i])
        i+=1
    return outlie
#outliers
print("Outliers of Data: ",outliers(x))
print("Outliers of Group 0(Zero): ",outliers(group_0_data))
print("Outliers of Group 1(One): ",outliers(group_1_data))
            
#question 3.a
print("----------------------------------------")
print("---------------question 3.a-------------")
print("----------------------------------------")
fraud = Fraud_df.loc[:,'FRAUD']
count = fraud.value_counts()
per_fraud = fraud.value_counts(normalize=True).mul(100).round(4).astype(str)+'%'
percent = pd.DataFrame({'Count':count,'Percentage':per_fraud})
print("Percentage of Fraud investigations "+percent.iloc[1,1])

int_var = Fraud_df.drop(['FRAUD'],axis=1).columns
#question 3.b
print("------------------------------------")
print("-----------question 3.b-------------")
print("------------------------------------")
def interval_variables(f):
    for r in list(f):
        plt.figure(figsize=(15,15))
        sns.boxplot(x=r,y='FRAUD',data=Fraud_df,orient='h')
        plt.show()

interval_variables(int_var)

#question 3.c
#Using First Principle
ortho_mat = Fraud_df.drop(['FRAUD','CASE_ID'],axis=1)
int_var_mat = np.matrix(ortho_mat.values)
print("-----------------------------------------------------------")
print("---------------------------question 3.c.i------------------")
print("-----------------------------------------------------------")
print("Number of Dimensions used: {} ".format(len(ortho_mat.columns)))
print("Input Matix \n",int_var_mat)
print("Number of Dimensions:",int_var_mat.ndim)
print("Number of Rows: ", np.size(int_var_mat,0))
print("Number of Columns: ", np.size(int_var_mat,1))
int_var_trans = int_var_mat.transpose()*int_var_mat
evals,evecs = LA.eigh(int_var_trans)
print("Eigen Values: ",evals)
print("\nEigen Vectors: ",evecs)
transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)))
print("Transformation Matrix: \n",transf)
transf_x = int_var_mat * transf
print("The Transformed Matrix: \n",transf_x)
print("----------------------------------------------------")
print("----------------------question 3.c.ii---------------")
print("----------------------------------------------------")
xtx = transf_x.transpose().dot(transf_x)
print("An Identity Matrix: \n",xtx)

from scipy import linalg as LA2
#Using SciPy Function
orthx = LA2.orth(int_var_mat)
print("The Orthonormalize of int_var_mat: ",orthx)
check = orthx.transpose().dot(orthx)
print("Identity Matrix: \n",check)
print("The Transformed Matrix: \n",orthx)
print("Variables are orthonormal: \n",check)


#question 3.d
kNNSpec = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')
y=np.array(Fraud_df.loc[:,'FRAUD'])
fit_=kNNSpec.fit(int_var_mat,y)
tar_var_pred = fit_.predict(int_var_mat)
print("--------------------------------------")
print("------------question 3.d.i------------") 
print("--------------------------------------")
print("Score: ",fit_.score(int_var_mat,y))
nbrs_un = kNNSpec.fit(transf_x,y)

#question 3.e
focal = np.array([7500,15,3,127,2,2])
transf_focal = focal*transf
neighbours = nbrs_un.kneighbors(transf_focal,return_distance=False)
class_result = fit_.predict(transf_focal)
print("-----------------------------------------")
print("---------------question 3.e--------------")
print("-----------------------------------------")
print(ortho_mat.columns)
print("Neighbours of the data: ",neighbours)
print("Neighbours input Variables: \n")
for i in neighbours:
    print(ortho_mat.iloc[i,:])
print("Predicted Target Class:",class_result)
#question 3.f
print("-----------------------------------------")
print("--------------question 3.f---------------")
print("-----------------------------------------")
class_prob = fit_.predict_proba(int_var_mat)
print("Predicting Probabilites on Traininig data:",class_prob)
prob = fit_.predict_proba(transf_focal)
print("Predicted Probabilites: ",prob)


#targetClass = [0,1]
#prob2 = fit_.predict_proba(int_var_mat)
#print(prob2)
#target = Fraud_df['FRAUD']
#nMissClass = 0
#for i in range(len(Fraud_df)):
#    j = np.argmax(prob[i][:])
#    predictClass = targetClass[j]
#    if (predictClass != target.iloc[i]):
#        nMissClass += 1
#print(nMissClass)
#rateMissClass = nMissClass /len(Fraud_df)
#print('Misclassification Rate = ', rateMissClass)






















    
    

    


















