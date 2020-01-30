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




file = "C:\\Users\\rkuma\\OneDrive\\Documents\\Courses\\Semester 2\\Machine Learning\\Assignment 1\\NormalSample.csv"
file2 = "C:\\Users\\rkuma\\OneDrive\\Documents\\Courses\\Semester 2\\Machine Learning\\Assignment 1\\Fraud.csv"
#CSV to DataFrame
NormalSample_df = pd.read_csv(file,sep=',')
Fraud_df = pd.read_csv(file2,sep=',')
x=NormalSample_df.iloc[:,2]
print(x)
#Total number of elements in the column 'x'
N = x.count()
lis = x.values.tolist()
lis.sort()
print(lis)
#Median and Median Location
med=st.median(lis)
medloc = lis.index(med)
leftlis = lis[0:medloc]
rightlis = lis[medloc+1:]
quartile1 = st.median(leftlis)
quartile3 = st.median(rightlis)
print(quartile1,quartile3)
#Calculating the IQR Value
IQR = quartile3-quartile1
print(IQR)
bw = 2*IQR*(N**(-1/3))
#Question 1.a
print("---------------------question 1.a-------------")
binwidth = np.round(bw,2)
print("Binwidth: ",binwidth)
#question 1.b
print("-----------question 1.b----------------")
print("Minimum Value: ",min(x))
print("Maximum Value: ",max(x))
#question 1.c
print("-----------------question 1.c--------------")
a = np.floor(min(x))
b = np.ceil(max(x))
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
print("-----------question 1.d------------")
print("-----------density estimates,midpoint values------------")
den_val,mid_val = hist_den_est(0.25)
dm = zip(den_val,mid_val)
print(list(dm))
#number of bins bin size = 0.25
nobin_1d = int((b-a)/0.25)
plt.figure(figsize=(15,15))
plt.hist(x,bins=nobin_1d)
plt.show()
#question 1.e
print("------------question 1.e--------------")
print("-----------density estimates,midpoint values------------")
den_val,mid_val = hist_den_est(0.5)
dm=zip(den_val,mid_val)
print(list(dm))
#number of bins bin size = 0.5
nobin_1e = int((b-a)/0.5)
plt.figure()
plt.hist(x,bins=nobin_1e)
plt.show()
#question 1.f
print("----------question 1.f-----------------")
print("-----------density estimates,midpoint values------------")
den_val,mid_val = hist_den_est(1)
dm=zip(den_val,mid_val)
print(list(dm))
#number of bins bin size = 1
nobin_1f = int((b-a)/1)
plt.figure()
plt.hist(x,bins=nobin_1f)
plt.show()
#question 1.g
print("----------question 1.g---------------")
print("-----------density estimates,midpoint values------------")
den_val,mid_val = hist_den_est(2)
dm=zip(den_val,mid_val)
print(list(dm))
#number of bins bin size = 2
nobin_1g= int((b-a)/2)
plt.figure()
plt.hist(x,bins=nobin_1g)
plt.show()

#Five number summary and IQR Whiskers
def Five_Number_Summary(f):
    Q1,Median,Q3 = np.percentile(f,[25,50,75])
    minimum = min(x)
    maximum = max(x)
    return Median,Q1,Q3,minimum,maximum
def IQR_Whiskers(g):
    Median,Q1,Q3,Minimum,Maximum = Five_Number_Summary(g)
    low_whisker = Q1 - (1.5*IQR)
    up_whisker = Q3 + (1.5*IQR)
    return low_whisker,up_whisker

#question 2.a
print("---------------------question 2.a----------------")
print("Five Number Summary")
print(dict(list(zip(['Median','Q1','Q3','Minimum','Maximum'],Five_Number_Summary(x)))))
print(Five_Number_Summary(x))
print("IQR Whiskers")
print(dict(list(zip(['Lower Whisker','Upper Whisker'],IQR_Whiskers(x)))))

#Grouping Data
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
print("----------------question 2.b----------")
print(grouping(NormalSample_df,NormalSample_df['group']))
#1.5 IQR Whiskers
print("Group 0 Whiskers") 
print(dict(list(zip(['Lower Whisker','Upper Whisker'],IQR_Whiskers(group_0_data)))))
print("group 1 Whiskers")
print(dict(list(zip(['Lower Whisker','Upper Whisker'],IQR_Whiskers(group_1_data)))))

#question 3.c
print("------------question 2.c-----------")
plt.figure(figsize=[15,15])
x_box = NormalSample_df.boxplot(column=['x'])

#question 3.d
print("-----------------question 2.d---------------")
group_data = pd.DataFrame(list(zip(group_0_data,group_1_data)),columns=['Group_0','Group_1'])
plt.figure(figsize=[15,15])
x_group_box = group_data.boxplot(column=['Group_0','Group_1'])
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
print(outliers(x))
print(outliers(group_0_data))
print(outliers(group_1_data))
            
#question 3.a
print("---------------question 3.a-------------")
fraud = Fraud_df.loc[:,'FRAUD']
count = fraud.value_counts()
per_fraud = fraud.value_counts(normalize=True).mul(100).round(4).astype(str)+'%'
percent = pd.DataFrame({'Count':count,'Percentage':per_fraud})
print("Percentage of Fraud investigations "+percent.iloc[1,1])

int_var = Fraud_df.drop(['FRAUD'],axis=1).columns
#question 3.b
print("-----------question 3.b-------------")
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
print("----------------------question 3.c.ii---------------")
xtx = transf_x.transpose().dot(transf_x)
print("An Identity Matrix: \n",xtx)

from scipy import linalg as LA2
#Using SciPy Function
orthx = LA2.orth(int_var_mat)
print("The Orthonormalize of int_var_mat: ",orthx)
check = orthx.transpose().dot(orthx)
print("Identity Matrix: \n",check)
print("---------------------------question 3.c.i------------------")
print("Number of Dimensions: {} ".format(len(ortho_mat.columns)))
print("----------------question 3.c.ii------------------")
print("The Transformed Matrix: \n",orthx)
print("Variables are orthonormal: \n",check)


#question 3.d
kNNSpec = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')
y=np.array(Fraud_df.loc[:,'FRAUD'])
fit_=kNNSpec.fit(int_var_mat,y)
tar_var_pred = fit_.predict(int_var_mat)
print("------------question 3.d.i------------") 
print("Score: ",fit_.score(int_var_mat,y))
nbrs_un = kNNSpec.fit(transf_x)

#question 3.e
focal = np.array([7500,15,3,127,2,2])
transf_focal = focal*transf
neighbours = nbrs_un.kneighbors(transf_focal,return_distance=False)
print("Neighbours of the data: \n",neighbours)

#question 3.f
neigh = kNN(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs_sp = neigh.fit(int_var_mat, y)
class_prob = nbrs_sp.predict_proba(int_var_mat)
print("Predicting Probabilites on Traininig data:",class_prob)
class_result = nbrs_sp.predict(transf_focal)
prob = nbrs_sp.predict_proba(transf_focal)
print("Predicted Target Class:",class_result)
print("Predicted Probabilites: ",prob)
accuracy = nbrs_sp.score(int_var_mat, y)
print('The score for our KNN neighbours model is: ',accuracy)






















    
    

    


















