# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:21:46 2020

@author: rkuma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes as km
import seaborn as sns
import sklearn.cluster as cl
from sklearn.neighbors import NearestNeighbors as NN
import math
import numpy.linalg as linalg
import sklearn.neighbors as neighbors
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

file = "Groceries.csv"

df = pd.read_csv(file)

item_group = df.groupby(['Customer'])['Item'].apply(list).values.tolist()
te = TransactionEncoder()
te_ary = te.fit(item_group).transform(item_group)
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)
#frequent_itemsets = apriori(ItemIndicator, min_support = 0.04, max_len = 2, use_colnames = True)
#print(frequent_itemsets)
#assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.2 )
#print(assoc_rules)

#question 1.a
print("----------------question 1.a------------")
df2 = pd.DataFrame(item_group)  
data = []
for i in range(len(item_group)):
    data.append(len(item_group[i]))
customer = [i for i in range(1,len(data)+1)]
#DataFrame containing unique number of items in each customer's Basket
item_df = pd.DataFrame(data=data,columns=['No of items'])
item_df['Customer']=customer
#Histogram for Number of Unique items
plt.figure(figsize=(10,10))
sns.distplot(item_df['No of items'],kde=False)
plt.show()
#25,50,75 Percentiles of histogram
LQ,Median,UQ = np.percentile(item_df['No of items'],[25,50,75])
print("25th Percentile of Histogram:",LQ)
print("50th Percentile of Histogram:",Median)
print("75th Percentile of Histogram:",UQ)

#question 1.b
print("----------------question 1.b--------------")
data2 = []
for j in range(len(item_group)):
    data2.append(item_group[j])
df3 = pd.DataFrame(data=customer,columns=['Customer'])
minSup = 75/len(df3)
#Apriori Algorithm
frequent_itemsets = apriori(ItemIndicator, min_support = minSup, max_len = Median, use_colnames = True)
#itemsets with atleast 75 customers
total_itemsets = len(frequent_itemsets)
print("Total number of itemsets with atleast 75 customers",total_itemsets)
#largest k-value
count_k = []
k_itemset = [count_k.append(len(frequent_itemsets.iloc[i,1])) for i in range(len(frequent_itemsets))]
max_k_itemset = max(count_k)
print("The largest k-value among the itemsets",max_k_itemset)
#question 1.c
print("--------------question 1.c--------------")
#Association Rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01 )
#droping na values to make sure every rule has antecendent and consequents
assoc_rules.dropna()
total_assoc_rules = len(assoc_rules)
print("Total number of Association Rules with > 1% Confidence ",total_assoc_rules)
assoc_rules.columns
#question 1.d
print("--------------question 1.d--------------")
print("Support Vs Confidence")
plt.figure(figsize=(15,15))
sns.scatterplot(data=assoc_rules,x="confidence",y="support",size="lift")
plt.show()
print("--------------question 1.e--------------")
assoc_rules_60 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6 )
print("Association rules with confidence >= 60% \n",assoc_rules_60.loc[:,['antecedents','consequents','confidence','support','lift']])

file2 = "cars.csv"

df4 = pd.read_csv(file2)
cars_df = df4.loc[:,['Type','Origin','DriveTrain','Cylinders']]
#a

print("--------------question 2.a--------------")
labels_type = list(cars_df['Type'].value_counts().index)
List_Of_Categories_In_Type = list(cars_df['Type'].value_counts())
print("Frequencies of Type")
type_freq = pd.DataFrame(list(zip(labels_type,List_Of_Categories_In_Type)),columns=['Type','Count'])
print(type_freq)
#b
print("--------------question 2.b--------------")
labels_drivetrain = list(cars_df['DriveTrain'].value_counts().index)
print("Frequencies of DriveTrain")
List_Of_Categories_In_DriveTrain = list(cars_df['DriveTrain'].value_counts())
drivetrain_freq = pd.DataFrame(list(zip(labels_drivetrain,List_Of_Categories_In_DriveTrain)),columns=['DriveTrain','Count'])
print(drivetrain_freq)
#c
print("--------------question 2.c--------------")
labels_origin = list(cars_df['Origin'].value_counts().index)
List_Of_Categories_In_Origin = list(cars_df['Origin'].value_counts())
origin_df = pd.DataFrame(List_Of_Categories_In_Origin)
origin = origin_df.T
o = origin.values.tolist()
ori_asia = cars_df.Origin.value_counts()['Asia']
ori_eur = cars_df.Origin.value_counts()['Europe']
dis_origin = (1/ori_asia)+(1/ori_eur)
print("Distance between Asia and Europe")
print(dis_origin)
#question 2.d
print("--------------question 2.d--------------")
List_Of_Categories_In_Cylinders = list(cars_df['Cylinders'].value_counts())
labels_Cylinders = list(cars_df['Cylinders'].value_counts().index)
cylinders_df = pd.DataFrame(List_Of_Categories_In_Cylinders)
cylinder = cylinders_df.T
c = cylinder.values.tolist()
cyc_5 = cars_df.Cylinders.value_counts()[5.0]
cyc_nan = cars_df.Cylinders.value_counts(dropna=False)[np.NaN]
dis_cylinder_df = pd.DataFrame(c,columns=labels_Cylinders)
dis_cyc = ((1/cyc_5)+(1/cyc_nan))
print("distance between 5 and missing")
print(dis_cyc)
#e
print("--------------question 2.e--------------")
cars=cars_df.fillna(0)
cars_km = km(n_clusters=3,max_iter=25,init='Cao')
clusters = cars_km.fit_predict(cars)
#number of elements in each cluster
unique, counts = np.unique(cars_km.labels_, return_counts=True)
num_ele=dict(zip(unique, counts))
print("cluster and Number of Elements")
print(num_ele)
#centeroids
print("centeroids")
print(cars_km.cluster_centroids_)
#f
print("--------------question 2.f--------------")
ff = pd.DataFrame(list(zip(clusters,cars_df['Origin'])),columns=['Cluster','Origin'])
g = ff.groupby(['Cluster','Origin']).size()
print("Frequency distribution table for Origin")
print(g)

#question 3
file3 = "FourCircle.csv"
#a
print("--------------question 3.a--------------")
four_circle_df = pd.read_csv(file3)
plt.figure(figsize=[15,15])
four_circle_sc = sns.scatterplot(x='x',y='y',data=four_circle_df)
plt.show()
#by visual inspection there are 4 clusters
#b
print("--------------question 3.b--------------")
x_y = four_circle_df[['x','y']]
fourcircle_km = cl.KMeans(n_clusters=4, random_state=60616)
fit_circle = fourcircle_km.fit(x_y)
plt.figure(figsize=[15,15])
sc_new = sns.scatterplot(x='x',y='y',data=four_circle_df,hue=fit_circle.labels_)
plt.show()
#c
#6 nearest observations
print("--------------question 3.c--------------")
kNNSpec = NN(n_neighbors = 6, algorithm = 'brute', metric = 'euclidean')
ndrs = kNNSpec.fit(x_y)
distances,indices = ndrs.kneighbors(x_y)
nObs = four_circle_df.shape[0]
#distances among observations
distObject = neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(x_y)
# Create the Adjacency matrix
print("--------------question 3.d--------------")
Adjacency = np.zeros((nObs, nObs))
for i in range(nObs):
    for j in indices[i]:
        Adjacency[i,j] = math.exp(- (distances[i][j])**2 )
# Make the Adjacency matrix symmetric
Adjacency = 0.5 * (Adjacency + Adjacency.transpose())
print("Adjacency_Matrix: \n",Adjacency)
# Create the Degree matrix
Degree = np.zeros((nObs, nObs))
print("Degree_Matrix: \n",Degree)
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
# Create the Laplacian matrix        
Lmatrix = Degree - Adjacency
print("Laplace Matrix: \n",Lmatrix,"\n")
# Obtain the eigenvalues and the eigenvectors of the Laplacian matrix
evals, evecs = linalg.eigh(Lmatrix)
for j in range(10):
    print('Eigenvalue: ', evals[j])
Z = evecs[:,[1,2,3,4]]
#for clusters
sequence = np.arange(1,6,1) 
plt.figure(figsize=[10,10])
plt.plot(sequence, evals[0:5,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.xticks(sequence)
plt.grid(True)
plt.show()
print("--------------question 3.e--------------")
kmeans_spectral = cl.KMeans(n_clusters = 4, random_state = 0).fit(Z)
four_circle_df['SpectralCluster'] = kmeans_spectral.labels_
#plt.scatter(four_circle_df['x'], four_circle_df['y'], c = four_circle_df['SpectralCluster'])
sns.scatterplot(x='x',y='y',data=four_circle_df,hue=kmeans_spectral.labels_)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
















