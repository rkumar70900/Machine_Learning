# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:52:30 2020

@author: rkuma
"""


import pandas as pd
import numpy as np
import sympy
import scipy
import statsmodels.api as stats
import naive_bayes



def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pd.DataFrame(np.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)

#Reading the input
file = pd.read_csv('Purchase_Likelihood.csv',usecols=['group_size','homeowner','married_couple','insurance'])
#dropping the missing rows
file = file.dropna()

#target variable
y = file['insurance'].astype('category')
#predictor variables
xG = pd.get_dummies(file[['group_size']].astype('category'))
xH = pd.get_dummies(file[['homeowner']].astype('category'))
xM = pd.get_dummies(file[['married_couple']].astype('category'))

#Intercept model
print('-'*50)
print("Intercept model")
print('-'*50)
designX = pd.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')
print('Number of Parameters = ',DF0)
print('Log Likelihood = ',LLK0)

#Intercept + group_size
print('-'*50)
print('Intercept + group_size')
print('-'*50)
designX = stats.add_constant(xG, prepend=True)
LLK_1G, DF_1G, fullParams_1G = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G - LLK0)
testDF = DF_1G - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_1G)
print('Log Likelihood = ',LLK_1G)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#Intercept + group_size + homeowner
print('-'*50)
print('Intercept + group_size + homeowner')
print('-'*50)
designX = xG
designX = designX.join(xH)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H, DF_1G_1H, fullParams_1G_1H = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H - LLK_1G)
testDF = DF_1G_1H - DF_1G
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_1G_1H)
print('Log Likelihood = ',LLK_1G_1H)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#Intercept +group_size + homeowner + married_couple
print('-'*50)
print('Intercept +group_size + homeowner + married_couple')
print('-'*50)
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M, DF_1G_1H_1M, fullParams_1G_1H_1M = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H_1M - LLK_1G_1H )
testDF = DF_1G_1H_1M - DF_1G_1H
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_1G_1H_1M)
print('Log Likelihood = ',LLK_1G_1H_1M)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#Intercept + group_size + homeowner + married_couple + group_size*homeowner'
print('-'*50)
print('Intercept + group_size + homeowner + married_couple + group_size*homeowner')
print('-'*50)
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)
xGH = create_interaction(xG, xH)
designX = designX.join(xGH)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M_GH, DF_1G_1H_1M_GH, fullParams_1G_1H_1M_GH = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H_1M_GH - LLK_1G_1H_1M )
testDF = DF_1G_1H_1M_GH - DF_1G_1H_1M
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_1G_1H_1M_GH)
print('Log Likelihood = ',LLK_1G_1H_1M_GH)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#Intercept + group_size + homeowner + married_couple + group_size*homeowner + group_size * married_couple
print('-'*50)
print('Intercept + group_size + homeowner + married_couple + group_size*homeowner + group_size * married_couple')
print('-'*50)
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)
xGH = create_interaction(xG, xH)
designX = designX.join(xGH)
designX = stats.add_constant(designX, prepend=True)
xGM = create_interaction(xG, xM)
designX = designX.join(xGM)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M_GH_GM, DF_1G_1H_1M_GH_GM, fullParams_1G_1H_1M_GH_GM = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H_1M_GH_GM - LLK_1G_1H_1M_GH )
testDF = DF_1G_1H_1M_GH_GM - DF_1G_1H_1M_GH
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_1G_1H_1M_GH_GM)
print('Log Likelihood = ',LLK_1G_1H_1M_GH_GM)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#Intercept + group_size + homeowner + married_couple + group_size*homeowner + group_size * married_couple + homeowner * married_couple
print('-'*50)
print('Intercept + group_size + homeowner + married_couple + group_size*homeowner + group_size * married_couple + homeowner * married_couple' )
print('-'*50)
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)
xGH = create_interaction(xG, xH)
designX = designX.join(xGH)
designX = stats.add_constant(designX, prepend=True)
xGM = create_interaction(xG, xM)
designX = designX.join(xGM)
designX = stats.add_constant(designX, prepend=True)
xHM = create_interaction(xH, xM)
designX = designX.join(xHM)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M_GH_GM_HM, DF_1G_1H_1M_GH_GM_HM, fullParams_1G_1H_1M_GH_GM_HM= build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1G_1H_1M_GH_GM_HM - LLK_1G_1H_1M_GH_GM )
testDF = DF_1G_1H_1M_GH_GM_HM - DF_1G_1H_1M_GH_GM
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_1G_1H_1M_GH_GM_HM)
print('Log Likelihood = ',LLK_1G_1H_1M_GH_GM_HM)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#question 1.b
print('-'*50)
print("question 1.b")
print('-'*50)
print('Degreee of Freedom = ', testDF)

#training the model
final_model = stats.MNLogit(y,designX)
#Fitting the model
final_model_fit = final_model.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)

#All possibilities
GS_P = [1, 2, 3, 4]
HO_P = [0, 1]
MC_P = [0, 1]

all_pos = []

for i in GS_P:
    for j in HO_P:
        for k in MC_P:
            pos = [i,j,k]
            all_pos = all_pos + [pos]


#all possibilites in a dataframe
inp = pd.DataFrame(all_pos,columns=['group_size','homeowner','married_couple']) 
#Target test 
xG = pd.get_dummies(inp[['group_size']].astype('category'))
xH = pd.get_dummies(inp[['homeowner']].astype('category'))
xM = pd.get_dummies(inp[['married_couple']].astype('category'))
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)
xGH = create_interaction(xG, xH)
designX = designX.join(xGH)
designX = stats.add_constant(designX,prepend = True)
xGM = create_interaction(xG, xM)
designX = designX.join(xGM)
designX = stats.add_constant(designX,prepend = True)
xHM = create_interaction(xH, xM)
designX = designX.join(xHM)
designX = stats.add_constant(designX, prepend=True)
        
#predicting
pred = final_model_fit.predict(exog=designX)

pred['0'] = pred[0]
pred['1'] = pred[1]
pred['2'] = pred[2]

print("-"*50)
print("question 2.a")
print("-"*50)
out = pd.concat([inp,pred[['0','1','2']]],axis=1)
print(out)

print("-"*50)
print("question 2.b")
print("-"*50)

max_odd = []
for i in range(len(out)):
    max_odd.append(out.iloc[i,4]/out.iloc[i,3])

t = max_odd.index(max(max_odd))

print('Value combination of the features with maximum odds value')
print('group_size: ',out.iloc[t,0])
print('homeowner: ',out.iloc[t,1])
print('married_couple: ',out.iloc[t,2])

#queston 2.c
print("-"*50)
print("question 2.c")
print("-"*50)

g3_i2 = file[file['group_size']==3].groupby('insurance').size()[2]/len(file[file['group_size']==3])
g3_i0 = file[file['group_size']==3].groupby('insurance').size()[0]/len(file[file['group_size']==3])
num = g3_i2/g3_i0
g1_i2 = file[file['group_size']==1].groupby('insurance').size()[2]/len(file[file['group_size']==1])
g1_i0 = file[file['group_size']==1].groupby('insurance').size()[0]/len(file[file['group_size']==1])
den = g1_i2/g1_i0

odds_ratio_c = num/den

print("Odds ratio = ",odds_ratio_c)
#question 2.d
print("-"*50)
print("question 2.d")
print("-"*50)

h1_i0 = file[file['homeowner']==1].groupby('insurance').size()[0]/len(file[file['homeowner']==1])
h1_i1 = file[file['homeowner']==1].groupby('insurance').size()[1]/len(file[file['homeowner']==1])
num = h1_i0/h1_i1
h0_10 = file[file['homeowner']==0].groupby('insurance').size()[0]/len(file[file['homeowner']==0])
h0_i1 = file[file['homeowner']==0].groupby('insurance').size()[1]/len(file[file['homeowner']==0])
den = h0_10/h0_i1

odds_ratio_d = num/den
print("Odds ratio = ",odds_ratio_d)

#question 3
print("-"*50)
print('question 3.a')
print("-"*50)

x = file.groupby("insurance").size()
freq_table = pd.DataFrame(columns=['frequency_count','probability'])
freq_table['frequency_count'] = x
freq_table['probability'] = freq_table['frequency_count']/len(file)
print(freq_table)

def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
             ):

   countTable = pd.crosstab(index = columnVar, columns = rowVar, margins = False, dropna = True)
   print("Cross Tabulation Table: \n", countTable)
   print( )
   
   cTotal = countTable.sum(axis = 1)
   rTotal = countTable.sum(axis = 0)
   nTotal = np.sum(rTotal)
   expCount = np.outer(cTotal, (rTotal / nTotal))
   chiSqStat = ((countTable - expCount)**2 / expCount).to_numpy().sum()
   cramerV = chiSqStat / nTotal
   if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
   else:
    cramerV = cramerV / (cTotal.size - 1.0)
   cramerV = np.sqrt(cramerV)
    
   print("CramerV: \n",cramerV)

   return
print("-"*50)
print('question 3.b')
print("-"*50)

RowWithColumn(rowVar = file['insurance'], columnVar = file['group_size'])
print("-"*50)
print('question 3.c')
print("-"*50)

RowWithColumn(rowVar = file['insurance'], columnVar = file['homeowner'])
print("-"*50)
print('question 3.d')
print("-"*50)

RowWithColumn(rowVar = file['insurance'], columnVar = file['married_couple'])

print("-"*50)
print("question 3.e")
print("-"*50)
print("Based on the cramer's value")

print("-"*50)
print("question 3.f")
print("-"*50)


xTrain = file[['group_size','homeowner','married_couple']].astype('category')
yTrain = file['insurance'].astype('category')
objNB = naive_bayes.MultinomialNB(alpha = 0)
thisModel = objNB.fit(xTrain, yTrain)
pred_naive = thisModel.predict_proba(inp)
naive_prob = pd.DataFrame(pred_naive,columns=['ins_0','ins_1','ins_2'])
out = pd.concat([inp,naive_prob],axis=1)
print(out)

print("-"*50)
print("question 3.g")
print("-"*50)

max_odd = []
for i in range(len(out)):
    max_odd.append(out.iloc[i,4]/out.iloc[i,3])

t = max_odd.index(max(max_odd))
print('Value combination of the features with maximum odds value')
print('group_size: ',out.iloc[t,0])
print('homeowner: ',out.iloc[t,1])
print('married_couple: ',out.iloc[t,2])


