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


file = pd.read_csv('Purchase_Likelihood.csv',usecols=['group_size','homeowner','married_couple','insurance'])

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

#Intercept + group_size
print('-'*50)
print('Intercept + group_size')
print('-'*50)
designX = stats.add_constant(xG, prepend=True)
LLK_1G, DF_1G, fullParams_1G = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G - LLK0)
testDF = DF_1G - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)

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
#question 1.b
print('-'*50)
print("question 1.b")
print('-'*50)
print('Degreee of Freedom = ', testDF)

#question 1.c
print('-'*50)
print('question 1.c and question 1.d')
print('-'*50)
print('-'*50)
print("Intercept model")
print('-'*50)
designX = pd.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')
print('Log Likelihood = ',LLK0)
print('Degree of Freedom = ',DF0)

print('-'*50)
print('Intercept + group_size')
print('-'*50)
designX = stats.add_constant(xG, prepend=True)
LLK_G, DF_G, fullParams_G = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_G - LLK0)
testDF = DF_G - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_G)
print('Log Likelihood = ',LLK_G)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#intercept + homeowner
print('-'*50)
print('Intercept + homeowner')
print('-'*50)
designX = stats.add_constant(xH, prepend=True)
LLK_H, DF_H, fullParams_H = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_H - LLK0)
testDF = DF_H - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_H)
print('Log Likelihood = ',LLK_H)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#intercept + marriedcouple
print('-'*50)
print('Intercept + marriedcouple')
print('-'*50)
designX = stats.add_constant(xM, prepend=True)
LLK_M, DF_M, fullParams_M = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_M - LLK0)
testDF = DF_M - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_M)
print('Log Likelihood = ',LLK_M)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#intercept + groupsize + homeowner
print('-'*50)
print('Intercept + groupsize + homeowner')
print('-'*50)
designX = xG
designX = designX.join(xH)
designX = stats.add_constant(designX, prepend=True)
LLK_GH, DF_GH, fullParams_GH = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_GH - LLK_G)
testDF = DF_GH - DF_G
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_GH)
print('Log Likelihood = ',LLK_GH)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#intercept + groupsize + marriedcouple
print('-'*50)
print('Intercept + groupsize + marriedcouple')
print('-'*50)
designX = xG
designX = designX.join(xM)
designX = stats.add_constant(designX, prepend=True)
LLK_GM, DF_GM, fullParams_GM = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_GM - LLK_G)
testDF = DF_GM - DF_G
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_GM)
print('Log Likelihood = ',LLK_GM)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#intercept + homeowner + marriedcouple
print('-'*50)
print('Intercept + homeowner + marriedcouple')
print('-'*50)
designX = xH
designX = designX.join(xM)
designX = stats.add_constant(designX, prepend=True)
LLK_HM, DF_HM, fullParams_HM = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_HM - LLK_H)
testDF = DF_HM - DF_H
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Number of Parameters = ',DF_HM)
print('Log Likelihood = ',LLK_HM)
print('Deviance = ', testDev)
print('Degreee of Freedom = ', testDF)
print('Significance = ', testPValue)
print("Feature Importance Index = ",round(-np.log10(testPValue)),4)

#question 2.a
print('The Model')
print('-'*50)
print('Intercept + group_size + homeowner + married_couple + group_size*homeowner + group_size * married_couple + homeowner * married_couple' )
print('-'*50)
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
y = file['insurance'].astype('category')

#Training the model
final_model = stats.MNLogit(y,designX)
final_model_fit = final_model.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)

#Getting all the possibilities
GS_P = [1, 2, 3, 4]
HO_P = [0, 1]
MC_P = [0, 1]

all_pos = []

for i in GS_P:
    for j in HO_P:
        for k in MC_P:
            pos = [i,j,k]
            all_pos = all_pos + [pos]

#All possibilities as DataFrame
inp = pd.DataFrame(all_pos,columns=['group_size','homeowner','married_couple']) 
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
        
#predicting for the possibilities of group_size, homeowner, married_couple
pred = final_model_fit.predict(exog=designX)
pred['insurance_0'] = pred[0]
pred['insurance_1'] = pred[1]
pred['insurance_2'] = pred[2]

out = pd.concat([inp,pred[['insurance_0','insurance_1','insurance_2']]],axis=1)

print('-'*50)
print('question 2.a')
print('-'*50)
print(out)

    



