import pandas as pd
import sklearn.neural_network as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

spiral = pd.read_csv('SpiralWithCluster.csv')

print(spiral.head())

total = len(spiral)

x = spiral['SpectralCluster'].value_counts()[1]

print('question 1.a')
print('Percent of total in SpiralCluster[1]',100*x/total)
threshold = 0.5
xVar = spiral[['id','x','y']]
y = spiral['SpectralCluster']
#question 1.b
def Build_NN_Toy (nLayer, nHiddenNeuron,act_func):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = act_func, verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20200408)
    # nnObj.out_activation_ = 'identity'
    thisFit = nnObj.fit(spiral[['x','y']], spiral['SpectralCluster'])
    y_pred = nnObj.predict_proba(spiral[['x','y']])

    Loss = nnObj.loss_
    #RSquare = 2

    y_t = spiral['SpectralCluster']

    pred = []

    for i in range(len(y_t)):
        if(y_pred[i][0]>=threshold):
            pred.append(0)
        else:
            pred.append(1)

    accuracy = metrics.accuracy_score(y_t,pred)
    missclassification_rate = 1-accuracy



    return (Loss,missclassification_rate,act_func)

result = pd.DataFrame(columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Activation Function'])
act_function = ['relu','identity','logistic','tanh']



for k in act_function:
    min_loss = float('inf')
    min_class = float('inf')
    for i in np.arange(1,6):
        for j in np.arange(1,11,1):
                Loss, RSquare, act_func = Build_NN_Toy (nLayer = i, nHiddenNeuron = j,act_func=k)
                if(Loss<min_loss and RSquare<min_class):
                    min_loss=Loss
                    min_class = RSquare
                    layer = i
                    hid = j
                    act_func = k
    result = result.append(pd.DataFrame([[layer,hid,min_loss,min_class,act_func]],
                                       columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Activation Function']))



pd.set_option('display.max_rows', result.shape[0]+1)
pd.set_option('display.max_columns', result.shape[1]+1)
#result = result.groupby('Activation Function')['Loss'].min()
print(result)

g = pd.pivot_table(result,index=["Activation Function"],values=['Loss','RSquare'],aggfunc={'Loss':np.min,'RSquare':np.min})
pd.set_option('display.max_rows', g.shape[0]+1)
pd.set_option('display.max_columns', g.shape[1]+1)
#print(g)
