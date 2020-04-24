import pandas as pd
import sklearn.neural_network as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics as st

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
    iter = nnObj.n_iter_

    y_t = spiral['SpectralCluster']

    pred = []
    for i in range(len(y_t)):
        if(y_pred[i][0]>=threshold):
            pred.append(0)
        else:
            pred.append(1)

    accuracy = metrics.accuracy_score(y_t,pred)
    missclassification_rate = 1-accuracy
    return (Loss,missclassification_rate,act_func,iter)

print('question 1.b')
result = pd.DataFrame(columns = ['Activation Function','nLayer', 'nHiddenNeuron','Iterations','Loss', 'MissClassification Rate'])
act_function = ['relu','identity','logistic','tanh']

for k in act_function:
    min_loss = float('inf')
    min_class = float('inf')
    for i in np.arange(1,6):
        for j in np.arange(1,11,1):
            Loss, missclass, act_func,iter = Build_NN_Toy(nLayer = i, nHiddenNeuron = j,act_func=k)
            if(Loss<=min_loss and missclass<=min_class):
                min_loss = Loss
                min_class = missclass
                layer = i
                hid = j
                it = iter
                act_func = k
    result = result.append(pd.DataFrame([[act_func,layer,hid,it,min_loss,min_class]],
                                       columns =['Activation Function','nLayer', 'nHiddenNeuron','Iterations','Loss', 'MissClassification Rate']))


pd.set_option('display.max_rows', result.shape[0]+1)
pd.set_option('display.max_columns', result.shape[1]+1)
#print(result2)
result.nLayer = result.nLayer.astype('int64')
result.nHiddenNeuron = result.nHiddenNeuron.astype('int64')
result.Iterations = result.Iterations.astype('int64')

result.reset_index(drop=True,inplace=True)
print(result)

index = result['Loss'].idxmin()
print('question 1.d')
pd.set_option('display.max_rows', 10)
print(result.iloc[index,:])
print('question 1.e')
min_hidden = int(result.iloc[index,2])
min_layers = int(result.iloc[index,1])
nnObj = nn.MLPClassifier(hidden_layer_sizes = (min_hidden,)*min_layers,
                        activation = 'relu', verbose = False,
                        solver = 'lbfgs', learning_rate_init = 0.1,
                        max_iter = 5000, random_state = 20200408)

thisFit = nnObj.fit(spiral[['x','y']], spiral['SpectralCluster'])
y_pred = nnObj.predict_proba(spiral[['x','y']])

y_t = spiral['SpectralCluster']
pred = []
for i in range(len(y_t)):
    if(y_pred[i][0]>=threshold):
        pred.append(0)
    else:
        pred.append(1)
spiral['Predicted'] = pred

colors = ['red', 'blue']
for i in range(2):
    i_data = spiral[spiral['Predicted'] == i]
    plt.scatter(x = i_data['x'],y=i_data['y'],c = colors[i], label = i, s = 25,linestyle='-')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
print('question 1.c')
print("Activation Function of output function= {}".format(nnObj.out_activation_))
print('question 1.f')

pred_1 = [y_pred[i][1] for i in range(len(y_pred)) if y_pred[i][1]>=0.5]
print(f"Count :{len(pred_1):.10f}")
print(f"Mean: {sum(pred_1)/len(pred_1):.10f}")
print(f"Standard Deviation: {st.stdev(pred_1):.10f}")
