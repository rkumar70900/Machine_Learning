import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import sklearn.svm as svm
import seaborn as sns

spiral = pd.read_csv('C:\\Users\\rkuma\\Documents\\Semester 2\\Machine Learning\\Assignment 5\\SpiralWithCluster.csv',usecols=['x','y','SpectralCluster'])

xTrain = spiral[['x','y']]
yTrain = spiral['SpectralCluster']

svm_model = svm.SVC(kernel='linear',decision_function_shape='ovr',random_state=20200408,max_iter=-1)
thisFit = svm_model.fit(xTrain,yTrain)
pred_class = thisFit.predict(xTrain)
spiral['prediction'] = pred_class
#question 2.a
print('question 2.a')
print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)
print(f'The Hyperplane is {thisFit.coef_[0][0]:.7f}x1 + {thisFit.coef_[0][1]:.7f}x2 = {thisFit.intercept_[0]:.7f}')
#question 2.b
print('question 2.b')
accuracy = metrics.accuracy_score(yTrain, pred_class)
missclassification_rate = 1-accuracy
print("Missclassification Rate = ",missclassification_rate)
#question 2.c
print('question 2.c')
w = thisFit.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - ((thisFit.intercept_[0]) / w[1])
colors = ['red', 'blue']
for i in range(2):
    i_data = spiral[spiral['prediction'] == i]
    plt.scatter(x = i_data['x'],y=i_data['y'],c = colors[i], label = i, s = 25,linestyle='-')
#sns.scatterplot(x='x',y='y',hue='prediction',data=spiral)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spectral Cluster with hyper plane seperating the clusters')
plt.plot(xx, yy,'k:')
plt.grid(True)
plt.legend(title='Group')
plt.show()
#question 3.d
print('question 3.d')
spiral['radius'] = np.sqrt(spiral['x']**2 + spiral['y']**2)
spiral['theta'] = np.arctan2(spiral['y'], spiral['x'])
def customArcTan (z):
    theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (theta)

spiral['theta'] = spiral['theta'].apply(customArcTan)

xTrain = spiral[['radius','theta']]
yTrain = spiral['SpectralCluster']
colors = ['red', 'blue']
for i in range(2):
    i_data = spiral[spiral['prediction'] == i]
    plt.scatter(x = i_data['radius'],y=i_data['theta'],c = colors[i], label = i, s = 25,linestyle='-')
plt.xlabel('radius')
plt.ylabel('theta')
plt.title('Spectral cluster with radius and theta coordinates')
plt.grid(True)
plt.legend(title='Group')
plt.show()

print('question 3.e')
group = np.zeros(spiral.shape[0])

# create four group by using the location of the coordinates
for i,r in spiral.iterrows():
    if r['radius'] < 1.5 and r['theta'] > 6:
        group[i] = 0
    elif r['radius'] < 2.5 and r['theta'] > 3:
        group[i] = 1
    elif 2.5 < r['radius'] < 3 and r['theta'] > 5.5:
        group[i] = 1
    elif r['radius'] < 2.5 and r['theta'] < 3:
        group[i] = 2
    elif 3 < r['radius'] < 4 and 3.5 < r['theta'] < 6.5:
        group[i] = 2
    elif 2.5 < r['radius'] < 3 and 2 < r['theta'] < 4:
        group[i] = 2
    elif 2.5 < r['radius'] < 3.5 and r['theta'] < 2.25:
        group[i] = 3
    elif 3.55 < r['radius'] and r['theta'] < 3.25:
        group[i] = 3

spiral['group']=group
colors = ['red', 'blue','green','black']
for i in range(4):
    i_data = spiral[spiral['group'] == i]
    plt.scatter(x = i_data['radius'],y=i_data['theta'],c = colors[i], label = i, s = 25,linestyle='-')
#sns.scatterplot(x='radius',y='theta',hue='group',data=spiral)
plt.xlabel('radius')
plt.ylabel('theta')
plt.title('Grouping according to radius and theta')
plt.grid(True)
plt.legend(title='Group')
plt.show()

#question 3.f
print('question 3.f')
def threehyper(i,j):
    spiral_i = spiral[spiral['group']==i]
    spiral_j = spiral[spiral['group']==j]
    spiral_ij = spiral_i.append(spiral_j)
    spiral_ij.reset_index(drop=True,inplace=True)
    x_Train = spiral_ij[['radius','theta']]
    y_Train = spiral_ij['group']
    svm_model = svm.SVC(kernel='linear',decision_function_shape='ovr',random_state=20200408,max_iter=-1)
    thisFit = svm_model.fit(x_Train,y_Train)
    w = thisFit.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(1,4)
    yy = a * xx - ((thisFit.intercept_[0]) / w[1])
    plt.plot(xx, yy,'k:')
    plt.grid(True)
    Intercept = thisFit.intercept_
    Coefficient = thisFit.coef_
    print(f'The Hyperplane of SVM {i}: Group{i} vs Group{j} is {Coefficient[0][0]:.7f}x1 + {Coefficient[0][1]:.7f}x2 = {Intercept[0]:.7f}')
    return Intercept,Coefficient,xx,yy

i_01,c_01,x_01,y_01 = threehyper(0,1)
i_12,c_12,x_12,y_12 = threehyper(1,2)
i_23,c_23,x_23,y_23 = threehyper(2,3)

print("SVM 0: Group 0 versus Group 1 \nIntercept {}\nCoefficients {}".format(i_01,c_01))
print("SVM 1: Group 1 versus Group 2 \nIntercept {}\nCoefficients {}".format(i_12,c_12))
print("SVM 2: Group 2 versus Group 3 \nIntercept {}\nCoefficients {}".format(i_23,c_23))

print('question 3.g')
colors = ['red', 'blue','green','black']
for i in range(4):
    i_data = spiral[spiral['group'] == i]
    plt.scatter(x = i_data['radius'],y=i_data['theta'],c = colors[i], label = i, s = 25,linestyle='-')
plt.xlabel('radius')
plt.ylabel('theta')
plt.title('Hyperplanes seperating the groups')
plt.grid(True)
plt.legend(title='Group')
plt.show()


print('question 3.h')
h0_xx = x_01 * np.cos(y_01)
h0_yy = x_01 * np.sin(y_01)
h1_xx = x_12 * np.cos(y_12)
h1_yy = x_12 * np.sin(y_12)
h2_xx = x_23 * np.cos(y_23)
h2_yy = x_23 * np.sin(y_23)
plt.plot(h0_xx, h0_yy, color = 'green', linestyle = ':')
plt.plot(h1_xx, h1_yy, color = 'black', linestyle = ':')
plt.plot(h2_xx, h2_yy, color = 'black', linestyle = ':')
colors = ['red', 'blue']
for i in range(2):
    i_data = spiral[spiral['SpectralCluster'] == i]
    plt.scatter(x = i_data['x'],y=i_data['y'],c = colors[i], label = i, s = 25,linestyle='-')
#sns.scatterplot(x='x',y='y',hue='SpectralCluster',data=spiral)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Support Vector machine on three segments')
plt.grid(True)
plt.legend(title='Group')
plt.show()
