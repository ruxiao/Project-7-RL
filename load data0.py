# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
import math
from numpy.random import choice
import random

#Import Model Packages for reinforcement learning
from keras import layers, models, optimizers
from keras import backend as K
from collections import namedtuple, deque


#The data already obtained from yahoo finance is imported.
dataset = read_csv(r"C:\Users\Administrator\py\lehigh project\修改实验\sh600900.csv",index_col=0)
#Diable the warnings
import warnings
warnings.filterwarnings('ignore')
type(dataset)


# shape
dataset.shape
# peek at data
set_option('display.width', 100)
dataset.head(5)
# describe data
set_option('precision', 3)
dataset.describe()
#Checking for any null values and removing the null values'''
dataset['close'].plot()
print('Null Values =',dataset.isnull().values.any())
# Fill the missing values with the last value available in the dataset. 
dataset=dataset.fillna(method='ffill')
dataset.head(2)
data = dataset
columns = data.columns
stdN = [[],[]]
for i in range(len(columns)):
        tempM = data[columns[i]].mean()
        stdN[0].append(tempM)
        tempS = data[columns[i]].std()
        stdN[1].append(tempS)
        data[columns[i]] -= data[columns[i]].mean()
        data[columns[i]] /= data[columns[i]].std()
y = pd.DataFrame(data['close'])
x = data.drop(columns = 'close')

#掐头去尾
x = x.iloc[:-1,:]
y = y.iloc[1:,:]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 7,shuffle=False)





# Y=list(dataset['close'])
# dataset = dataset.drop(columns='close')



# X=list(dataset["close"])
# X=[float(x) for x in X]
# factorname = list(dataset)
# X=[]
# for i in factorname:
#     X.append(list(dataset[i]))
# validation_size = 0.2
# #In case the data is not dependent on the time series, then train and test split should be done based on sequential sample
# #This can be done by selecting an arbitrary split point in the ordered list of observations and creating two new datasets.
# train_size = int(len(X[0]) * (1-validation_size))
# #X_train, X_test = X[0:train_size], X[train_size:len(X)]
# X_train = []
# X_test = []
# for i in X:
#     X_train.append(i[0:train_size])
#     X_test.append(i[train_size:len(X[0])])

# Y_train=Y[0:train_size]
# Y_test=Y[train_size:len(Y)]