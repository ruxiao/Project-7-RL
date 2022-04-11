#Helper functions
import numpy as np
import math
import pandas as pd

# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

#standardize the data


# # returns the vector containing stock data from a fixed file 
# def getStockData(key):
#     vec = []
#     lines = open("data/" + key + ".csv", "r").read().splitlines()

#     for line in lines[1:]:
#         vec.append(float(line.split(",")[4])) #Only Close column

#     return vec

# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t

def getState(data,t):
    state = pd.DataFrame(data.iloc[t,:]).T
    return state


# def getState(data,t):
#     block=[]
#     for i in data:
#         temp = i[t]
#         block.append(temp)
#     return np.array([block])
        

# def getState(data, t, n):    
#     d = t - n + 1
#     block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
#     #block is which is the for [1283.27002, 1283.27002]
#     res = []
#     for i in range(n - 1):
#         res.append(sigmoid(block[i + 1] - block[i]))
#     return np.array([res])

# Plots the behavior of the output
def plot_behavior(data_input, states_buy, states_sell, profit):
    fig = plt.figure(figsize = (15,5))
    plt.plot(data_input, color='r', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='m', label = 'Buying signal', markevery = states_buy)
    plt.plot(data_input, 'v', markersize=10, color='k', label = 'Selling signal', markevery = states_sell)
    plt.title('Total gains: %f'%(profit))
    plt.legend()
    #plt.savefig('output/'+name+'.png')
    plt.show()