#K-NN

import random
import numpy as np
import math
import heapq
import pandas as pd

k = 3

df=pd.read_csv('dist_d',sep=',', header=None)

def getKNeighbours(index):
    global df
    global k
    dist = df[index]
    #print([dist[1],1])
    h = []
    for i in range(len(dist)):
        heapq.heappush(h,[dist[i],i])
    n = []
    for i in range(k+1):
        n.append(heapq.heappop(h)[1])
    return n[1:]


df1 = pd.read_csv('nursery.data', sep=',', header=None)


classes=df1.iloc[:,-1]
def classify(index):
	neighbours = getKNeighbours(index)
	
	val = dict()
	
	for n in neighbours:
		c = classes[n]
		
		if c in val:
			val[c]+=1
		else:
			val[c]=1
	
	maxIn = -1
	maxC = -1
	for k in val:
		if val[k] > maxC:
			maxC = val[k]
			maxIn = k

	return maxIn


def accuracy(test_pred,test_class):
	test_class=test_class.values.tolist()

	acc = 0
	for i in range(len(test_pred)):
		if (test_pred[i]==test_class[i]):
			acc+=1
	# print(acc)
	# print(len(test_pred))
	return acc/len(test_pred)


train_len = int(0.7*len(df))
train = df[:train_len]
test = df[train_len:]



pred = []
for i in range(train_len,train_len+len(test)):
	pred.append(classify(i))


acc = accuracy(pred,classes.iloc[train_len:train_len+len(test)])
print(acc*100)