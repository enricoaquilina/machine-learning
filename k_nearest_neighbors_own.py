from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import pandas as pd
import random
from collections import Counter
style.use('fivethirtyeight')

# dataset = {'m': [[1,2],[3,4],[4,3]], 'g': [[7,8],[8,10],[9,9]]}
# predict = [1,4]

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
# random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
# get 80% for training
train_data = full_data[:-int(test_size*len(full_data))]
# get 20% for testing
test_data = full_data[int(test_size*len(full_data)):]

# X = np.array(df.drop(['class'], 1))
# y = np.array(df['class'])

def k_nearest_neighbors(data, predict, k=3):
    distances = []
    for cluster in data:
        for feature in data[cluster]:
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([euclidean_distance, cluster])

    # print(sorted(distances)[:k])
    votes = [i[1] for i in sorted(distances)[:k]]
    # print(votes)
    result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][0] / k

    return result, confidence

# print(k_nearest_neighbors(dataset,predict))
# [[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(predict[0],predict[1],color=result[0][0])

[train_set[i[-1]].append(i[:-1]) for i in train_data]
[test_set[i[-1]].append(i[:-1]) for i in test_data]

counter = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, con = k_nearest_neighbors(train_set, data, k=5)
        # print(vote)
        if group == vote:
            counter+=1
        else:
            print(con)
        total+=1

print('Accuracy:', (counter/total))

plt.show()