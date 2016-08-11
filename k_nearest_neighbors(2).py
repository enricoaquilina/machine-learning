from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
style.use('fivethirtyeight')

dataset = {'m': [[1,2],[3,4],[4,3]], 'g': [[7,8],[8,10],[9,9]]}
predict = [1,4]

def k_nearest_neighbors(data, predict, k=3):
    distances = []
    for cluster in dataset:
        for feature in dataset[cluster]:
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([euclidean_distance, cluster])

    # for i in sorted(distances):
    #     print(i)
    #     print(i[1])

    votes = [i[1] for i in sorted(distances)[:k]]
    return Counter(votes).most_common(1)


result = k_nearest_neighbors(dataset,predict)
print(result)

[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(predict[0],predict[1],color=result[0][0])

plt.show()