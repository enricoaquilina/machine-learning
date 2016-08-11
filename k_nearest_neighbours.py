import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

example_data = np.array([[1,3,1,3,2,1,3,1,1],[4,2,3,4,2,6,3,5,1]])
example_data = example_data.reshape(2, -1)
predict = clf.predict(example_data)
print(predict)
# print('prediction' + str(clf.predict([[1,3,1,3,2,1,3,1,1,2]])))

accuracy = clf.score(X_test, y_test)
#
print(accuracy)
