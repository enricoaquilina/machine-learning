import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

class SVM:
    def __init__(self, visualisation=True):
        self.visualisation = visualisation
        self.colors = {1:'r',-1:'b'}
        if self.visualisation:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, features):
        self.features = features
        opt_dict = {}

        transforms = [[1,1],[-1,1],[1,-1],[-1,-1]]

        all_data = []
        for y in self.features:
            for featureset in self.features[y]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [
            self.max_feature_value * 0.1,
            self.max_feature_value * 0.01,
            self.max_feature_value * 0.001,
        ]
        # this is expensive..
        b_range_multiple = 5
        b_multiple = 5
        # first element in vector w
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimised = False
            while not optimised:
                for b in np.arange(-1*self.max_feature_value*b_range_multiple,
                                      self.max_feature_value * b_range_multiple,
                                      step*b_multiple
                                   ):
                    for transform in transforms:
                        w_t = w * transform
                        found_option = True
                        # yi(xi.w+b) >= 1
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] < 0:
                    optimised = True
                    print('Optimised a step')
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice[0][0]+step*2

        return self.max_feature_value, self.min_feature_value

    def predict(self, features):
        # sign(x.w + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        return classification

data_dict = {-1: np.array([[1,7],[2,8],[3,8]]),
              1: np.array([[5,1],[6,-1],[7,3]])}

print(SVM().fit(data_dict))