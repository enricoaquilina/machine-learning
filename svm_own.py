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
                        for yi in self.features:
                            for xi in self.features[yi]:
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
        if classification != 0 and self.visualisation:
            self.ax.scatter(features[0], features[1],s=200,marker='*',c=self.colors[classification])
        else:
            print('featureset ',features,' is on the boundary')

        return classification

    def visualise(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        #hyperplane = x.w+b
        # v = x.w + b
        # psv = 1
        # nsv = -1
        # db = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]
        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # 1= w.x +b
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min,self.w,self.b,1)
        psv2 = hyperplane(hyp_x_max,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])

        # -1= w.x +b
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        # 0= w.x +b
        # decision boundary
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2])

        plt.show()


data_dict = {-1: np.array([[1,7],[2,8],[3,8]]),
              1: np.array([[7,8],[8,3],[9,7]])}

svm = SVM()
svm.fit(data_dict)
svm.visualise()