from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use('ggplot')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(no_datapoints, variance, step=2, correlation=False):
    val=1
    ys=[]
    for i in range(no_datapoints):
        val = 1
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = ( (mean(xs)*mean(ys)) - (mean(xs*ys)) ) / ( ((mean(xs))**2) - (mean(xs**2)) )
    b = mean(ys) - m * (mean(xs))
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_orig-ys_line)**2)

def r_squared(ys_orig, regression_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regression = squared_error(ys, regression_line)
    squared_error_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regression / squared_error_mean)


xs, ys = create_dataset(40, 100, 2, correlation='pos')

m, b = best_fit_slope_and_intercept(xs,ys)

predict_x = 7
predict_y = (m * predict_x) + b

regression_line = [(m*x)+b for x in xs]

coefficient_of_determination = r_squared(ys, regression_line)
print('coefficient of determination: ' + str(coefficient_of_determination))

plt.scatter(xs, ys, marker='*', color='#FF00D9')
# plt.scatter(predict_x, predict_y, marker='_', color='#003F72')
plt.plot(xs, regression_line)
plt.show()
print(m, b)
