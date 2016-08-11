from math import sqrt

plot1 = [1,3]
plot2 = [2,5]

def calculate_distance(point1, point2):
    return sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)

print(calculate_distance(plot1, plot2))