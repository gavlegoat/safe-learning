from random import random
from numpy import linspace
from numpy.random import random as random_array
from matplotlib import pyplot as plt

import numpy

LEFT_X = -12
RIGHT_X = 12
LIMITS_X = (LEFT_X, RIGHT_X)

UPPER_Y = 12
BOTTOM_Y = -12
LIMITS_Y = (BOTTOM_Y, UPPER_Y)

RESOLUTION = 1000

def get_poly(x):
    return ((x-2)**2, x)

a = numpy.array([1,0])
b = numpy.array([1,0])
c = -9

print a 
print b
print c

x = linspace(LEFT_X, RIGHT_X, RESOLUTION)
y = linspace(BOTTOM_Y, UPPER_Y, RESOLUTION)
x_poly = get_poly(x)
y_poly = get_poly(y)

eq = a.dot(x_poly) + b.dot(y_poly)[:,None] + c
print eq
#x**2 + y**2 - 1

plt.contour(x, y[:,None].ravel(), eq, [1])
plt.show()