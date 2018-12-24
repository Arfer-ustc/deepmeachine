import copy
import numpy as np
import pandas as pd
from sklearn import datasets
import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.animation as ani

w = [0, 0]
b = 0
yita = 0.5
data = [[(1, 4), 1], [(0.5, 2), 1], [(2, 2.3), 1], [(1, 0.5), -1],
        [(2, 1), -1], [(4, 1), -1], [(3.5, 4), 1], [(3, 2.2), -1]]

record = []

'''
if y(wx+b) <=0, return false, else return true
'''

def sign(vec):
    global w, b
    res = 0
    res = vec[1]*(w[0]*vec[0][0]+w[1]*vec[0][1]+b)

    if res >0: return 1
    else: return -1

def update(vec):
    global w, b, record
    w[0] = w[0] + yita * vec[1] * vec[0][0]
    w[1] = w[1] + yita * vec[1] * vec[0][1]
    b = b+yita*vec[1]

    record.append([copy.copy(w), b])

def perceptron():
    count = 1
    for ele in data:
        flag = sign(ele)
        if not flag > 0:
            count = 1
            update(ele)
        else:
            count += 1
        if count >= len(data):
            return 1

x1 = []
y1 = []
x2 = []
y2 = []

fig = plt.figure()
ax = plt.axes(xlim = (-1, 5), ylim = (-1, 5))
line, = ax.plot([], [], 'g', lw=2)

def init():
    line.set_data([], [])
    for p in data:
        if p[1] > 0:
            x1.append(p[0][0])
            y1.append(p[0][1])
        else:
            x2.append(p[0][0])
            y2.append(p[0][1])
    plt.plot(x1, y1, 'or')
    plt.plot(x2, y2, 'ob')
    # print('x1 y1 x2 y2: ', x1, y1, x2, y2)
    return line,

def animate(i):
    global record, ax, line
    w = record[i][0]
    b = record[i][1]
    x1 = -5
    y1 = -(b+w[0]*x1)/w[1]
    x2 = 6
    y2 = -(b+w[0]*x2)/w[1]
    line.set_data([x1, x2], [y1, y2])
    print(type(line), line)
    print([x1, x2], [y1, y2])
    return line,


if __name__ == '__main__':
    while 1:
        acc = perceptron()
        # isinstance(acc, None)
        if  acc != None and acc > 0:
            break
    print(record)

    animat = ani.FuncAnimation(fig, animate, init_func=init, frames=len(record),
                               interval=1000, repeat=True, blit=True)
    plt.show()
    animat.save('perceptron.gif', fps=2, writer='pillow')








