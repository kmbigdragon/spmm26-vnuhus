import numpy as np
import matplotlib.pyplot as plt

def myFunc(n):
    r = 3.0
    tmp = np.exp(r*(1-n))*n
    return tmp

tEnd = 50
t = 0
popSize = 1
while t < tEnd:
    print("time =" , t, popSize)
    popSize = popSize * myFunc(0.5)
    t = t + 1