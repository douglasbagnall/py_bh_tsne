#!/usr/bin/python
import gzip, cPickle
import numpy as np
import matplotlib.pyplot as plt
import sys

from fasttsne import fast_tsne
import random
random.seed(1)

def generate_angular_clusters(n, d, extra_d=10):
    data = []
    classes = []
    for i in range(n):
        scale = random.random() * 5 + 0.1
        centre = [random.randrange(2) for x in range(d)]
        _class = ''.join(str(x) for x in centre)
        row = [scale * (random.random() * 0.05 + x - 0.5)
               for x in centre]
        row += [random.random() * 0.01 for x in range(extra_d)]
        data.append(row)
        classes.append(_class)
    data = np.asarray(data)
    return data, classes



def main():
    data, classes = generate_angular_clusters(5000, 4)
    #print data
    Y = fast_tsne(data, perplexity=10, normalise=1)
    #print zip(classes, Y)[:50]
    digits = set(classes)
    fig = plt.figure()
    colormap = plt.cm.spectral
    plt.gca().set_color_cycle(colormap(i) for i in np.linspace(0, 0.6, 10))
    ax = fig.add_subplot(111)
    labels = []
    for d in sorted(digits):
        idx = np.array([x == d for x in classes], dtype=np.bool)
        ax.plot(Y[idx, 0], Y[idx, 1], 'o')
        labels.append(d)
    ax.legend(labels, numpoints=1, fancybox=True)
    plt.show()

main()
