#!/usr/bin/python
import gzip, cPickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

from fasttsne import fast_tsne

def load_mnist(fn="mnist.pkl.gz"):
    f = gzip.open(fn, "rb")
    train, val, test = cPickle.load(f)
    f.close()
    return train, val, test




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pca-first', type=int, default=0,
                       help='first reduce to this dimensionality with PCA')
    parser.add_argument('--normalise', action="store_true",
                       help='normalise each vector to a unit hypersphere')
    parser.add_argument('--small', action="store_true",
                       help='use 10k instead of 70k datapoints')
    parser.add_argument('-p', '--perplexity', type=float, default=30.0,
                       help='perplexity for t-SNE')

    args = parser.parse_args()

    train, val, test = load_mnist()

    if args.small:
        # Just use the validation examples (10k)
        mnist = np.asarray(val[0], dtype=np.float64)
        classes = val[1]
    else:
        # Get all data in one array
        mnist = np.vstack(np.asarray(x[0], dtype=np.float64)
                          for x in (train, val, test))
        # Also the classes, for labels in the plot later
        classes = np.hstack((train[1], val[1], test[1]))

    Y = fast_tsne(mnist, perplexity=args.perplexity, theta=0.5,
                  normalise=args.normalise, pca_d=args.pca_first)

    digits = set(classes)
    fig = plt.figure()
    colormap = plt.cm.spectral
    plt.gca().set_color_cycle(colormap(i) for i in np.linspace(0, 1.9, 20))
    ax = fig.add_subplot(111)
    labels = []
    for d in digits:
        idx = classes==d
        ax.plot(Y[idx, 0], Y[idx, 1], 'o')
        labels.append(d)
        ax.legend(labels, numpoints=1, fancybox=True)
    plt.show()

main()
