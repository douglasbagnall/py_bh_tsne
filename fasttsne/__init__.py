import scipy.linalg as la
import numpy as np
import time
import sys

from fasttsne import _TSNE as TSNE

def timed_reducer(f):
    def f2(data, d, mode, **kwargs):
        t = time.time()
        print >> sys.stderr, "Reducing to %dd using %s..." % (d, f.__name__)
        if mode == 1:
            from sklearn.preprocessing import Normalizer
            data = Normalizer(copy=False).fit_transform(data)
        X = f(data, d, mode, **kwargs)
        print >> sys.stderr, "%s -> %s. Took %.1fs" % (data.shape, X.shape, time.time() - t)
        return X
    return f2

@timed_reducer
def pca_reduce(data, pca_d, mode,
               algorithm='RandomizedPCA',
               whiten=False
           ):
    import sklearn.decomposition as deco
    alg = getattr(deco, algorithm)
    print >> sys.stderr, "pca..."
    try:
        pca = alg(n_components=pca_d, whiten=whiten)
    except TypeError, e:
        #some don't take whiten argument. If whiten is false, carry on.
        if whiten:
            raise
        pca = alg(n_components=pca_d)
    X = pca.fit_transform(data)
    return X

def fast_tsne(data, pca_d=None, d=2, perplexity=30., theta=0.5, mode=0, normalise_mean=0,
              whiten=0, pca_algo='RandomizedPCA'):
    """
    Run Barnes-Hut T-SNE on _data_.

    @param data         The data.

    @param pca_d        The dimensionality of data is reduced via PCA
                        to this dimensionality.

    @param d            The embedding dimensionality. Must be fixed to
                        2.

    @param perplexity   The perplexity controls the effective number of
                        neighbors.

    @param theta        Degree of BH optimisation (0-1; higher -> faster, worse).

    @param mode         0: Euclidean; 1: normalised Euclidean.

    @param normalise_mean Normalise mean around zero.

    @param whiten       Whiten when doing PCA.

    @param pca_algo     Which scikit-learn PCA implementation to use.
    """

    # inplace!!
    if normalise_mean:
        print >> sys.stderr, "normalising..."
        data = data - data.mean(axis=0)

    if not pca_d or pca_d > data.shape[1]:
        X = data
    else:
        X = pca_reduce(data, pca_d, mode, algorithm=pca_algo, whiten=whiten)
        del data

    N, vlen = X.shape
    print >> sys.stderr, X.shape

    tsne = TSNE()
    Y = tsne.run(X, N, vlen, d, perplexity, theta, mode)
    return Y
