import scipy.linalg as la
import numpy as np


from fasttsne import _TSNE as TSNE


def fast_tsne(data, pca_d=None, d=2, perplexity=30., theta=0.5, mode=0):
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
    """

    # inplace!!

    if not pca_d:
        X = data
    else:
        # do PCA
        print "Reducing to %dd using PCA..." % pca_d
        if mode == 1:
            from sklearn.preprocessing import Normalizer
            data = Normalizer().fit_transform(data)

        import sklearn.decomposition as deco
        print "normalising..."
        data = data - data.mean(axis=0)
        print "pca..."
        #pca = deco.RandomizedPCA(pca_d)
        pca = deco.TruncatedSVD(n_components=pca_d)
        X = pca.fit_transform(data)
        print "%s -> %s" % (data.shape, X.shape)
        del data

    N, vlen = X.shape
    print X.shape

    tsne = TSNE()
    Y = tsne.run(X, N, vlen, d, perplexity, theta, mode)
    return Y
