import scipy.linalg as la
import numpy as np


from fasttsne import _TSNE as TSNE


def fast_tsne(data, pca_d=None, d=2, perplexity=30., theta=0.5, cosine=0):
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

    @param cosine       Set to 1 to use cosine distance, 0 for euclidean distance.
    """

    # inplace!!

    if pca_d is None:
        X = data
    else:
        # do PCA
        print "Reducing to %dd using PCA..." % pca_d
        import sklearn.decomposition as deco
        norm_data = data - data.mean(axis=0)
        pca = deco.PCA(pca_d)
        X = pca.fit_transform(norm_data)
        print "%s -> %s" % (data.shape, X.shape)

    N, vlen = X.shape

    tsne = TSNE()
    Y = tsne.run(X, N, vlen, d, perplexity, theta, cosine)
    return Y
