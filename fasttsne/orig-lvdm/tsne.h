/*
 *  tsne.h
 *  Header file for t-SNE.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 */


#ifndef TSNE_H
#define TSNE_H

enum {
    D_EUCLIDEAN = 0,
    D_NORMALIZED,
};

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }


class TSNE
{    
public:
    void run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int mode);

    void symmetrizeMatrix(int** row_P, int** col_P, double** val_P, int N); // should be static?!

    
private:
    void computeGradient(double* P, int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);
    double evaluateError(int* row_P, int* col_P, double* val_P, double* Y, int N, double theta);
    void zeroMean(double* X, int N, int D);
    void computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, int K);
    double randn();
};

#endif

