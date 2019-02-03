import numpy as np

import scipy as sp
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import copy




def distcorr(u, v, pval, nruns=500):

    X = u.flatten().reshape(-1, 1)
    Y = v.flatten().reshape(-1, 1)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    if pval:
        greater = 0
        Y_r = Y.copy()
        for i in range(nruns):
            np.random.shuffle(Y_r)
            if distcorr(X.copy(), Y_r, pval=False) >= dcor:
                greater += 1
        return (dcor, greater / float(nruns))
    else:
        return dcor


def pearson(u, v, pval):
    if pval == True:
        corr, pval = sp.stats.pearsonr(u, v)
        return(float(corr), float(pval))
    else:
        return(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

def tau(u, v, pval):

    tau, pval = stats.stats.kendalltau(u, v)
    if pval == True:
        return(float(tau), float(pval))
    else:
        return(float(tau))


################################################################################################################################
################################################################################################################################
################################################################################################################################
# Here is the tau* method of calculating associations. Amazing method:
################################################################################################################################
################################################################################################################################
################################################################################################################################
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
utils = importr("TauStar")
ro.r('library("TauStar")')

def taustar(u, v, pval):

    ro.r('x = c{}'.format(tuple(u)))
    ro.r('y = c{}'.format(tuple(v)))
    tau_star = ro.r('tStar(x, y)')[0]
    if pval == False:
        return(tau_star)
    elif pval == True:
        ro.r('testResults = tauStarTest(x,y)')
        pvalue = ro.r('testResults$pVal[1]')[0]
        return([float(tau_star)] + [float(pvalue)])

def matrix_associations(A, method, pval=False):

    n, p = A.shape

    if method == "pearson":
        method = pearson
    elif method == "tau":
        method = tau
    elif method == "distcorr":
        method = distcorr
    elif method == "taustar":
        method = taustar
    if pval == False:   
        distances = [np.array([float(method(A[:, j], A[:, k], pval=pval)) for k in range(p) if j < k]) for j in range(p)]
        indeces = [[(j, k) for k in range(p) if j<k] for j in range(p)]
        return(indeces, distances)
    else:
        raise ValueError("pvalue should be False, since this option is not available yet.")



# A = np.random.uniform(0, 1, size = (100, 3))
# print(matrix_associations(A, "distcorr"))
# print(matrix_associations(A, pearson))
# print(matrix_associations(A, tau))
# print(matrix_associations(A, taustar))


