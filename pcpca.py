import numpy as np
from numba import jit
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal

import time

def getNonMissIndicatorMat(X, idx):
    # get the missing data indicator matrix (see 6.1 for definitionn of L and M) for the sample idx
    # the returned matrix is always sparse, anyway to speed up computation and use less memory?
    D = X.shape[1]
    notMissing = ~np.isnan(X[idx])
    indict = np.zeros((np.sum(notMissing), D))
    for i, non_missing_idx in enumerate(np.arange(D)[notMissing]):
        indict[i, non_missing_idx] = 1.0
    
    return indict

def getMissIndicatorMat(X, idx):
    # get the missing data indicator matrix (see 6.1 for definitionn of L and M) for the sample idx
    # the returned matrix is always sparse, anyway to speed up computation and use less memory?
    D = X.shape[1]
    Missing = np.isnan(X[idx])
    indict = np.zeros((np.sum(Missing), D))
    for i, missing_idx in enumerate(np.arange(D)[Missing]):
        indict[i, missing_idx] = 1.0
    return indict


def loglik_helper(X, W, sigma2):
    n, D = X.shape
    accu = 0.0
    for i in np.arange(n):
        Li = getNonMissIndicatorMat(X, i)
        Ai = Li @ (W @ W.T + sigma2*np.eye(D)) @ Li.T
        Di = Li.shape[0]
        Xi = X[i, ~np.isnan(X[i])]
        accu += Di*np.log(2*np.pi) + np.linalg.slogdet(Ai)[1] + np.trace(np.linalg.inv(Ai) @ Xi.reshape(Di, 1) @ Xi.reshape(1, Di))
    return accu

def calcGradient(X, Y, W, sigma2, gamma):
    # calculate gradient with respect to W and sigma2
    D = X.shape[1]
    n, m = X.shape[0], Y.shape[0]
    accu_Wf = np.zeros((D, D))
    accu_sigma2f = 0.0
    for i in np.arange(n):
        # calculate the part of gradient assoc with foreground data
        Li = getNonMissIndicatorMat(X, i)
        Di = Li.shape[0]
        Ai = Li @ (W @ W.T + sigma2*np.eye(D)) @ Li.T
        Ai_inv = np.linalg.inv(Ai)
        xi = X[i, ~np.isnan(X[i])]
        accu_Wf += Li.T @ Ai_inv @ (np.eye(Di) - xi.reshape(Di, 1) @ xi.reshape(1, Di) @ Ai_inv) @ Li
        accu_sigma2f += np.trace(Ai_inv @ Li @ Li.T) - np.trace(Ai_inv @ xi.reshape(Di, 1) @ xi.reshape(1, Di) @ Ai_inv @ Li @ Li.T)

    accu_Wb = np.zeros((D, D))
    accu_sigma2b = 0.0
    for j in np.arange(m):
        # calculate the part of gradient assoc with background data
        Mj = getNonMissIndicatorMat(Y, j)
        Ej = Mj.shape[0]
        Bj = Mj @ (W @ W.T + sigma2*np.eye(D)) @ Mj.T
        Bj_inv = np.linalg.inv(Bj)
        yj = Y[j, ~np.isnan(Y[j])]
        accu_Wb += Mj.T @ Bj_inv @ (np.eye(Ej) - yj.reshape(Ej, 1) @ yj.reshape(1, Ej) @ Bj_inv) @ Mj
        accu_sigma2b += np.trace(Bj_inv @ Mj @ Mj.T) - np.trace(Bj_inv @ yj.reshape(Ej, 1) @ yj.reshape(1, Ej) @ Bj_inv @ Mj @ Mj.T)

    return -(accu_Wf - gamma*accu_Wb) @ W, -0.5*accu_sigma2f + gamma*0.5*accu_sigma2b

def covVecProduct(X, v):
    # v: a vector (1-d, not a column vector)
    # return X.T@X@v
    # use this when n_features is very large so the sample covariance matrix can't be explicitly formed
    n, D = X.shape
    result = np.zeros(D)
    tmp = np.zeros(D)
    for i in range(n):
        np.multiply(X[i], np.sum(X[i]*v), out=tmp)
        np.add(result, tmp, out=result)
    return result


def ColSpaceApprox(A, r=10, tol=1e-8):
    # Implementation of algorithm 4.2 in Halko, Martinsson, and Tropp, SIREV, 2011.

    m, n = A.shape
    Omega = np.random.normal(loc=0.0, scale=1.0, size=(n, r))
    Y = A@Omega

    j = 0
    cutoff = tol/(10*np.sqrt(2/np.pi))
    Q = np.zeros((m, 0))

    while np.max(np.linalg.norm(Y[:, j:j+r], axis=0)) > cutoff:
        j += 1
        #print(f'j = {j}', flush=True)
        Y[:, j-1] = ((np.eye(m) - Q@Q.T)@(Y[:, j-1].reshape(m, 1))).flatten()
        Q = np.hstack((Q, (Y[:, j-1]/np.linalg.norm(Y[:, j-1])).reshape(m, 1)))

        omega_new = np.random.normal(loc=0.0, scale=1.0, size=(n, 1))
        y_new = (np.eye(m) - Q@Q.T) @ (A@omega_new)
        Y = np.hstack((Y, y_new))
        for i in range(j, j+r-1):
            Y[:, i] = Y[:, i] - Q[:,-1]*np.dot(Q[:,-1], Y[:,i])

    print(f'j = {j}', flush=True)
    #print(f'orthogonality of Q: {np.linalg.norm(Q.T@Q - np.eye(j))}')
    return Q


def randomEigenDecomp(A, r=10, tol=1e-8):
    # Assume A is hermitian, this is an implementation of 5.3 in Halko, Martinsson, and Tropp, SIREV, 2011.
    Q = ColSpaceApprox(A, r, tol)
    B = Q.T@A@Q
    lambdas, V = np.linalg.eigh(B)
    return lambdas, Q@V

def covCVecProduct(X, Y, gamma, v):
    # return (X.T@X - gamma*Y.T@Y)v
    return covVecProduct(X, v) - gamma*covVecProduct(Y, v)

def covCMatMul(X, Y, gamma, Omega):
    assert(X.shape[1] == Omega.shape[0])
    nrow, ncol = X.shape[1], Omega.shape[1]
    result = np.zeros((nrow, ncol))
    for j in range(ncol):
        result[:, j] = covCVecProduct(X, Y, gamma, Omega[:, j])
    return result

def ColSpaceApprox_MatC(Xdata, Ydata, gamma, r=10, tol=1e-8, oversampling=0.1):
    m = n = Xdata.shape[1]
    #Omega = multivariate_normal.rvs(mean=np.zeros(n), size=r).T
    Omega = np.random.normal(loc=0.0, scale=1.0, size=(n, r))
    Y = covCMatMul(Xdata, Ydata, gamma, Omega)

    j = 0
    cutoff = tol/(10*np.sqrt(2/np.pi))
    maxRank = np.min(Xdata.shape) + np.min(Ydata.shape)
    Q = np.zeros((m, 0))

    while np.max(np.linalg.norm(Y[:, j:j+r], axis=0)) > cutoff and j <= (1 + oversampling)*maxRank:
        t1 = time.time()
        j += 1
        Y[:, j-1] -= covVecProduct(Q.T, Y[:, j-1])
        Q = np.hstack((Q, (Y[:, j-1]/np.linalg.norm(Y[:, j-1])).reshape(m, 1)))

        #omega_new = np.random.multivariate_normal(np.zeros(n), np.eye(n))
        omega_new = np.random.normal(loc=0.0, scale=1.0, size=n)
        CtimesOmega_new = covCVecProduct(Xdata, Ydata, gamma, omega_new)
        y_new = CtimesOmega_new - covVecProduct(Q.T, CtimesOmega_new)
        Y = np.hstack((Y, y_new[:, np.newaxis]))
        for i in range(j, j+r-1):
            Y[:, i] = Y[:, i] - Q[:,-1]*np.dot(Q[:,-1], Y[:,i])
        #print(f'j = {j}, takes time: {time.time()-t1}', flush=True)

    print(f'j = {j}', flush=True)
    return Q

def covCRandomEigenDecomp(Xdata, Ydata, gamma, r=10, tol=1e-8):
    # Assume A is hermitian, this is an implementation of 5.3 in Halko, Martinsson, and Tropp, SIREV, 2011.
    Q = ColSpaceApprox_MatC(Xdata, Ydata, gamma, r, tol)
    B = Q.T@covCMatMul(Xdata, Ydata, gamma, Q)
    lambdas, V = np.linalg.eigh(B)
    return lambdas, Q@V


class pcpca:

    # X: n*d matrix, foregrounnd data
    # Y: m*d matrix, background data
    def __init__(self, X, Y, d=2):
        assert(X.shape[1] == Y.shape[1])
        #if X.shape[1] <= 1000:
        self.X = (X - np.nanmean(X, axis=0))/np.nanstd(X, axis=0)
        self.Y = (Y - np.nanmean(Y, axis=0))/np.nanstd(Y, axis=0)
        #else:
        #    pca = PCA(n_components=np.min([1000, X.shape[0], Y.shape[0]]))
        #    self.X = pca.fit_transform(X)
        #    self.Y = pca.fit_transform(Y)
        self.d = d
    
    # fit the PCPCA model as described in Theorem 3 in Didong et al.
    # return W, sigma^2
    # gamma: tuning parameter, should be within range (0, n/m)
    def fit(self, gamma, r=10, tol=1e-8):
        n, m = self.X.shape[0], self.Y.shape[0]
        D = self.X.shape[1]
        if D <= 1000:
            lambdas, V = np.linalg.eigh((self.X).T@self.X - gamma*(self.Y).T@self.Y)
        else:
            print(f"use randomized linear alg to avoid computing empirical covariance matrix explicitly", flush=True)
            lambdas, V = covCRandomEigenDecomp(self.X, self.Y, gamma, r, tol)
        order = np.argsort(lambdas)[::-1]
        lambdas = lambdas[order]
        V = V[:, order]
        sigma2 = np.sum(lambdas[self.d:])/((n - gamma*m)*(D - self.d))
        W = V[:,:self.d] @ sqrtm(np.diag(lambdas[:self.d]/(n - gamma*m) - sigma2))
        return W, sigma2

    def fitAndproject(self, gamma, r=10, tol=1e-8):
        if np.any(np.isnan(self.X)) or np.any(np.isnan(self.Y)):
            print(f"missingness in foreground data: {np.sum(np.isnan(self.X))/(self.X.shape[0]*self.X.shape[1])}")
            print(f"missingness in background data: {np.sum(np.isnan(self.Y))/(self.Y.shape[0]*self.Y.shape[1])}")
            W, _, X_imputed, Y_imputed = self.gradient_descent(gamma)
            return X_imputed@W, Y_imputed@W
        else:
            W, _ = self.fit(gamma, r, tol)
            return self.X@W, self.Y@W
    
    def __loglik(self, W, sigma2, gamma):
        return -0.5*loglik_helper(self.X, W, sigma2) + gamma*0.5*loglik_helper(self.Y, W, sigma2)

    def gradient_descent(self, gamma, maxIter=50, tol=1e-2):
        # initialize W and sigma2 by filling the nan value with column mean and do a MLE fit on it ?
        D = self.X.shape[1]
        np.random.seed(seed=2021)
        W = np.random.rand(D, self.d)
        sigma2 = 5.0

        X_copy, Y_copy = np.nan_to_num(self.X), np.nan_to_num(self.Y) # np.nan_to_num will make a copy of input
        pcpca_obj = pcpca(X_copy, Y_copy, self.d)
        W, sigma2 = pcpca_obj.fit(gamma)
        prev_loglik = self.__loglik(W, sigma2, gamma)
        print(f"initial log-likelihood: {prev_loglik}")
        niter = 0

        # set up params for Adam
        alpha = 0.01
        rho1 = 0.9
        rho2 = 0.999
        epsilon = 1e-5
        mom1_W = np.zeros_like(W)
        mom2_W = np.zeros_like(W)
        mom1_sigma2 = 0.0
        mom2_sigma2 = 0.0


        while niter < maxIter:
            gradW, gradSigma2 = calcGradient(self.X, self.Y, W, sigma2, gamma)
            niter += 1

            # update W and sigma2 by gradients
            mom1_W = rho1*mom1_W + (1-rho1)*gradW
            mom2_W = rho2*mom2_W + (1-rho2)*np.square(gradW)
            mom1_sigma2 = rho1*mom1_sigma2 + (1-rho1)*gradSigma2
            mom2_sigma2 = rho2*mom2_sigma2 + (1-rho2)*(gradSigma2**2)

            mom1_What = mom1_W/(1-rho1**niter)
            mom2_What = mom2_W/(1-rho2**niter)
            mom1_sigma2hat = mom1_sigma2/(1-rho1**niter)
            mom2_sigma2hat = mom2_sigma2/(1-rho2**niter)

            W += alpha*mom1_What/np.sqrt(epsilon + mom2_What)
            sigma2 += alpha*mom1_sigma2hat/np.sqrt(epsilon +  mom2_sigma2hat)

            # re-calculate loglikelihood and check if converged
            curr_loglik = self.__loglik(W, sigma2, gamma)
            print(f"current log-likelihood: {curr_loglik}")
            if (abs(prev_loglik - curr_loglik) < tol):
                print("converged!")
                break
            prev_loglik = curr_loglik

            
        # now impute missing data by posterior conditional mean
        n, m = self.X.shape[0], self.Y.shape[0]
        D = self.X.shape[1]
        X_imputed, Y_imputed = np.copy(self.X), np.copy(self.Y)
        for i in range(n):
            if np.sum(np.isnan(X_imputed[i])) == 0:
                continue
            Li = getNonMissIndicatorMat(X_imputed, i)
            Di = Li.shape[0]
            Ai = Li @ (W @ W.T + sigma2*np.eye(D)) @ Li.T
            Ai_inv = np.linalg.inv(Ai)
            xi = X_imputed[i, ~np.isnan(X_imputed[i])]
            Pi = getMissIndicatorMat(X_imputed, i)
            X_imputed[i, np.where(np.isnan(X_imputed[i]))[0]] = (Pi @ (W @ W.T + sigma2*np.eye(D)) @ Li.T @ Ai_inv @ xi.reshape(Di, 1)).flatten()

        for j in range(m):
            if np.sum(np.isnan(Y_imputed[j])) == 0:
                continue
            Mj = getNonMissIndicatorMat(Y_imputed, j)
            Ej = Mj.shape[0]
            Bj = Mj @ (W @ W.T + sigma2*np.eye(D)) @ Mj.T
            Bj_inv = np.linalg.inv(Bj)
            yj = Y_imputed[j, ~np.isnan(Y_imputed[j])]
            Pj = getMissIndicatorMat(Y_imputed, j)
            Y_imputed[j, np.where(np.isnan(Y_imputed[j]))[0]] = (Pj @ (W @ W.T + sigma2*np.eye(D)) @ Mj.T @ Bj_inv @ yj.reshape(Ej, 1)).flatten()

        return W, sigma2, X_imputed, Y_imputed
    
    def get_max_gamma(self):
        n, m, D = self.X.shape[0], self.Y.shape[0], self.X.shape[1]
        _, sx, _ = np.linalg.svd(self.X)
        _, sy, _ = np.linalg.svd(self.Y)
        eval_CX = np.square(sx)/n
        eval_CY = np.square(sy)/m
        return np.sum(eval_CX[self.d:])/((D-self.d)*eval_CY[0])


