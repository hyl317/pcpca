from pcpca import *
#import pcpca_ac
import time
from scipy.stats import ortho_group
import matplotlib.pyplot as plt 

import sys

# Simulate data to visulize role of \gamma in PCPCA as described in 5.1 in Didong et al.
def simulate_2d():
    n = 200
    m = 200
    sigma = np.array([[2.7, 2.6], [2.6, 2.7]])
    X = np.zeros((n, 2))
    half = int(n/2)
    X[:half] = np.random.multivariate_normal((1, -1), sigma, half)
    X[half:] = np.random.multivariate_normal((-1, 1), sigma, half)
    Y = np.random.multivariate_normal((0, 0), sigma, m)

    gammas = [0, 0.25, 0.75, 0.99]
    fig, axes = plt.subplots(1,4, figsize=(30,10))
    fig.suptitle('a simple 2D simulation', fontsize=28)
    pcpca_obj = pcpca(X, Y, 1)
    for i, gamma in enumerate(gammas):
        W, sigma2 = pcpca_obj.fit(gamma)

        # for debugging purpose
        # pca_obj = pcpca_ac.PCPCA(1, gamma)
        # X_tmp = X - np.mean(X, axis=0)
        # Y_tmp = Y - np.mean(X, axis=0)
        # pca_obj.fit(X_tmp.T, Y_tmp.T)
        # print(f"fitted sigma2 from ac implementation: {pca_obj.sigma2_mle}")
        # print(f"fitted W from ac implementation: {pca_obj.W_mle}")
        # print("--------------------------------------------------")



        axes[i].set_xlim(-6, 6)
        axes[i].set_ylim(-6, 6)
        axes[i].scatter(X[:half, 0], X[:half, 1], label="foreground 1", color="red")
        axes[i].scatter(X[half:, 0], X[half:, 1], label="foreground 2", color="blue")
        axes[i].scatter(Y[:, 0], Y[:, 1], label="background", color="gray")
        # plot the line defined by W
        slope = W[1, 0]/W[0, 0]
        xs = np.linspace(-10, 10, 1000)
        ys = slope*xs
        axes[i].plot(xs[np.abs(ys)<=10], ys[np.abs(ys)<=10], linestyle="dashed")
        axes[i].legend(loc="upper right", fontsize="medium")
        axes[i].set_title(rf"$\gamma=${gamma}")

    plt.savefig("2d_simu.png", dpi=300)


def mice_protein():
    prefix = "C:\\Users\\10453\\Desktop\\Spring2021\\contrastive\\experiments\\datasets"
    data = np.genfromtxt(f'{prefix}\\Data_Cortex_Nuclear.csv',delimiter=',',
                     skip_header=1,usecols=range(1,78),filling_values=0)
    classes = np.genfromtxt(f'{prefix}\\Data_Cortex_Nuclear.csv',delimiter=',',
                        skip_header=1,usecols=range(78,81),dtype=None)
    target_idx_A = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))[0]
    target_idx_B = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Ts65Dn'))[0]

    target_idx = np.concatenate((target_idx_A,target_idx_B))                                                                          
    target = data[target_idx]

    background_idx = np.where((classes[:,-1]==b'C/S') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))
    background = data[background_idx]
    print(f"target data dim: {target.shape}")
    print(f"background data dim: {background.shape}")

    pcpca_obj = pcpca(target, background, 2)
    gammas = [0.0, 0.5, 0.9, 0.9999]
    fig, axes = plt.subplots(1,4, figsize=(30,10))
    fig.suptitle('Mice Protein Expression Data', fontsize=28)
    for i, gamma in enumerate(gammas):
        X_lowdim, Y_lowdim = pcpca_obj.fitAndproject(gamma)
        axes[i].scatter(X_lowdim[:len(target_idx_A), 0], X_lowdim[:len(target_idx_A), 1], color='red')
        axes[i].scatter(X_lowdim[len(target_idx_A):, 0], X_lowdim[len(target_idx_A):, 1], color='blue')
        axes[i].set_title(rf"$\gamma=${gamma}")
    plt.savefig("mice_protein_rand.png", dpi=300)

def mice_protein_miss_data_gradDescent():
    prefix = "C:\\Users\\10453\\Desktop\\Spring2021\\contrastive\\experiments\\datasets"
    data = np.genfromtxt(f'{prefix}\\Data_Cortex_Nuclear.csv',delimiter=',',
                     skip_header=1,usecols=range(1,78),filling_values=0)
    classes = np.genfromtxt(f'{prefix}\\Data_Cortex_Nuclear.csv',delimiter=',',
                        skip_header=1,usecols=range(78,81),dtype=None)
    target_idx_A = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))[0]
    target_idx_B = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Ts65Dn'))[0]

    target_idx = np.concatenate((target_idx_A,target_idx_B))                                                                          
    target = data[target_idx]

    background_idx = np.where((classes[:,-1]==b'C/S') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))
    background = data[background_idx]
    print(f"target data dim: {target.shape}")
    print(f"background data dim: {background.shape}")

    gamma = 0.5
    ps = [0.0, 0.05, 0.15, 0.3, 0.6, 0.9]
    fig, axes = plt.subplots(1, 6, figsize=(30,10))
    fig.suptitle('Mice Protein Expression Data with Missingness: Gradient Descent', fontsize=28)
    for i, prob in enumerate(ps):
        target_tmp = np.copy(target)
        background_tmp = np.copy(background)
        # randomly add missingness to the foreground and background data
        n, D = target_tmp.shape
        m = background_tmp.shape[0]
        target_tmp[np.random.rand(n, D) <= prob] = np.nan
        background_tmp[np.random.rand(m, D) <= prob] = np.nan

        pcpca_obj = pcpca(target_tmp, background_tmp, 2)
        X_lowdim, Y_lowdim = pcpca_obj.fitAndproject(gamma)
        axes[i].scatter(X_lowdim[:len(target_idx_A), 0], X_lowdim[:len(target_idx_A), 1], color='red')
        axes[i].scatter(X_lowdim[len(target_idx_A):, 0], X_lowdim[len(target_idx_A):, 1], color='blue')
        axes[i].set_title(rf"$missingness=${prob}")
    plt.savefig("mice_protein_miss_gd.png", dpi=300)


def mice_protein_miss_data_fillbymean():
    prefix = "C:\\Users\\10453\\Desktop\\Spring2021\\contrastive\\experiments\\datasets"
    data = np.genfromtxt(f'{prefix}\\Data_Cortex_Nuclear.csv',delimiter=',',
                     skip_header=1,usecols=range(1,78),filling_values=0)
    classes = np.genfromtxt(f'{prefix}\\Data_Cortex_Nuclear.csv',delimiter=',',
                        skip_header=1,usecols=range(78,81),dtype=None)
    target_idx_A = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))[0]
    target_idx_B = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Ts65Dn'))[0]

    target_idx = np.concatenate((target_idx_A,target_idx_B))                                                                          
    target = data[target_idx]

    background_idx = np.where((classes[:,-1]==b'C/S') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))
    background = data[background_idx]
    print(f"target data dim: {target.shape}")
    print(f"background data dim: {background.shape}")

    gamma = 0.5
    ps = [0.0, 0.05, 0.15, 0.3, 0.6, 0.9]
    fig, axes = plt.subplots(1, 6, figsize=(30,10))
    fig.suptitle('Mice Protein Expression Data with Missingness: Fill by Mean', fontsize=28)
    for i, prob in enumerate(ps):
        target_tmp = np.copy(target)
        background_tmp = np.copy(background)
        # randomly add missingness to the foreground and background data
        n, D = target_tmp.shape
        m = background_tmp.shape[0]
        target_tmp[np.random.rand(n, D) <= prob] = np.nan
        background_tmp[np.random.rand(m, D) <= prob] = np.nan

        # fill missing data by mean value
        col_mean = np.nanmean(target_tmp, axis=0)
        inds = np.where(np.isnan(target_tmp))
        target_tmp[inds] = np.take(col_mean, inds[1])

        col_mean = np.nanmean(background_tmp, axis=0)
        inds = np.where(np.isnan(background_tmp))
        background_tmp[inds] = np.take(col_mean, inds[1])

        pcpca_obj = pcpca(target_tmp, background_tmp, 2)
        X_lowdim, Y_lowdim = pcpca_obj.fitAndproject(gamma)
        axes[i].scatter(X_lowdim[:len(target_idx_A), 0], X_lowdim[:len(target_idx_A), 1], color='red')
        axes[i].scatter(X_lowdim[len(target_idx_A):, 0], X_lowdim[len(target_idx_A):, 1], color='blue')
        axes[i].set_title(rf"$missingness=${prob}")
    plt.savefig("mice_protein_miss_fillbymean.png", dpi=300)





def test_covVecProduct():
    n = 500
    D = 250000
    rep = 5
    X = np.random.rand(n, D)
    v = np.random.rand(D)
    t1 = time.time()
    for i in range(rep):
        result = covVecProduct(X, v)
    print(f"my version takes: {(time.time()-t1)/rep}")


def test_colSpaceApprox():
    print("A rank 20 example")
    base = np.random.rand(500, 20)
    A = base@base.T
    Q = ColSpaceApprox(A)
    print(f"approximation error: {np.linalg.norm(A - Q@Q.T@A)}")

    print("A rank 30 example")
    base = np.random.rand(500, 30)
    A = base@base.T
    Q = ColSpaceApprox(A)
    print(f"approximation error: {np.linalg.norm(A - Q@Q.T@A)}")

    print("A rank 50 example")
    base = np.random.rand(500, 50)
    A = base@base.T
    Q = ColSpaceApprox(A)
    print(f"approximation error: {np.linalg.norm(A - Q@Q.T@A)}")

    print("A rank 100 example")
    base = np.random.rand(500, 100)
    A = base@base.T
    Q = ColSpaceApprox(A)
    print(f"approximation error: {np.linalg.norm(A - Q@Q.T@A)}")

    print("A rank 250 example")
    base = np.random.rand(500, 250)
    A = base@base.T
    Q = ColSpaceApprox(A)
    print(f"approximation error: {np.linalg.norm(A - Q@Q.T@A)}")




def plot_colSpaceApprox():
    Sigma = np.zeros((1000, 1000))
    diag = np.concatenate((np.linspace(10000,0,101)[:-1], np.zeros(900)))
    np.fill_diagonal(Sigma, diag)

    U = ortho_group.rvs(1000)
    A = U@Sigma@U.T

    Sigma_random, _  = randomEigenDecomp(A)
    Sigma_random = Sigma_random[np.argsort(Sigma_random)[::-1]]
    Sigma_np, _ = np.linalg.eigh(A)
    Sigma_np = Sigma_np[np.argsort(Sigma_np)[::-1]]

    plt.plot(np.arange(1, 101), Sigma_random[:100], label="random linalg", linewidth=0.5)
    plt.plot(np.arange(1, 101), Sigma_np[:100], label="np.linalg.eigh", linewidth=0.5)
    plt.plot(np.arange(1, 101), np.linspace(10000, 0, 101)[:-1], label="exact", linewidth=0.5,color='black')
    plt.yscale('log')
    plt.xlabel('j')
    plt.title(rf'Estimated Eigenvalues $\lambda_j$')
    plt.legend(loc='upper right')
    plt.savefig('eigen.png', dpi=300)


def test_covCRandomEigenDecomp():
    n = 100
    D = 1000
    X = np.random.rand(n, D)
    Y = np.random.rand(n, D)
    gamma = 0.1

    #print(f'rank of C matrix: {np.linalg.matrix_rank(X.T@X-gamma*Y.T@Y)}')

    t1 = time.time()
    lambdas1, V1 = covCRandomEigenDecomp(X, Y, gamma)
    print(f'time for random eigen decomp without explicit forminng sample cov: {time.time()-t1}')
    #sys.exit()

    #t1 = time.time()
    #lambdas2, V2 = randomEigenDecomp(X.T@X -gamma*Y.T@Y)
    #print(f'time for random eigen decomp with forminng sample cov: {time.time()-t1}')



    # Use the std numpy operation
    tmpX = np.zeros((D, D))
    tmpY = np.zeros((D, D))
    tmp = np.zeros((D, D))
    t1 = time.time()
    np.matmul(X.T, X, out=tmpX)
    np.matmul(Y.T, Y, out=tmpY)
    np.multiply(gamma, tmpY, out=tmp)
    np.subtract(tmpX, tmp, out=tmp)
    print(f'time for calculating matmul: {time.time()-t1}')
    #print(f'check my noalloc matmul: {np.linalg.norm(tmp-(X.T@X-gamma*Y.T@Y))}')
    lambdas, V = np.linalg.eigh(tmp)
    print(f'time for det eigen decomp: {time.time()-t1}')

    # lambdas = lambdas[np.argsort(lambdas)[::-1]]
    # print(lambdas)

    # lambdas1 = lambdas1[np.argsort(lambdas1)[::-1]]
    # lambdas2 = lambdas2[np.argsort(lambdas2)[::-1]]
    # print(lambdas1)
    #print(f'diff1: {lambdas[:len(lambdas1)]-lambdas1}')
    #print(f'diff2: {lambdas[:len(lambdas2)]-lambdas2}')

    #r = len(lambdas1)
    #plt.plot(np.arange(r), lambdas[:r], linewidth=0.8)
    #plt.plot(np.arange(r), lambdas1, linewidth=0.8)
    #plt.plot(np.arange(r), lambdas2, linewidth=0.8)
    #plt.yscale('log')
    #plt.show()

    # _, sigma, _ = np.linalg.svd(X.T@X-gamma*Y.T@Y)
    # plt.plot(np.arange(200), sigma[:200])
    # plt.yscale('log')
    # plt.show()

def runtime_D():
    n = 100
    Ds = [500, 1000, 2500, 5000, 7500, 10000]
    rand = []
    det = []
    rep = 3
    gamma = 0.2
    for D in Ds:        
        X = np.random.rand(n, D)
        Y = np.random.rand(n, D)

        t1 = time.time()
        for i in range(rep):
            lambdas1, V1 = covCRandomEigenDecomp(X, Y, gamma)
        rand.append((time.time()-t1)/rep)

        tmpX = np.zeros((D, D))
        tmpY = np.zeros((D, D))
        tmp = np.zeros((D, D))
        t1 = time.time()
        for i in range(rep): 
            np.matmul(X.T, X, out=tmpX)
            np.matmul(Y.T, Y, out=tmpY)
            np.multiply(gamma, tmpY, out=tmp)
            np.subtract(tmpX, tmp, out=tmp)
            lambdas, V = np.linalg.eigh(tmp)
        det.append((time.time()-t1)/rep)
    
    plt.plot(Ds, rand, label='randomized scheme', linestyle='--', marker='o')
    plt.plot(Ds, det, label='deterministic scheme', linestyle='--', marker='o')
    plt.xlabel('D')
    plt.ylabel('wall clock time')
    plt.title('runtime comparison')
    plt.legend(loc='upper left')
    plt.savefig('runtime_D.png', dpi=300)

def runtime_rank():
    ns = [50, 100, 200, 500, 1000, 2500]
    D = 10000
    gamma = 0.2
    for n in ns:
        print(f'n: {n}')
        X = np.random.rand(n, D)
        Y = np.random.rand(n, D)
        Q = ColSpaceApprox_MatC(X, Y, gamma)
        tmp = X.T@X - gamma*Y.T@Y
        print(f'approx error: {np.linalg.norm(tmp - Q@Q.T@tmp)}')


def test_tols():
    prefix = "C:\\Users\\10453\\Desktop\\Spring2021\\contrastive\\experiments\\datasets"
    data = np.genfromtxt(f'{prefix}\\Data_Cortex_Nuclear.csv',delimiter=',',
                     skip_header=1,usecols=range(1,78),filling_values=0)
    classes = np.genfromtxt(f'{prefix}\\Data_Cortex_Nuclear.csv',delimiter=',',
                        skip_header=1,usecols=range(78,81),dtype=None)
    target_idx_A = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))[0]
    target_idx_B = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Ts65Dn'))[0]

    target_idx = np.concatenate((target_idx_A,target_idx_B))                                                                          
    target = data[target_idx]

    background_idx = np.where((classes[:,-1]==b'C/S') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))
    background = data[background_idx]
    print(f"target data dim: {target.shape}")
    print(f"background data dim: {background.shape}")

    pcpca_obj = pcpca(target, background, 2)
    fig, axes = plt.subplots(1,4, figsize=(30,10))

    tols = [1e-2, 1e-4, 1e-6, 1e-8]
    gamma = 0.5
    for i, tol in enumerate(tols):
        X_lowdim, Y_lowdim = pcpca_obj.fitAndproject(gamma, tol=tol)
        axes[i].scatter(X_lowdim[:len(target_idx_A), 0], X_lowdim[:len(target_idx_A), 1], color='red')
        axes[i].scatter(X_lowdim[len(target_idx_A):, 0], X_lowdim[len(target_idx_A):, 1], color='blue')
        axes[i].set_title(rf"$\epsilon=${tol}")
    plt.savefig("mice_protein_tols.png", dpi=300)



if __name__ == "__main__":
    #simulate_2d()
    mice_protein()
    #mice_protein_miss_data_gradDescent()
    #mice_protein_miss_data_fillbymean()
    #test_covVecProduct()
    #test_colSpaceApprox()
    #plot_colSpaceApprox()
    #test_covCRandomEigenDecomp()
    #runtime_D()
    #runtime_rank()
    #test_tols()