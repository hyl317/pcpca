import numpy as np
import allel
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
from pcpca import *

chromLens = np.array([286.272, 268.839, 223.321, 214.555, 204.089, 192.04, 187.161, 168.003, 166.355, 181.131, 158.219,
    174.679, 125.703, 120.201, 141.843, 134.037, 128.49, 117.709, 107.732, 108.215, 62.771, 74.104])

def admixProp():
    admixPerChrom = defaultdict(lambda: [])
    for i in range(1, 23):
        with open(f'/fs/cbsubscb09/storage/yilei/simulate/1000G/rfmix_results/1000G.asw.rfmix.chr{i}.rfmix.Q') as rfmix:
            for line in rfmix:
                if line.startswith('#'):
                    continue
                else:
                    iid, ceu, yri = line.strip().split()
                    admixPerChrom[iid].append(float(yri))
    
    admixPropMap = {}
    for iid, yriPerChrom in admixPerChrom.items():
        admixPropMap[iid] = np.sum(np.array(yriPerChrom)*chromLens)/np.sum(chromLens)
    return admixPropMap


def readVCF(filePath):
    # read a vcf file, return a numpy array of dimention N*S
    # N = nunmber of diploid samples
    # S = number of SNPs
    # missing genotype should be represented as np.nan

    print(f"reading vcf file: {filePath}", flush=True)
    callset = allel.read_vcf(filePath)
    genotypes = callset["calldata/GT"].astype('float32')
    S, N, _ = genotypes.shape
    print(f"number of samples: {N}", flush=True)
    print(f"number of SNPs: {S}", flush=True)
    genotypes[genotypes == -1] = np.nan
    geno = np.sum(genotypes, axis=2).T
    return geno, callset["samples"]

def readVCFandRemoveNonPolymorphicSites(aswPath, ceuPath, yriPath):
    ASWgeno, ASWsamples = readVCF(aswPath)
    CEUgeno, _ = readVCF(ceuPath)
    YRIgeno, _ = readVCF(yriPath)
    background = np.concatenate((CEUgeno, YRIgeno), axis=0)
    tokeep_foreground = np.where(np.nanstd(ASWgeno, axis=0) != 0)[0]
    tokeep_background = np.where(np.nanstd(background, axis=0) != 0)[0]
    tokeep = np.intersect1d(tokeep_foreground, tokeep_background)
    print(f"remove {ASWgeno.shape[1] - len(tokeep)} non polymorphic sites")
    return ASWgeno[:, tokeep], background[:, tokeep], ASWsamples, CEUgeno.shape[0]



def pca_1kg(asw_geno, ceu_geno, yri_geno, asw_samples, admixPropMap):
    numASW = asw_geno.shape[0]
    numCEU = ceu_geno.shape[0]
    data = np.concatenate((asw_geno, ceu_geno, yri_geno), axis=0)
    pca = PCA(n_components=2)
    X_lowdim = pca.fit_transform(data)

    admix = []
    for sample in asw_samples:
        admix.append(admixPropMap[sample])

    plt.scatter(X_lowdim[:numASW, 0], X_lowdim[:numASW, 1], c=admix, marker="o", label="ASW")
    plt.scatter(X_lowdim[numASW:numASW + numCEU, 0], X_lowdim[numASW:numASW + numCEU, 1], color="grey", marker="^", label="CEU")
    plt.scatter(X_lowdim[numASW + numCEU:, 0], X_lowdim[numASW + numCEU:, 1], color="grey", marker="x", label="YRI")
    plt.legend(loc='upper left')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA on ASW vs. CEU & YRI')
    plt.savefig('1kg.pca.png', dpi=300)

def pcpca_1kg(foreground, background, asw_samples, admixPropMap, numCEU):

    admix = []
    for sample in asw_samples:
        admix.append(admixPropMap[sample])

    pcpca_obj = pcpca(foreground, background, 2)
    gammas = [0.0, 0.1, 0.2]
    fig, axes = plt.subplots(1,3, figsize=(30,10))
    fig.suptitle('1000Genome ASW vs CEU & YRI', fontsize=28)
    for i, gamma in enumerate(gammas):
        print(f"fitting gamma={gamma}", flush=True)
        X_lowdim, Y_lowdim = pcpca_obj.fitAndproject(gamma)
        axes[i].scatter(X_lowdim[:, 0], X_lowdim[:, 1], c=admix, marker='o', label="ASW")
        axes[i].scatter(Y_lowdim[:numCEU, 0], Y_lowdim[:numCEU, 1], c='grey', marker='^', label="CEU")
        axes[i].scatter(Y_lowdim[numCEU:, 0], Y_lowdim[numCEU:, 1], c='grey', marker='x', label="YRI")
        axes[i].set_title(rf"$\gamma=${gamma}")
        axes[i].legend(loc="upper left")
    plt.savefig("1kg.pcpca.png", dpi=300)

def ppca_1kg(foreground, background, asw_sampls, admixPropMap, nunmCEU):
    # I switch foreground and background data here for fun
    admix = []
    for sample in asw_samples:
        admix.append(admixPropMap[sample])

    pcpca_obj = pcpca(background, foreground, 2)
    X_lowdim, Y_lowdim = pcpca_obj.fitAndproject(0)
    plt.scatter(X_lowdim[:numCEU, 0], X_lowdim[:numCEU, 1], c='grey', marker='^', label='CEU')
    plt.scatter(X_lowdim[numCEU:, 0], X_lowdim[numCEU:, 1], c='grey', marker='x', label='YRI')
    plt.scatter(Y_lowdim[:, 0], Y_lowdim[:, 1], c=admix, marker='o', label='ASW')
    plt.legend(loc='upper left')
    plt.savefig('1kg.ppca.png', dpi=300)



if __name__ == "__main__":
    #ASW = "/fs/cbsubscb09/storage/yilei/simulate/1000G/unr_vcf/1000G.unr.ASW.ld_pruned.vcf.gz"
    #CEU = "/fs/cbsubscb09/storage/yilei/simulate/1000G/unr_vcf/1000G.unr.CEU.ld_pruned.vcf.gz"
    #YRI = "/fs/cbsubscb09/storage/yilei/simulate/1000G/unr_vcf/1000G.unr.YRI.ld_pruned.vcf.gz"
    ASW = "/fs/cbsubscb09/storage/yilei/simulate/1000G/unr_vcf/1000G.unr.ASW.vcf.gz"
    CEU = "/fs/cbsubscb09/storage/yilei/simulate/1000G/unr_vcf/1000G.unr.CEU.vcf.gz"
    YRI = "/fs/cbsubscb09/storage/yilei/simulate/1000G/unr_vcf/1000G.unr.YRI.vcf.gz"

    admixPropMap = admixProp()

    # run PCA
    #asw_geno, asw_samples = readVCF(ASW)
    #ceu_geno, _ = readVCF(CEU)
    #yri_geno, _ = readVCF(YRI)
    #pca_1kg(asw_geno, ceu_geno, yri_geno, asw_samples, admixPropMap)
    
    
    # run PCPCA
    foreground, background, asw_samples, numCEU = readVCFandRemoveNonPolymorphicSites(ASW, CEU, YRI)
    pcpca_1kg(foreground, background, asw_samples, admixPropMap, numCEU)
    #ppca_1kg(foreground, background, asw_samples, admixPropMap, numCEU)


