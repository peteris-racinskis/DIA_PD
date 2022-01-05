import numpy as np
import random
from sys import float_info
from typing import List

from numpy.lib.shape_base import split
from datagen import STARTYEAR, STOPYEAR, OUTFILE as INFILE
OUTFILE=f"Results-{STARTYEAR}-{STOPYEAR}"
eps = float_info.epsilon
random.seed(100)

# How many times each country voted on by each other country?
def coincidence_count(tensor: np.ndarray) -> np.ndarray:
    binary = np.where(tensor == -1, 0, 1)
    return np.count_nonzero(binary, axis=0)

# total score received by country A from country B
def coincidence_total(tensor: np.ndarray) -> np.ndarray:
    zero_trim = np.where(tensor == -1, 0, tensor)
    return np.sum(zero_trim, axis=0)

def edge_average(coincidence: np.ndarray, totals: np.ndarray):
    n,_ = coincidence.shape
    return totals / (coincidence + eps)

def clear_dataset(dataset: np.ndarray, index: dict, threshold=10) -> np.ndarray:
    coincidence = coincidence_count(dataset)
    n, _ = coincidence.shape
    dropped =[]
    for i in range(n):
        if np.max(coincidence[i,:]) < threshold:
            dropped.append(countries[i])
    return clear_matrix(dataset, index, dropped)

def sorted_values(mat: np.ndarray, index: dict, inv=False):
    n, m = mat.shape
    sign = 1 if inv else -1
    l = []
    for i in range(n):
        for j in range(m):
            if i == j:
                continue
            l.append((index[j], index[i], mat[i][j]))
    return sorted(l, key=lambda x: sign * x[2])

def clear_matrix(dataset: np.ndarray, index: dict, dropped: list):
    ind = [index[d] for d in dropped]
    ax1, ax2 = (1, 0) if len(dataset.shape) == 2 else (2, 1)
    output = np.delete(dataset, [ind], axis=ax1)
    output = np.delete(output, [ind], axis=ax2) 
    for d in dropped:
        if d in index:
            index.pop(d)    
    new_index_base = sorted([(x,y) for x,y in index.items()], key=lambda z: z[1])
    countries = {}
    for i in range(len(new_index_base)):
        countries[i] = new_index_base[i][0]
    index = {v:k for k,v in countries.items()}
    return output, index, countries

def distance_measure(corr: np.ndarray) -> np.ndarray:
    return (corr + corr.T) / 2

# STEPS:
# 1. get probability a vote is j -> i
# 2. find quality correction: p_i * c_i = 1/n forall i
# 3. find bias correction: p_ij * c_i * c_j = 1/nn for all ij
# 4. bias correction can be applied to scores when converted to prob
def normalized_score(total: np.ndarray, alpha=1) -> np.ndarray:
    total = total + alpha - np.identity(total.shape[0]) * alpha
    N = np.sum(total)
    return total / N

# I keep getting this wrong otherwise
def expand(vector: np.ndarray, axis=0):
    N = vector.shape[0]
    res = np.zeros((N,N)) + vector
    return res.T if axis == 0 else res

# 2d probability distribution, get corrections to produce uniform distribution
def corrections(prob: np.ndarray) -> np.ndarray:
    K, _ = prob.shape
    row_prob = expand(np.sum(prob, axis=1))
    row_cor = 1 / (K * row_prob + eps)
    prob_cor = prob * row_cor
    col_prob = expand(np.sum(prob_cor, axis=0), 1)
    cor = col_prob / (K * prob_cor + eps)
    cor = np.where(np.identity(K) == 1, 0, cor)
    return cor

def weights(cij: np.ndarray) -> np.ndarray:
    N, _ = cij.shape
    return np.where(np.identity(N) == 1, 0, 1 / (cij + eps)) * 1 / N

def get_correction_matrix(dataset: np.ndarray) -> np.ndarray:
    coincidence = coincidence_count(dataset)
    totals = coincidence_total(dataset) # raw totals - more coincident, more score! need to average
    avg = edge_average(coincidence, totals) 
    score_probability = normalized_score(avg, alpha=0.3)
    cors = corrections(score_probability)
    return cors

def write_matrix(filename, comment, matrix: np.ndarray):
    with open(filename, 'w') as  f:
        f.write(f"#{comment}\n#\n")
        for row in matrix:
            s = ["{:.4f}".format(x) for x in row]
            f.write(" ".join(s) + '\n')

def apply_correction_matrix(data: np.ndarray, corr: np.ndarray) -> np.ndarray:
    nonnegative = np.where(data == -1, 0, data)
    negative = np.where(data == -1, data, 0)
    corrected = nonnegative * corr
    #corrected = nonnegative * np.sqrt(corr)
    return corrected + negative

def rank_corrected(dataset: np.ndarray) -> np.ndarray:
    _, N, _ = dataset.shape
    out = np.copy(dataset)
    for stage in out:
        for i in range(N):
            scores = stage[:,i]
            if np.max(scores) < 10:
                continue
            sort_indices = np.argsort(scores)
            last = sort_indices[-10:]
            sclist = np.arange(1,11,1)
            scores[last] = sclist
    return out.astype(np.int64)

def weighted_transform(dataset: np.ndarray) -> np.ndarray:
    d = np.where(dataset == 10, 12, dataset)
    d = np.where(d == 9, 10, d)
    return d

def total_scores(dataset: np.ndarray) -> np.ndarray:
    ax = 1 if len(dataset.shape) == 2 else 2
    return np.sum(np.where(dataset == -1, 0, dataset), axis=ax)

def pre_post_variance(train: np.ndarray, test: np.ndarray):
    # compute correction matrix on training data
    cor_matrix = get_correction_matrix(train)
    # find weights, corrections for test data set, compute variances
    v1 = np.average(np.var(np.where(test == -1, 0, test), axis=(1,2)))
    # get the correction matrix, weights for test data set
    # did the variance go down?
    corrected_dataset = apply_correction_matrix(test, cor_matrix)
    v2 = np.average(np.var(np.where(corrected_dataset == -1, 0, corrected_dataset), axis=(1,2)))
    return v1, v2

def split_dataset(dataset: np.ndarray, testsize=5) -> List[np.ndarray]:
    T = dataset.shape[0]
    remaining = T
    test_indices = []
    train_indices = [x for x in range(T)]
    for _ in range(testsize):
        r = random.random() * remaining
        i = int(r // 1)
        test_indices.append(train_indices[i])
        del train_indices[i]
        remaining -= 1
    return dataset[train_indices,:,:], dataset[test_indices,:,:]

if __name__ == "__main__":
    shapefile = INFILE + "-shape.txt"
    stagefile = INFILE + "-stages.txt"
    countryfile = INFILE + "-countries.txt"
    datafile = INFILE + "-data.bin"
    with open(shapefile, 'r') as f:
        lines = f.readlines()
        shape = tuple(int(x) for x in lines)
    with open(stagefile, 'r') as f:
        stages = {int(z):y[:-1] for z,y in [x.split("\t") for x in f.readlines()]}
    with open(countryfile, 'r') as f:
        countries = {int(z):y[:-1] for z,y in [x.split("\t") for x in f.readlines()]}
        index = {v:k for k,v in countries.items()}
    
    arr = np.reshape(np.fromfile(datafile, dtype=np.int64), shape)
    arr_c, index, countries = clear_dataset(arr, index)
    
    print("Variance before and after correction. Train set - 55; test set 55")
    v1, v2 = pre_post_variance(arr_c, arr_c)
    print("{:.5f} / {:.5f} : {:.5f}".format(v1,v2,v2-v1))
    print("Variance before and after correction. Train set - 51; test set 5; random splits")
    for _ in range(10):
        trn, tst = split_dataset(arr_c)
        v1, v2 = pre_post_variance(trn, tst)
        print("{:.5f} / {:.5f} : {:.5f}".format(v1,v2,v2-v1))
    
    correction_matrix = get_correction_matrix(arr_c)
    voting_weights = weights(correction_matrix)
    dist_matrix = distance_measure(correction_matrix)
    d_list = sorted_values(dist_matrix, countries, inv=True)
    newshape = correction_matrix.shape

    biasfile = OUTFILE + "-bias.txt"
    corrfile = OUTFILE + "-corr.txt"
    distfile = OUTFILE + "-dist.txt"
    namefile = OUTFILE + "-name.txt"
    shapefile_new = OUTFILE + "-shape.txt"
    write_matrix(biasfile, 
    "Model weights: score = w_i * s_i, with s_i being latent song quality variable", 
    voting_weights)
    write_matrix(corrfile, 
    "Correction matrix for multiplying with normalized score matrix", 
    correction_matrix)
    write_matrix(distfile, 
    "Distance matrix from Eurovision score data. p_ij * p_ji / 2", 
    dist_matrix)
    with open(namefile, 'w') as f:
        [f.write(f"{y}\n") for x,y in countries.items()]
    with open(shapefile_new, 'w') as f:
        [f.write(str(x) + "\n") for x in newshape]
    
    print("10 closest distances:")
    for i in range(10):
        print(d_list[i*2])
    print("10 farthest distances:")
    for i in range(10):
        print(d_list[len(d_list)-1-i*2])
    print("Distances to Latvia:")
    ds = [x for x in filter(lambda x: x[0] == "lv", d_list)]
    [print(x) for x in ds[:5]]
    print("...")
    [print(x) for x in ds[-5:]]