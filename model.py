import numpy as np
from sys import float_info
from datagen import STARTYEAR, STOPYEAR, OUTFILE as INFILE
OUTFILE=f"Results-{STARTYEAR}-{STOPYEAR}"
eps = float_info.epsilon

# How many times each country voted on by each other country?
def coincidence_count(tensor: np.ndarray) -> np.ndarray:
    binary = np.where(tensor == -1, 0, 1)
    return np.count_nonzero(binary, axis=0)

# total score received by country A from country B
def coincidence_total(tensor: np.ndarray) -> np.ndarray:
    zero_trim = np.where(tensor == -1, 0, tensor)
    return np.sum(zero_trim, axis=0)

def edge_average(coincidence: np.ndarray, totals: np.ndarray, countries: dict, thresh=0):
    n,_ = coincidence.shape
    res = totals / (coincidence + eps)
    dropped = []
    if thresh > 0:
        for i in range(n):
            if np.max(coincidence[i,:]) < thresh:
                res[:,i] = 0
                res[i,:] = 1
                dropped.append(countries[i])
    return res, dropped

def sorted_values(bias: np.ndarray, index: dict, inv=False):
    n, m = bias.shape
    sign = 1 if inv else -1
    l = []
    for i in range(n):
        for j in range(m):
            if i == j:
                continue
            l.append((index[j], index[i], bias[i][j]))
    return sorted(l, key=lambda x: sign * x[2])

def clear_matrix(dataset: np.ndarray, index: dict, countries: dict, dropped: list):
    ind = [index[d] for d in dropped]
    output = np.delete(dataset, [ind], axis=1)
    output = np.delete(output, [ind], axis=0) 
    for d in dropped:
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
    return row_cor, cor

def weights(cij: np.ndarray) -> np.ndarray:
    N, _ = cij.shape
    return np.where(np.identity(N) == 1, 0, 1 / (cij + eps)) * 1 / N

def get_correction_matrix(dataset: np.ndarray, index: dict, countries: dict, norm=False) -> np.ndarray:
    coincidence = coincidence_count(dataset)
    totals = coincidence_total(dataset) # raw totals - more coincident, more score! need to average
    avg, dropped = edge_average(coincidence, totals, countries, thresh=10) # mark low incidence countries for removal, add small base score constant
    cleared_avg, index, countries = clear_matrix(avg, index, countries, dropped) # remove low incidence countries
    score_probability = normalized_score(cleared_avg, alpha=0.3)
    quals, cors = corrections(score_probability)
    return quals, cors, index, countries

def write_matrix(filename, comment, matrix: np.ndarray):
    with open(filename, 'w') as  f:
        f.write(f"#{comment}\n#\n")
        for row in matrix:
            s = ["{:.4f}".format(x) for x in row]
            f.write(" ".join(s) + '\n')


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
    average_quality, correction_matrix, index, countries = get_correction_matrix(arr, index, countries)
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