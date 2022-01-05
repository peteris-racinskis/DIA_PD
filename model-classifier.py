from numpy.ma.core import count
from pandas import DataFrame, read_csv
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import LinearSVC
from vectorize import OUTFILE as INFILE, vec
from model import sorted_values, write_matrix, distance_measure, normalized_score, corrections
VECTORIZED=True
LOAD=True
COUNTRIES=43
SCORECOUNT=-11 if VECTORIZED else -1
INFILE=INFILE + ("-vec.csv" if VECTORIZED else "-cat.csv")

def score_expectation(c1, c2, model: MLPClassifier, index: dict = None):
    if index is None:
        index = {v:v for v in range(COUNTRIES*2)}
    c1_v = vec(index, c1, False)
    c2_v = vec(index, c2, False)
    argument = (c1_v + c2_v).reshape(1,-1)
    scores = np.arange(11)
    probs = model.predict_proba(argument).flatten()
    return probs @ scores

def score_distribution(c1, c2, model: MLPClassifier, index: dict = None):
    if index is None:
        index = {v:v for v in range(COUNTRIES*2)}
    c1_v = vec(index, c1, False)
    c2_v = vec(index, c2, False)
    argument = (c1_v + c2_v).reshape(1,-1)
    probs = model.predict_proba(argument).flatten()
    return probs

def country_index(l, inv=False) -> dict:
    d = {}
    i = 0
    for c in l[:COUNTRIES*2]:
        d[c] = i
        i+=1
    return d if not inv else {v:k for k,v in d.items()}

def prediction_matrix(clf):
    mat = np.zeros((COUNTRIES, COUNTRIES))
    for i in range(COUNTRIES):
        for j in range(COUNTRIES):
            mat[i][j] = score_expectation(i, j+COUNTRIES, clf)
    return mat

def distribution_tensor(clf):
    mat = np.zeros((COUNTRIES, COUNTRIES, 11))
    for i in range(COUNTRIES):
        for j in range(COUNTRIES):
            mat[i][j] = score_distribution(i, j+COUNTRIES, clf)
    return mat

def correction_matrix(pred: np.ndarray):
    score_prob = normalized_score(pred, 0.01)
    return corrections(score_prob)


if __name__ == "__main__":
    df = read_csv(INFILE)
    labels = df.iloc[:,SCORECOUNT:].to_numpy(dtype=np.float64)
    data = df.iloc[:,:SCORECOUNT].to_numpy(dtype=np.float64)
    ind = country_index(list(df.columns))
    if not LOAD:
        clf = MLPClassifier(hidden_layer_sizes=(400,100), verbose=4)
        #clf = CategoricalNB()
        #clf = LinearSVC()
        clf.fit(data, labels)
        with open("dumpfile", "wb") as f:
            pickle.dump(clf, f)
    else:
        with open("dumpfile", "rb") as f:
            clf = pickle.load(f)
    clf.out_activation_ = "softmax"
    predictions = prediction_matrix(clf)
    cor = correction_matrix(predictions)
    distances = distance_measure(cor)
    one_axis = country_index(list(df.columns)[:COUNTRIES], True)
    pred_list = sorted_values(predictions, one_axis)
    d_list = sorted_values(distances, one_axis, True)

    distributions = distribution_tensor(clf)

    namefile="classifier-names.txt"
    distfile="classifier-distances.txt"
    predfile="classifier-predictions.txt"
    with open(namefile, 'w') as f:
        [f.write(f"{y}\n") for x,y in one_axis.items()]
    write_matrix(distfile, 
    "Distances: inverse of score expected value, symmetric", 
    distances)
    write_matrix(predfile, 
    "Expected values of the score as per the classifier output distribution", 
    predictions)
    
    print("10 closest distances:")
    for i in range(10):
        print(d_list[i*2])
    print("10 farthest distances:")
    for i in range(10):
        print(d_list[len(pred_list)-1-i*2])
    print("Distances to Latvia:")
    [print(x) for x in filter(lambda x: x[0] == "lv", d_list)]