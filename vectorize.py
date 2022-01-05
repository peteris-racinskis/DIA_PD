from pandas import read_csv, DataFrame
import numpy as np
INFILE="votes.csv"
STARTYEAR=1980
STOPYEAR=2015
OUTFILE=f"Vectorized-{STARTYEAR}-{STOPYEAR}"
YEAR="year"
ROUND="round"
CNT_RCV="to_country_id"
CNT_SND="from_country_id"
SCR="total_points"
VECTOR_OUTPUT=True
DROP=['ma', 'sk', 'cs', 'ad', 'mc', 'me', 'cz', 'sm', 'au']

def country_index(df: DataFrame) -> dict:
    codes = df[CNT_SND].unique()
    d = {}
    i = 0
    for code in codes:
        d[code] = i
        i+=1
    return d

def score_index() -> dict:
    d = {}
    base = range(9)
    for i in base:
        d[i] = i
    d[10] = 9
    d[12] = 10
    return d

def select_years(df: DataFrame, minyear, maxyear) -> DataFrame:
    return df.loc[lambda d: (d[YEAR] >= minyear) & (d[YEAR] <= maxyear), :]

def filter_values(df: DataFrame, values: list, index: dict) -> DataFrame:
    ind = index.copy()
    for v in values:
        ind.pop(v)
    df = df[df[CNT_RCV].isin(ind.keys())]
    df = df[df[CNT_SND].isin(ind.keys())]
    new_index_base = sorted([(x,y) for x,y in ind.items()], key=lambda z: z[1])
    ind = {}
    for i in range(len(new_index_base)):
        ind[i] = new_index_base[i][0]
    inv = {v:k for k,v in ind.items()}
    return df, ind, inv

def vec(index, value, st=True):
    vector = np.zeros(len(index))
    vector[index[value]] = 1
    return  ",".join([str(v) for v in vector]) if st else vector

def vectorize(df: DataFrame, index: dict, score: dict, filename: str):
    i = 0
    with open(filename, 'w') as f:
        inv = {v:k for k,v in index.items()}
        clist = ",".join([inv[i] for i in range(len(inv))])
        scorelist = ",".join([str(x) for x in [i for i in range(9)] + [10,12]])
        scorelist = "scr" if not VECTOR_OUTPUT else scorelist
        f.write(f"{clist},{clist},{scorelist}\n")
        for row in df.iloc:
            i+=1
            snd = vec(index, row[CNT_SND])
            rcv = vec(index, row[CNT_RCV])
            scr = vec(score, row[SCR]) if VECTOR_OUTPUT else str(row[SCR])
            f.write(f"{snd},{rcv},{scr}\n")
            if i % 2000 == 0:
                print(f"Row {i}")


if __name__ == "__main__":
    df = select_years(read_csv(INFILE), STARTYEAR, STOPYEAR)
    index = country_index(df)
    df, ind, inv = filter_values(df, DROP, index)
    score = score_index()
    outfile = OUTFILE + ("-vec.csv" if VECTOR_OUTPUT else "-cat.csv")
    vectorize(df, inv, score, outfile)
    print()