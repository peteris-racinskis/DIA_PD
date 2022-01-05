from pandas import read_csv, DataFrame
import numpy as np
INFILE="votes.csv"
STARTYEAR=1980
STOPYEAR=2015
OUTFILE=f"Vectorized-{STARTYEAR}-{STOPYEAR}.csv"
YEAR="year"
ROUND="round"
CNT_RCV="to_country_id"
CNT_SND="from_country_id"
SCR="total_points"
VECTOR_OUTPUT=True

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

def vec(index, value) -> str:
    vector = [0] * len(index)
    vector[index[value]] = 1
    return  ",".join([str(v) for v in vector])

def vectorize(df: DataFrame, index: dict, score: dict, filename: str):
    i = 0
    with open(filename, 'w') as f:
        for row in df.iloc:
            i+=1
            snd = vec(index, row[CNT_SND])
            rcv = vec(index, row[CNT_RCV])
            scr = vec(score, row[SCR]) if VECTOR_OUTPUT else str(scr)
            f.write(f"{snd},{rcv},{scr}\n")
            if i % 2000 == 0:
                print(f"Row {i}")

if __name__ == "__main__":
    df = select_years(read_csv(INFILE), STARTYEAR, STOPYEAR)
    df_sub = df[[CNT_SND, CNT_RCV, SCR]]
    index = country_index(df)
    score = score_index()
    vectorize(df, index, score, OUTFILE)
    print()