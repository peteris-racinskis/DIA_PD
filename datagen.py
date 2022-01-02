from pandas import read_csv, DataFrame
import numpy as np
from typing import List
INFILE="votes.csv"
STARTYEAR=1980
STOPYEAR=2015
OUTFILE=f"ScoreTensor-{STARTYEAR}-{STOPYEAR}"
YEAR="year"
ROUND="round"
CNT_RCV="to_country_id"
CNT_SND="from_country_id"
SCR="total_points"

# stolen from stackoverflow because I'm lazy.
# https://stackoverflow.com/a/37630397
def progress_bar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

def pack(value, len=2) -> tuple:
    if not isinstance(value, tuple):
        return (x for x in [value] * len)
    return value

def select_years(df: DataFrame, year) -> DataFrame:
    minyear, maxyear = pack(year)
    return df.loc[lambda d: (d[YEAR] >= minyear) & (d[YEAR] <= maxyear), :]

# This will break if you try to give 2 rounds in  tuple.
def select_rounds(df: DataFrame, round) -> DataFrame:
    rounds = [x for x in pack(round, 4)]
    return df.loc[lambda d: 
          (d[ROUND] == rounds[0]) 
        | (d[ROUND] == rounds[1]) 
        | (d[ROUND] == rounds[2])
        | (d[ROUND] == rounds[3]), :]

def get_stage(df: DataFrame, year, round) -> DataFrame:
    ds_years = select_years(df, year)
    ds_rounds = select_rounds(ds_years, round)
    return ds_rounds

def get_participants(df: DataFrame, year, round):
    stage_data = get_stage(df, year, round)
    return stage_data[CNT_RCV].unique()

def get_voters(df: DataFrame, year, round):
    stage_data = get_stage(df, year, round)
    return stage_data[CNT_SND].unique()

def get_score(df: DataFrame, fr, to):
    row = df.loc[lambda d: (d[CNT_SND] == fr) & (d[CNT_RCV] == to)]
    if not len(row) == 1:
        raise ValueError("Score row not unique! Only use on single stages.")
    return row[SCR]

def index_codes(codes) -> dict:
    d = {}
    i = 0
    for code in codes:
        d[code] = i
        i+=1
    return d

def get_stage_matrix(df: DataFrame, year, round, totalsize, index: dict) -> np.ndarray:
    stg = get_stage(df, year, round)
    participants = get_participants(df, year, round)
    voters = get_voters(df, year, round)
    scores = np.full((totalsize,totalsize), -1)
    for p in participants:
        for v in voters:
            i, j = index[p], index[v]
            scores[i,j] = get_score(stg, v, p)
    return normalize(scores)

def normalize(arr: np.ndarray) -> np.ndarray:
    s1 = np.where(arr == 10, 9, arr)
    s2 = np.where(s1 == 12, 10, s1)
    return s2

def round_names(df: DataFrame, year):    
    return select_years(df, year)[ROUND].unique()

def get_stage_matrices(df: DataFrame, minyear, maxyear) -> List[np.ndarray]:
    limyears = (minyear, maxyear)
    trimmed = select_years(df, limyears)
    years = trimmed[YEAR].unique()
    rounds = tuple(trimmed[ROUND].unique())
    all_participants = get_participants(trimmed, limyears, rounds)
    totalsize = len(all_participants)
    index = index_codes(all_participants)
    matrices = []
    steps = len(years)
    i = 0
    instances = []
    for year in years:
        progress_bar(i,steps)
        i+=1
        rounds = round_names(trimmed, year)
        for rnd in rounds:
            matrices.append(get_stage_matrix(df, year, rnd, totalsize, index))
            instances.append((year, rnd))
    progress_bar(i,steps)
    print()
    return matrices, instances


if __name__ == "__main__":
    df = read_csv(INFILE)
    print(f"Reading dataset from: {INFILE}")
    print("Processing ... ")
    stage_matrices, stages = get_stage_matrices(df, STARTYEAR, STOPYEAR)
    score_dataset = np.stack(stage_matrices)
    datashape = score_dataset.shape
    shapefile = OUTFILE + "-shape.txt"
    stagefile = OUTFILE + "-stages.txt"
    datafile = OUTFILE + "-data.bin"
    with open(shapefile, 'w') as f:
        f.write(str(datashape))
    with open(stagefile, 'w') as f:
        i = 0
        for x in stages:
            f.write(f"{i}\t{x}\n")
            i += 1
    score_dataset.tofile(datafile)
    print(f"Preprocessed dataset shape in: {shapefile}")
    print(f"Preprocessed dataset contents in: {datafile}")