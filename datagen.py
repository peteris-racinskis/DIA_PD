from pandas import read_csv, DataFrame
import numpy as np
from typing import List
INFILE="votes.csv"
STARTYEAR=1980
STOPYEAR=2015
YEAR="year"
ROUND="round"
CNT_RCV="to_country_id"
CNT_SND="from_country_id"
FIN="final"
SF1="semi-final-1"
SF2="semi-final-2"
SCR="total_points"

def pack(value, len=2) -> tuple:
    if not isinstance(value, tuple):
        return (x for x in [value] * len)
    return value

def select_years(df: DataFrame, year) -> DataFrame:
    minyear, maxyear = pack(year)
    return df.loc[lambda d: (d[YEAR] >= minyear) & (d[YEAR] <= maxyear), :]

def select_rounds(df: DataFrame, round) -> DataFrame:
    rounds = [x for x in pack(round, 3)]
    return df.loc[lambda d: 
          (d[ROUND] == rounds[0]) 
        | (d[ROUND] == rounds[1]) 
        | (d[ROUND] == rounds[2]), :]

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
    return scores

def get_stage_matrices(df: DataFrame, minyear, maxyear) -> List[np.ndarray]:
    limyears = (minyear, maxyear)
    rounds = (FIN, SF1, SF2)
    trimmed = select_years(df, limyears)
    years = trimmed[YEAR].unique()
    all_participants = get_participants(trimmed, limyears, rounds)
    totalsize = len(all_participants)
    index = index_codes(all_participants)
    matrices = []
    for year in years:
        for rnd in rounds:
            matrices.append(get_stage_matrix(df, year, rnd, totalsize, index))
    return matrices


if __name__ == "__main__":
    df = read_csv(INFILE)
    trimmed = select_years(df, (STARTYEAR, STOPYEAR))
    part = get_participants(trimmed, 2015, FIN)
    stg = get_stage(trimmed, STOPYEAR, FIN)
    scr = get_score(stg, "lv", "ru")
    sg1 = get_stage_matrices(df, 2015, 2015)
    print()