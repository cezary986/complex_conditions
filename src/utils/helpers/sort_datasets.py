"""Skrypt sortuje zbiory danych rosnąco po ich złożoności tak by późniejsze obliczenia
zaczynać od tych najprostrzych i kończyć na najtrudniejszych
"""
import _
from typing import List
from utils.experiments_utils.helpers import datasets as dataset_helpers
import pandas as pd
import math
import os
from glob import glob
import global_config


def main():
    def dataset_accessor(dastaset_name: str) -> pd.DataFrame:
        return pd.read_parquet(
            f'{global_config.DATASETS_BASE_PATH}/{dastaset_name}/{dastaset_name}.parquet'
        )

    def calculate_df_complexity(df: pd.DataFrame) -> float:
        nominal_columns = len(df.select_dtypes('object').columns.tolist())
        numerical_columns = len(df.select_dtypes('number').columns.tolist())
        return df.shape[0] * math.pow((nominal_columns + 5 * numerical_columns), 2)

    dataset_helpers.calculate_df_complexity = calculate_df_complexity

    dataset_names: List[str] = []
    for path in glob(f'{global_config.DATASETS_BASE_PATH}/*'):
        if os.path.isdir(path):
            dataset_names.append(os.path.basename(path))

    res = dataset_helpers.sort_dataset_by_complexity(
        dataset_names, dataset_accessor)

    print('[')
    for e in res:
        print(f"    '{e['name']}',")
    print(']')


if __name__ == '__main__':
    main()
