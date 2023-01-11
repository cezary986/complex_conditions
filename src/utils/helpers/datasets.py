from __future__ import annotations
import os
import _
from typing import List, Tuple
import pandas as pd
import numpy as np
from glob import glob
import sys, os

dir_path = os.path.dirname(os.path.realpath(__file__))

DATASETS_BASE_PATH = f'{dir_path}/../../../datasets'


class Dataset:
    """Helper class for fetching datasets
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.path: str = f'{DATASETS_BASE_PATH}/{name}'

        self._full: Tuple[pd.DataFrame, pd.Series] = None
        self._train: Tuple[pd.DataFrame, pd.Series] = None
        self._test: Tuple[pd.DataFrame, pd.Series] = None
        self._cv: List[Tuple[pd.DataFrame, pd.Series]] = None
        self._cv_folds_count: int = None

    def get_full(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self._full is None:
            self._full = pd.read_parquet(f'{self.path}/{self.name}.parquet')
        return self._get_X_y(self._full)

    def get_cv(self) -> List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        if self._cv is None:
            self._cv = [self.get_cv_fold(i)
                        for i in range(1, self.cv_folds_count + 1)]
        if len(self._cv) == 0 or self._cv[0] is None:
            return None
        else:
            return self._cv

    def get_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self._train is None:
            self._train = self._get_X_y(pd.read_parquet(
                f'{self.path}/train_test/train.parquet'))
        return self._train

    def get_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self._test is None:
            self._test = self._get_X_y(pd.read_parquet(
                f'{self.path}/train_test/test.parquet'))
        return self._test

    def get_train_test(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        X_train, y_train = self.get_train()
        X_test, y_test = self.get_test()
        return X_train, y_train, X_test, y_test

    @property
    def cv_folds_count(self) -> int:
        return len(glob(f'{self.path}/cv/*'))

    def was_discretized(self) -> bool:
        return self.k_bins is not None

    def get_cv_fold(self, index: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        if not os.path.exists(f'{self.path}/cv'):
            return None
        X_train, y_train = self._get_X_y(pd.read_parquet(f'{self.path}/cv/{index}/train.parquet'))
        X_test, y_test = self._get_X_y(pd.read_parquet(f'{self.path}/cv/{index}/test.parquet'))
        return X_train, y_train, X_test, y_test

    def _get_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        y = df['class']
        X = df.drop('class', 1)
        all_nans_columns = X.columns[X.isna().all()].tolist()
        X.loc[:, all_nans_columns].fillna(value=np.nan, inplace=True)
        return X.dropna(axis=1, how='all'), y


    @property
    def k_bins(self) -> Dataset:
        """Get K-bins discretized dataset 

        Returns:
            Dataset: discretized dataset 
        """
        path = f'{DATASETS_BASE_PATH}/{self.name}/5_bins'
        if os.path.exists(path):
            discretized_dataset = Dataset(self.name)
            discretized_dataset.path = path
            return discretized_dataset
        else:
            return None

    @property
    def entropy_mdl(self) -> Dataset:
        """Get entropy MDL discretized dataset 

        Returns:
            Dataset: discretized dataset 
        """
        path = f'{DATASETS_BASE_PATH}/{self.name}/entropy_mdl'
        if os.path.exists(path):
            discretized_dataset = Dataset(self.name)
            discretized_dataset.path = path
            return discretized_dataset
        else:
            return None


    @property
    def rulekit_supplemented(self) -> Dataset:
        """Get dataset discretized by rulekit (supplemented) 

        Returns:
            Dataset: discretized dataset 
        """
        path = f'{DATASETS_BASE_PATH}/{self.name}/rulekit_supplemented'
        if os.path.exists(path):
            discretized_dataset = Dataset(self.name)
            discretized_dataset.path = path
            return discretized_dataset
        else:
            return None

if __name__ == '__main__':
    import time
    dataset = Dataset('diabetes')

    start_time = time.time()
    X_train, y_train = dataset.get_train()
    X_test, y_test = dataset.get_test()
    folds = dataset.get_cv()
    print(folds)
