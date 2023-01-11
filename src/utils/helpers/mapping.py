from __future__ import annotations
from typing import Dict, List, TypedDict
import json
import numpy as np
import pandas as pd


class _BinsMappingSerialized(TypedDict):
    closed: bool
    cut_points: Dict[str, List[float]]


class BinsMapping():
    """Klasa przechowująca mapowania przedziałów dyskretyzacji
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cut_points: np.ndarray,
        closed: str = 'right'
    ) -> None:
        """
        Args:
            df (pd.DataFrame): dataframe
            cut_points (np.ndarray): tablica z punktami cięć kolejne wiersze tablicy to tablice punktów
                cięć dla kolejnych kolumn
            closed (str, optional): określa z której strony zamknięte są przedziały mapowania utworzone z punktów
                cięć. Domyślnie przedziały zamykane są z prawej strony
        """
        self.closed: str = closed
        self.columns: List[str] = df.select_dtypes('number').columns.tolist()
        self.cut_points: np.ndarray = cut_points

    def save(self, path: str):
        with open(path, 'w+') as file:
            json.dump({
                'closed': self.closed,
                'cut_points': {
                    column: self.cut_points[i].tolist() for i, column in enumerate(self.columns)
                }
            }, file, indent=2)

    @staticmethod
    def from_file(path: str) -> BinsMapping:
        with open(path, 'r') as file:
            data: _BinsMappingSerialized = json.load(file)
            cut_points: dict = data['cut_points']
            mapping = BinsMapping(
                columns=list(cut_points.keys()),
                cut_points=np.array(list(cut_points.values())),
                closed=data['closed']
            )
            return mapping
