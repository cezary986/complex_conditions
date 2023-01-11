from _ import *
from typing import Dict, List
from utils.rulekit.params import Measures


# JVM initial heap size and max heap size - could be decreased probably
RULEKIT_INITIAL_HEAP_SIZE: int = 20480  # (mb) = 20gb
RULEKIT_MAX_HEAP_SIZE: int = 81920  # (mb) = 80gb

VERSION: str = '4.0.1'

# main directory with datasets
DATASETS_BASE_PATH: str = f'{dir_path}/../../../datasets'

# rule induction measure used for growing, pruning and voting
MEASURE = Measures.C2

# main directory with results
RESULTS_BASE_PATH: str = f'{dir_path}/results/{VERSION}/{str(MEASURE).replace("Measures.", "")}'

# lista zbiorów danych posortowana od majmniej do najbardziej złożónych
DATASETS: List[str] = [
    'anneal',
    'auto-mpg',
    'autos',
    'balance-scale',
    'bupa-liver-disorders',
    'car',
    'cleveland',
    'credit-a',
    'cylinder-bands',
    'echocardiogram',
    'ecoli',
    'flag',
    'glass',
    'hayes-roth',
    'heart-c',
    'heart-statlog',
    'hepatitis',
    'horse-colic',
    'hungarian-heart-disease',
    'iris',
    'lymphography',
    'monk-1',
    'monk-2',
    'monk-3',
    'mushroom',
    'nursery',
    'soybean',
    'tic-tac-toe',
    'titanic',
    'vote',
    'wine',
    'zoo'
]


# bazowe parametry rulekit - wspólne dla wszystkich wywołań
BASE_RULEKIT_PARAMS: dict = {
    "min_rule_covered": 5,
    "induction_measure": MEASURE,
    "pruning_measure": MEASURE,
    "voting_measure": MEASURE,
    "max_growing": 0.0,
    "enable_pruning": True,
    "ignore_missing": False,
    "max_uncovered_fraction": 0.0,
    "select_best_candidate": False,

    "inner_alternatives_search_beam_size": 3,
    "inner_alternatives_max_search_iterations": 5
}


VARIANTS: Dict[str, dict] = {
    # original rulekit all complex conditions disabled
    'plain': {
        **BASE_RULEKIT_PARAMS,
        "discrete_set_conditions_enabled": False,
        "negated_conditions_enabled": False,
        "intervals_conditions_enabled": False,
        "numerical_attributes_conditions_enabled": False,
        "nominal_attributes_conditions_enabled": False,
        "inner_alternatives_enabled": False,
    },
    # complex conditions enabled - alternatives conditions disabled
    'complex_conditions': {
        **BASE_RULEKIT_PARAMS,
        "discrete_set_conditions_enabled": True,
        "negated_conditions_enabled": True,
        "intervals_conditions_enabled": True,
        "numerical_attributes_conditions_enabled": True,
        "nominal_attributes_conditions_enabled": True,
        "inner_alternatives_enabled": False,
    },
    # all complex conditions enabled
    'complex_conditions_inner_alternatives': {
        **BASE_RULEKIT_PARAMS,
        "discrete_set_conditions_enabled": True,
        "negated_conditions_enabled": True,
        "intervals_conditions_enabled": True,
        "numerical_attributes_conditions_enabled": True,
        "nominal_attributes_conditions_enabled": True,
        "inner_alternatives_enabled": True,
    },
}

PARAMSETS: Dict[str, dict] = {
    dataset_name: {
        'dataset_name': dataset_name,
        'variants': VARIANTS
    } for dataset_name in DATASETS
}
