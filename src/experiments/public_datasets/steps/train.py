import _
from utils.experiments_utils import *
from utils.helpers.datasets import Dataset
from utils.rulekit.classification import RuleClassifier


class TrainedModelsResults:

    def __init__(self) -> None:
        self.original: List[RuleClassifier] = []
        self.k_bins: List[RuleClassifier] = []
        self.entropy_mdl: List[RuleClassifier] = []
        self.rulekit_supplemented: List[RuleClassifier] = []
    

def train_models(rulekit_params: dict, dataset_name: str, logger: Logger) -> TrainedModelsResults:
    results = TrainedModelsResults()
    dataset = Dataset(dataset_name)
    
    # cross validation
    folds = dataset.get_cv() if dataset.get_cv() is not None else [dataset.get_train_test()]
    if len(folds) == 1:
        logger.info(f'Dataset "{dataset_name}" has not CV files, train_test will be used')
    for i, fold in enumerate(folds):
        logger.info(f'   train for fold: {i}')
        X_train, y_train = fold[0:2]
        clf = RuleClassifier(**rulekit_params)
        clf.fit(X_train, y_train)
        results.original.append(clf)
    
    return results



@step()
def train_plain_models(rulekit_params: dict, dataset_name: str) -> TrainedModelsResults:
    return train_models(rulekit_params, dataset_name, train_plain_models.logger)

@step()
def train_models_complex_conditions(rulekit_params: dict, dataset_name: str) -> TrainedModelsResults:
    return train_models(rulekit_params, dataset_name, train_models_complex_conditions.logger)

@step()
def train_models_inner_alternatives(rulekit_params: dict, dataset_name: str) -> TrainedModelsResults:
    return train_models(rulekit_params, dataset_name, train_models_inner_alternatives.logger)

@step()
def train_models_complex_conditions_and_alternatives(rulekit_params: dict, dataset_name: str) -> TrainedModelsResults:
    return train_models(rulekit_params, dataset_name, train_models_complex_conditions_and_alternatives.logger)
