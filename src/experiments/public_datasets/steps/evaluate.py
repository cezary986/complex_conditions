import _
import config
from logging import Logger
from sklearn.metrics import *
from utils.experiments_utils import *
from utils.experiments_utils.results.tables import *
from utils.helpers.datasets import Dataset
from utils.rulekit.classification import RuleClassifier
from steps.train import TrainedModelsResults


def reduce_mean(df: pd.DataFrame) -> pd.DataFrame:
    numerical_columns = df.select_dtypes('number').columns.tolist()
    data = {}
    for column in df.columns.tolist():
        if column in numerical_columns and '(std)' not in column:
            data[column] = [df[column].mean()]
            data[f'{column} (std)'] = [df[column].std()]
        else:
            data[column] = [df[column].iloc[0]]
    return pd.DataFrame(data)


def write_rules(
    results_dir: str,
    models: TrainedModelsResults
):
    models = getattr(models, 'original')
    for i, model in enumerate(models):
        results_cv_dir = f'{results_dir}/original/cv/{i + 1}'
        os.makedirs(results_cv_dir, exist_ok=True)
        rule_file_path = f'{results_cv_dir}/rules.txt'
        with open(rule_file_path, 'w+') as file:
            for rule in model.model.rules:
                file.write(
                    f'{str(rule)} (p={int(rule.weighted_p)}, n={int(rule.weighted_n)}, P={int(rule.weighted_P)}, N={int(rule.weighted_N)})\n')


def evaluate_models(
    dataset: Dataset,
    variant_name: str,
    dataset_type: str,
    models: List[RuleClassifier],
    logger: Logger
) -> None:
    logger.info(
        f'Evalute model for dataset: "{dataset.name}" ({dataset_type}) for variant: "{variant_name}"')
    metrics: Table = Tables.get(dataset.name, variant_name, dataset_type, 'metrics')
    conditions_stats: Table = Tables.get(
        dataset.name, variant_name, dataset_type, 'conditions_stats')
    confusion_matrix_train_avg = None
    confusion_matrix_test_avg = None

    for i, model in enumerate(models):
        if len(models) == 1:
            logger.info(f'Evaluate on train_test')
            X_train, y_train, X_test, y_test = dataset.get_train_test()
        else:
            logger.info(f'Evaluate for fold: {i + 1}')
            X_train, y_train, X_test, y_test = dataset.get_cv_fold(i + 1)

        fold_metrics: Table = Tables.get(
            dataset.name, variant_name, dataset_type, 'cv', str(i + 1), 'metrics')

        prediction_test = model.predict(X_test)
        prediction_train = model.predict(X_train)

        fold_metrics.rows.append({
            'dataset': dataset.name,
            'variant': variant_name,
            'dataset type': dataset_type,

            'BAcc (test)': balanced_accuracy_score(y_test, prediction_test),
            'BAcc (train)': balanced_accuracy_score(y_train, prediction_train),
            'Acc (test)': accuracy_score(y_test, prediction_test),
            'Acc (train)': accuracy_score(y_train, prediction_train),
            'rules':  sum([
                value if 'statistics' not in key else 0 for key, value in model.model.stats.conditions_stats.stats.items()
            ]), # change conditiosn counting so that alternatives are counted as one conditions
            'conditions_count': model.model.stats.rules_count * model.model.stats.conditions_per_rule,
            'avg conditions per rule': model.model.stats.conditions_per_rule,
            'avg rule quality': model.model.stats.avg_rule_quality,
            'avg rule precision': model.model.stats.avg_rule_precision,
            'avg rule coverage': model.model.stats.avg_rule_coverage,
            'training time total (s)': model.model.stats.time_total_s,
            'training time growing (s)': model.model.stats.time_growing_s,
            'training time pruning (s)': model.model.stats.time_pruning_s,

            'induction measure': model.get_params()['induction_measure'].replace('Measures.', ''),
            'pruning measure': model.get_params()['pruning_measure'].replace('Measures.', ''),
            'voting measure': model.get_params()['voting_measure'].replace('Measures.', ''),
        })
        fold_metrics.save()

        labels_values = Dataset(dataset.name).get_full()[1].unique().tolist()
        cm = confusion_matrix(
            y_train, prediction_train, labels=labels_values)
        if confusion_matrix_train_avg is None:
            confusion_matrix_train_avg = cm
        else:
            confusion_matrix_train_avg += cm
        train_confusion_matrix = pd.DataFrame(
            cm,
            index=[f'true:{value}' for value in labels_values],
            columns=[f'pred:{value}' for value in labels_values]
        )
        train_confusion_matrix.to_csv(
            f'{os.path.dirname(fold_metrics._file_path)}/confusion_matrix_train.csv')

        cm = confusion_matrix(
            y_test, prediction_test, labels=labels_values)
        if confusion_matrix_test_avg is None:
            confusion_matrix_test_avg = cm
        else:
            confusion_matrix_test_avg += cm
        test_confusion_matrix = pd.DataFrame(
            cm,
            index=[f'true:{value}' for value in labels_values],
            columns=[f'pred:{value}' for value in labels_values]
        )
        test_confusion_matrix.to_csv(
            f'{os.path.dirname(fold_metrics._file_path)}/confusion_matrix_test.csv')

        fold_conditions_stats: Table = Tables.get(
            dataset.name, variant_name, dataset_type, 'cv', str(i + 1), 'conditions_stats')

        conditions_stats_dict: dict = model.model.stats.conditions_stats.stats
        conditions_stats_dict['dataset'] = dataset.name
        conditions_stats_dict['variant'] = variant_name
        conditions_stats_dict['dataset type'] = dataset_type
        tmp = conditions_stats_dict
        if 'Inner alternatives statistics' in conditions_stats_dict:
            for key, value in conditions_stats_dict['Inner alternatives statistics'].items():
                tmp[f'Inner alternatives - {key}'] = value
            del conditions_stats_dict['Inner alternatives statistics']
        fold_conditions_stats.rows.append(conditions_stats_dict)
        fold_conditions_stats.save()

        metrics.rows += fold_metrics.rows
        conditions_stats.rows += fold_conditions_stats.rows

    confusion_matrix_train_avg = confusion_matrix_train_avg / len(models)
    confusion_matrix_train_avg = pd.DataFrame(
        confusion_matrix_train_avg,
        index=[f'true:{value}' for value in labels_values],
        columns=[f'pred:{value}' for value in labels_values]
    )
    confusion_matrix_train_avg.to_csv(
        f'{os.path.dirname(metrics._file_path)}/confusion_matrix_train.csv')

    confusion_matrix_test_avg = confusion_matrix_test_avg / len(models)
    confusion_matrix_test_avg = pd.DataFrame(
        confusion_matrix_test_avg,
        index=[f'true:{value}' for value in labels_values],
        columns=[f'pred:{value}' for value in labels_values]
    )
    confusion_matrix_test_avg.to_csv(
        f'{os.path.dirname(metrics._file_path)}/confusion_matrix_test.csv')

    metrics.set_df(reduce_mean(metrics.as_pandas()))
    conditions_stats.set_df(reduce_mean(conditions_stats.as_pandas()))


def evaluate_all_models(
    dataset_name: str,
    variant_name: str,
    models: TrainedModelsResults,
    logger: Logger
) -> None:
    logger.info(
        f'Evalute model for dataset: "{dataset_name}" for variant: "{variant_name}"')
    Tables.configure(directory=config.RESULTS_BASE_PATH)

    logger.info('Write model rules')
    write_rules(
        f'{config.RESULTS_BASE_PATH}/{dataset_name}/{variant_name}', models)
    if len(models.original) > 0:
        dataset = Dataset(dataset_name)
        evaluate_models(dataset, variant_name, 'original',
                        models.original, logger)


@step()
def evaluate_plain_model(
    dataset_name: str,
    variant_name: str,
    models: TrainedModelsResults,
):
    evaluate_all_models(dataset_name, variant_name, models,
                        evaluate_plain_model.logger)


@step()
def evaluate_models_inner_alternatives(
    dataset_name: str,
    variant_name: str,
    models: TrainedModelsResults,
):
    evaluate_all_models(dataset_name, variant_name, models,
                        evaluate_models_inner_alternatives.logger)


@step()
def evaluate_models_complex_conditions(
    dataset_name: str,
    variant_name: str,
    models: TrainedModelsResults,
):
    evaluate_all_models(dataset_name, variant_name, models,
                        evaluate_models_complex_conditions.logger)


@step()
def evaluate_models_complex_conditions_and_alternatives(
    dataset_name: str,
    variant_name: str,
    models: TrainedModelsResults,
):
    evaluate_all_models(dataset_name, variant_name, models,
                        evaluate_models_complex_conditions_and_alternatives.logger)
