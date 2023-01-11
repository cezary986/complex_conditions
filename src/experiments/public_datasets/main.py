import config
from utils.experiments_utils import *
from utils.experiments_utils.results.tables import *
from utils.rulekit import RuleKit
from utils.rulekit import __version__ as rulekit_wrapper_version
from steps.train import *
from steps.evaluate import *


@experiment(
    name='complex_conditions',
    version=config.VERSION,
    paramsets=config.PARAMSETS,
    n_jobs=8
)
def main(
    dataset_name: str,
    variants: dict
):
    s = Store()

    RuleKit.init(
        initial_heap_size=config.RULEKIT_INITIAL_HEAP_SIZE,
        max_heap_size=config.RULEKIT_MAX_HEAP_SIZE,
    )
    main._logger.debug(f'Run using RuleKit jar in version: {RuleKit.version} and wrapper in version: {rulekit_wrapper_version}')

    # Train and evaluate original RuleKit
    s.clf_plain = train_plain_models(
        variants['plain'], dataset_name)
    evaluate_plain_model(dataset_name, 'plain', s.clf_plain)

    # Train and evaluate RuleKit with complex conditions - without alternatives conditions
    s.clf_complex = train_models_complex_conditions(
        variants['complex_conditions'], dataset_name)
    try:
        evaluate_models_complex_conditions(dataset_name, 'complex_conditions', s.clf_complex)
    except NameError:
        pass

    # Train and evaluate RuleKit with all complex conditions enabled
    s.clf_all = train_models_complex_conditions_and_alternatives(
        variants['complex_conditions_inner_alternatives'], dataset_name)
    try:
        evaluate_models_complex_conditions_and_alternatives(dataset_name, 'complex_conditions_inner_alternatives', s.clf_all)
    except NameError:
        pass

if __name__ == '__main__':
    main()
