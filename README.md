## Results structure

All collected results are stored in the following directory: `src\experiments\public_datasets\results\4.0.1\C2`. The directory structure is as follows:
```
results\4.0.1\C2/
├─ dataset_1/
│  ├─ plain/
│  │  ├─ original/
│  │  │  ├─ cv/
│  │  │  ├─ conditions_stats.csv
│  │  │  ├─ confusion_matrix_test.csv
│  │  │  ├─ confusion_matrix_train.csv
│  │  │  ├─ metrics.csv
│  ├─ complex_conditions/
│  ├─ complex_conditions_inner_alternatives/
|  ...
├─ dataset_N/
├─ metrics.csv
```
The root directory contains `metrics.csv` file with all datasets results suitable for easy analysis (produced by `./src/experiments/public_datasets/results_analysis/main_v4.0.1.ipynb` notebook). It also contains directories with the results of every evaluated dataset. Those folders contain three subdirectories for every tested variant:
* plain - original RuleKit implementations with no complex conditions
* complex_conditions - complex conditions but with disabled Alternatives conditions
* complex_conditions_inner_alternatives - all complex conditions enabled (all)

In those directories there is `original` subdirectory containing four csv files:
* `conditions_stats.csv` - statistics of how many different types of conditions were present in rule sets
* `confusion_matrix_test.csv` - confusion matrix for test dataset
* `confusion_matrix_train.csv` - confusion matrix for train dataset
* `metrics.csv` - average results from all cross validations folds

Additionally, results from every cross-validation are present in `cv` directory for all variants.


## Building your own RuleKit jar file (optional)
The first step is building a custom RuleKit jar version utilizing complex conditions. This jar file is later used by Python wrapper to 
conduct experiments. To build a jar file one should execute the following command in `./src/RuleKit/adaa.analytics.rules` directory.
    ```bash
    ./gradlew rjar
    ```

The jar file will be generated to the following location: `./src/RuleKit/adaa.analytics.rules/adaa.analytics.rules/build/libs`. Copy this file to: `./src/utils/rulekit/jar`.



## Running experiments

### Running experiments on synthetic datasets

To run experiments on synthetic datasets use notebooks in: `src\experiments\synthetic_datasets`.

### Running experiments on public datasets

Those experiments are more complicated and quite time-consuming to run. That is why the [experiments_utils](https://github.com/cezary986/experiments_utils/tree/master/experiments_utils) package was used to write and run them. It automatically runs experiments on different datasets using a multiprocessing pool, while giving detailed logs and reproducibility.

To run an experiment on the public datasets run the following command in `./src/experiments/public_datasets` directory:
    ```bash
    python main.py
    ```

Experiments configuration and which dataset should be evaluated could be modified using `./src/experiments/public_datasets/config.py` file.