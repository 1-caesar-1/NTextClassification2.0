import json

from pandas import np
from sklearn.model_selection import RandomizedSearchCV

from textclassification_app.classes.Experiment import Experiment
from enum import Enum
import pandas as pd


class ParameterTuning(Enum):
    RANDOM = 0
    GRID = 1


def parameter_tuning(experiment: Experiment, tuning_type: ParameterTuning = None):
    if tuning_type == ParameterTuning.GRID:
        grid_search(experiment)
    elif tuning_type == ParameterTuning.RANDOM:
        randomized_search(experiment)


def grid_search(experiment: Experiment):
    from sklearn.model_selection import GridSearchCV
    # Create the parameter grid based on the results of random search
    param_grid = {
        "classification__warm_start": [True, False],
        "classification__verbose": [1400, 1500, 1600],
        "classification__oob_score": [True, False],
        "classification__n_estimators": [615, 620, 625, 630, 635],
        "classification__min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3, 0.4],
        "classification__min_samples_split": [5, 10, 15, 20, 25],
        "classification__min_samples_leaf": [1, 2, 3, 4, 5],
        "classification__min_impurity_decrease": [0, 5, 10, 15, 20],
        "classification__max_leaf_nodes": [1235, 1245, 1255, 1265, 1275],
        "classification__max_features": ['auto', 'sqrt', 'log2', None],
        "classification__max_depth": [90, 95, 100, 105, 110],
        "classification__criterion": ['gini', 'entropy'],
        "classification__class_weight": [None],
        "classification__ccp_alpha": [0, 1, 2, 3, 4],
        "classification__bootstrap": [True, False]
    }
    print(param_grid)

    # for the convenience of reading
    clf = experiment.get_pipeline(experiment.classifiers[0])
    k_fold = experiment.classification_technique.k_fold
    X = experiment.documents
    y = experiment.labels

    # Instantiate the grid search model
    rf_grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=k_fold, n_jobs=-1, verbose=2)

    # Fit the random search model
    rf_grid.fit(X, y)

    print(rf_grid.best_params_)
    experiment.classifiers[0] = rf_grid.best_estimator_['classification']
    with open("best_estimator.json", "w") as file:
        json.dump(rf_grid.best_params_, file, indent=6)
    pd.DataFrame.from_dict(rf_grid.cv_results_, orient='index').to_excel("grid table.xlsx")


def randomized_search(experiment: Experiment):
    amount = 5
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=1, stop=2500, num=amount)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2', None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 200, num=amount)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [int(x) for x in np.linspace(1, 30, num=amount)]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [int(x) for x in np.linspace(1, 15, num=amount)]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    max_leaf_nodes = [int(x) for x in np.linspace(start=10, stop=2500, num=amount)]
    max_leaf_nodes.append(None)
    oob_score = [True, False]
    warm_start = [True, False]
    criterion = ['gini', 'entropy']
    min_weight_fraction_leaf = [x for x in np.linspace(start=0, stop=0.5, num=amount)]
    min_impurity_decrease = [int(x) for x in np.linspace(start=0, stop=2000, num=amount)]
    verbose = [int(x) for x in np.linspace(start=0, stop=2000, num=amount)]
    class_weight = ["balanced", "balanced_subsample", None]
    ccp_alpha = [int(x) for x in np.linspace(start=0, stop=2000, num=10)]

    # Create the random grid
    random_grid = {'classification__n_estimators': n_estimators,
                   'classification__max_features': max_features,
                   'classification__max_depth': max_depth,
                   'classification__min_samples_split': min_samples_split,
                   'classification__min_samples_leaf': min_samples_leaf,
                   'classification__max_leaf_nodes': max_leaf_nodes,
                   'classification__oob_score': oob_score,
                   'classification__warm_start': warm_start,
                   'classification__bootstrap': bootstrap,
                   'classification__criterion': criterion,
                   'classification__min_weight_fraction_leaf': min_weight_fraction_leaf,
                   'classification__min_impurity_decrease': min_impurity_decrease,
                   'classification__verbose': verbose,
                   'classification__class_weight': class_weight,
                   'classification__ccp_alpha': ccp_alpha
                   }
    print(random_grid)

    # for the convenience of reading
    clf = experiment.get_pipeline(experiment.classifiers[0])
    k_fold = experiment.classification_technique.k_fold
    X = experiment.documents
    y = experiment.labels

    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=500, cv=k_fold, verbose=2,
                                   random_state=42, n_jobs=-1)

    # Fit the random search model
    rf_random.fit(X, y)

    print(rf_random.best_params_)
    experiment.classifiers[0] = rf_random.best_estimator_['classification']
    with open("best_estimator.json", "w") as file:
        json.dump(rf_random.best_params_, file, indent=6)
    pd.DataFrame.from_dict(rf_random.cv_results_, orient='index').to_excel("random table.xlsx")
