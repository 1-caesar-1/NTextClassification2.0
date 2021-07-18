import json

from pandas import np
from sklearn.model_selection import RandomizedSearchCV

from textclassification_app.classes.Experiment import Experiment


def parameter_tuning(experiment: Experiment, tuning_type: str = None):
    if tuning_type == "GridSearchCV":
        grid_search(experiment)
    elif tuning_type == "RandomizedSearchCV":
        randomized_search(experiment)


def grid_search(experiment: Experiment):
    from sklearn.model_selection import GridSearchCV
    # Create the parameter grid based on the results of random search
    param_grid = {
        "classification__n_estimators": [1900],
        "classification__min_samples_split": [5],
        "classification__min_samples_leaf": [3],
        "classification__max_features": ["sqrt"],
        "classification__max_depth": [105],
        "classification__bootstrap": [True]
    }
    print(param_grid)

    # for the convenience of reading
    clf = experiment.get_pipeline(experiment.classifiers[0])
    k_fold = experiment.classification_technique.k_fold
    X = experiment.documents
    y = experiment.labels

    # Instantiate the grid search model
    rf_random = GridSearchCV(estimator=clf, param_grid=param_grid, cv=k_fold, n_jobs=-1, verbose=2)

    # Fit the random search model
    rf_random.fit(X, y)

    print(rf_random.best_params_)
    experiment.classifiers[0] = rf_random.best_estimator_['classification']
    with open("best_estimator.json", "w") as file:
        json.dump(rf_random.best_params_, file, indent=6)


def randomized_search(experiment: Experiment):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 200, num=20)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = list(range(1, 21, 2))
    # Minimum number of samples required at each leaf node
    min_samples_leaf = list(range(1, 7))
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'classification__n_estimators': n_estimators,
                   'classification__max_features': max_features,
                   'classification__max_depth': max_depth,
                   'classification__min_samples_split': min_samples_split,
                   'classification__min_samples_leaf': min_samples_leaf,
                   'classification__bootstrap': bootstrap
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
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=150, cv=k_fold, verbose=2,
                                   random_state=42, n_jobs=-1)

    # Fit the random search model
    rf_random.fit(X, y)

    print(rf_random.best_params_)
    experiment.classifiers[0] = rf_random.best_estimator_['classification']
    with open("best_estimator.json", "w") as file:
        json.dump(rf_random.best_params_, file, indent=6)
