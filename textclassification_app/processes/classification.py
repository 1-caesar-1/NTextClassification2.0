from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate

from textclassification_app.classes.Experiment import Experiment
from textclassification_app.classes.TrainTest import TrainTest
from textclassification_app.utils import print_message


def classify(experiment: Experiment):
    if isinstance(experiment.classification_technique, TrainTest):
        classify_using_train_test(experiment)
    else:
        classify_using_cv(experiment)


def classify_using_train_test(experiment: Experiment):
    print_message("classifying " + experiment.experiment_name + " using train & test split", num_tabs=1)

    # split the features and the labels
    X_train, X_test, y_train, y_test = experiment.classification_technique.split(experiment.documents,
                                                                                 experiment.labels)

    # initialize the result dict
    result = dict()
    for measure in experiment.measurements:
        result[measure] = dict()
        for model in experiment.classifiers:
            result[measure][type(model).__name__] = list()

    # for each classification model classify the data using its pipeline
    for clf in experiment.classifiers:
        # get the pipeline from the experiment
        pipeline = experiment.get_pipeline(clf)

        # fit the pipeline
        pipeline.fit(X_train, y_train)

        # calculate the decision & prediction
        prediction = pipeline.predict(X_test)
        try:
            decision = pipeline.decision_function(X_test)
        except:
            decision = pipeline.predict_proba(X_test)
            decision = decision[:, 1]

        # evaluate the score for each measure
        for measure in experiment.measurements:
            result[measure][type(clf).__name__] += [evaluate(measure, y_test, prediction, decision)]

    # save the final results into experiment
    experiment.classification_results = result


def classify_using_cv(experiment: Experiment):
    print_message("classifying " + experiment.experiment_name + " using CV", num_tabs=1)

    # for the convenience of reading
    X = experiment.documents
    y = experiment.labels

    # initialize the result dict
    result = dict()
    for measure in experiment.measurements:
        result[measure] = dict()
        for model in experiment.classifiers:
            result[measure][type(model).__name__] = list()

    # for each classification model classify the data using its pipeline
    for clf in experiment.classifiers:
        # get the pipeline from the experiment
        pipeline = experiment.get_pipeline(clf)

        # run the number of iteration
        for _ in range(experiment.classification_technique.iteration):
            # create list of scorers
            scoring = [measures[m] for m in experiment.measurements]

            # run CV on the pipeline
            scores = cross_validate(pipeline, X, y, cv=experiment.classification_technique.k_fold, scoring=scoring,
                                    n_jobs=-1)

            # store the scores for each measure
            for measure in experiment.measurements:
                result[measure][type(clf).__name__] += list(scores["test_" + measures[measure]])

    # save the final results into experiment
    experiment.classification_results = result


def evaluate(measure, true_labels, prediction, decision):
    measures = {
        "accuracy_score": accuracy_score(true_labels, prediction),
        "f1_score": f1_score(true_labels, prediction),
        "precision_score": precision_score(true_labels, prediction),
        "recall_score": recall_score(true_labels, prediction),
        "roc_auc_score": roc_auc_score(true_labels, decision)
    }
    return measures[measure]


measures = {
    "accuracy_score": "accuracy",
    "f1_score": "f1",
    "precision_score": "precision",
    "recall_score": "recall",
    "roc_auc_score": "roc_auc"
}
