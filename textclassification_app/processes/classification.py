from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from textclassification_app.classes.Experiment import Experiment
from textclassification_app.classes.TrainTest import TrainTest


def classify(experiment: Experiment):
    if isinstance(experiment.classification_technique, TrainTest):
        classify_using_train_test(experiment)
    else:
        classify_using_cv(experiment)


def classify_using_train_test(experiment: Experiment):
    # split the features and the labels
    X_train, X_test, y_train, y_test = experiment.classification_technique.split(experiment.extracted_features, experiment.labels)

    result = dict()
    # for every measurement method:
    for measure in experiment.measurements:
        measure_result = dict()
        # for every classifier do:
        for model in experiment.classifiers:
            # fit the model
            model.fit(X_train, y_train)
            # make predictions
            prediction = model.predict(X_test)
            decision = model.decision(X_test)  # TODO: למצוא את הפונקציה המתאימה
            # evaluate results
            measure_result[model] = evaluate(measure, y_test, prediction, decision)
        result[measure] = measure_result
    # put the result back in the experiment
    experiment.classification_results = result


def classify_using_cv(experiment: Experiment):
    pass


def evaluate(measure, ts_labels, prediction, decision):
    measures = {
        "accuracy_score": accuracy_score(ts_labels, prediction),
        "f1_score": f1_score(ts_labels, prediction),
        "precision_score": precision_score(ts_labels, prediction),
        "recall_score": recall_score(ts_labels, prediction),
        "roc_auc_score": roc_auc_score(ts_labels, decision),
        "confusion_matrix": confusion_matrix(ts_labels, prediction)
    }
    return measures[measure]
