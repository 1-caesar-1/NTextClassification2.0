import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import random
from textclassification_app.rw_files.r_files import read_json_corpus
import numpy as np
from textclassification_app.classes.Experiment import Experiment


def run_rnn(experiment: Experiment, cv=True):
    k_fold = KFold()
    # for the convenience of reading
    X = experiment.documents
    y = experiment.labels

    # initialize the result dict
    result = dict()
    for measure in experiment.measurements:
        result[measure] = dict()
        for model in experiment.classifiers:
            result[measure][type(model).__name__] = list()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    if cv:
        for train, test in k_fold.split(X, y):
            for i in range(experiment.classification_technique.iteration):
                model = get_model(X[train])
                history = model.fit(
                    np.array(X[train]),
                    y[train],
                    epochs=75,
                    validation_data=(np.array(X[test]), y[test]),
                    validation_steps=10,
                )
                result["accuracy"]["RNNClassifier"] += history.history["val_accuracy"]
                test_loss, test_acc = model.evaluate(np.array(X[test]), y[test])
                print("Test Loss: {}".format(test_loss))
                print("Test Accuracy: {}".format(test_acc))
        experiment.classification_results = result
        experiment.general_data["num_of_features"] = 0
    else:
        model = get_model(X_train)
        history = model.fit(
            np.array(X_train),
            y_train,
            epochs=75,
            validation_data=(np.array(X_test), y_test),
            validation_steps=10,
        )
        result["accuracy"]["RNNClassifier"] += history.history["val_accuracy"]
        test_loss, test_acc = model.evaluate(np.array(X_test), y_test)
        print("Test Loss: {}".format(test_loss))
        print("Test Accuracy: {}".format(test_acc))
    experiment.classification_results = result
    experiment.general_data["num_of_features"] = 0


def get_model(X):
    VOCAB_SIZE = 600
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=VOCAB_SIZE
    )
    encoder.adapt(X)
    model = tf.keras.Sequential(
        [
            encoder,
            tf.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()),
                output_dim=64,
                # Use masking to handle the variable sequence lengths
                mask_zero=True,
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="Adamax", metrics=["accuracy"])
    return model

