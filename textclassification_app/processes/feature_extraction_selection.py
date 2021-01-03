import json
import os
import random
from pathlib import Path

from sklearn import preprocessing

from textclassification_app.classes.Experiment import Experiment
from textclassification_app.utils import print_message
from textclassification_app.rw_files.r_files import read_json_corpus


def extract_data(experiment: Experiment):
    print_message("extracting features for " + experiment.experiment_name, num_tabs=1)

    # load all files and labels into docs
    dir = find_corpus_path(experiment)
    docs = read_json_corpus(dir, onlyLabel=True)

    # sort all docs and then shuffle them using const seed
    docs.sort(key=lambda doc: doc[0])
    random.Random(4).shuffle(docs)

    # split the labels from the data
    X, y = zip(*docs)

    # encode the labels
    le = preprocessing.LabelEncoder()
    experiment.labels = le.fit_transform(y)

    # save the documents
    experiment.documents = X


def find_corpus_path(experiment: Experiment):
    parent_folder = os.path.join(
        Path(__file__).parent.parent.parent, "corpus", experiment.language
    )
    for inside_folder in os.listdir(parent_folder):
        with open(
            os.path.join(parent_folder, inside_folder, "info.json"),
            "r",
            encoding="utf8",
            errors="replace",
        ) as f:
            if json.load(f)["normalizations"] == [
                function.__name__ for function in experiment.preprocessing_functions
            ]:
                return os.path.join(parent_folder, inside_folder)
    return os.path.join(parent_folder, "originals")
