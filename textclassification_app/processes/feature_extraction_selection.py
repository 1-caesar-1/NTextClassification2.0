import json
import os
from pathlib import Path
import random

from sklearn import preprocessing

from textclassification_app.classes.Experiment import Experiment
from textclassification_app.utils import print_message


def extract_features(experiment: Experiment):
    print_message("extracting features for " + experiment.experiment_name, num_tabs=1)

    # load all files and labels into docs
    dir = os.path.join(Path(__file__).parent.parent.parent, 'corpus')
    docs = []
    for file in os.listdir(dir):
        if file.endswith(".json"):
            with open(dir + "\\" + file, "r", encoding="utf8", errors="replace") as f:
                label = json.load(f)["classification"]
            with open(dir + "\\" + file.replace('.json', '.txt'), "r", encoding="utf8", errors="replace") as f:
                data = f.read()
            docs += [(data, label)]

    # sort all docs and then shuffle them using const seed
    docs.sort(key=lambda doc: doc[0])
    random.Random(4).shuffle(docs)

    # split the labels from the data
    X, y = zip(*docs)

    # encode the labels
    le = preprocessing.LabelEncoder()
    experiment.labels = le.fit_transform(y)

    # extract features
    experiment.extracted_features = experiment.features_extraction_transformers.fit_transform(X, experiment.labels)


def select_features(experiment: Experiment):
    pass
