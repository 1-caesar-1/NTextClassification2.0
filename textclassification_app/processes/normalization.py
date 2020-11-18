from textclassification_app.classes.Experiment import Experiment
from textclassification_app.utils import print_message
import os
import re
import string
import json
from pathlib import Path
import autocorrect

from autocorrect import Speller
from nltk import WordNetLemmatizer, SnowballStemmer


def normalize(experiment: Experiment):
    path_parent = os.path.join(
        Path(__file__).parent.parent.parent, "corpus", experiment.language
    )
    path_original = os.path.join(path_parent, "originals")
    _, dirnames, _ = os.walk(path_parent)
    flag = False
    for inside_folder in dirnames:
        with open(
            os.path.join(path_parent, inside_folder, "description.json"),
            "r",
            encoding="utf8",
            errors="replace",
        ) as f:
            if json.load(f)["classification"] == [
                function.__name__ for function in experiment.preprocessing_functions
            ]:
                flag = True

    if not flag:
        with open(
            os.path.join(path_original, "description.json"),
            "r",
            encoding="utf8",
            errors="replace",
        ) as f:
            counter = json.load(f)["counter"] + 1
            os.makedirs(os.path.join(path_parent, "norm" + str(counter)))
        for post in os.listdir():
            pass


def html_tags(text: str):
    return re.sub(r"((<.*)(>|/>))", "", text)


def lowercase(text: str):
    return text.lower()


def Spelling_correction(text: str):
    return Speller().autocorrect_sentence(text)


def punctuation(text: str):
    return text.translate(str.maketrans("", "", string.punctuation))


def Repeated_characters(text: str):
    pass


def Stemming(text: str):
    pass


def Lemmatizing(text: str):
    pass


def stopwords(text: str):
    pass

