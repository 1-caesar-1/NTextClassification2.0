from textclassification_app.classes.Experiment import Experiment
from textclassification_app.utils import print_message
import os
import re
import string
import json
from pathlib import Path
import autocorrect
from nltk.corpus import stopwords
from autocorrect import Speller
from nltk import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.parsing.preprocessing import remove_stopwords as rs


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
    stemmer = SnowballStemmer("english")
    text_lines = text.split("\n")
    stemmed_lines = []
    for line in text_lines:
        line_words = line.split(" ")
        stemmed_line = []
        for word in line_words:
            tokinized_word = word_tokenize(word)
            stemmed_word = []
            for part in tokinized_word:
                stemmed_word.append(stemmer.stem(part))
            stemmed_line.append("".join(stemmed_word))
        stemmed_lines.append(" ".join(stemmed_line))
    return "\n".join(stemmed_lines)


def Lemmatizing(text: str):
    text_lines = text.split("\n")
    lemmatizer = WordNetLemmatizer()
    lemmatized_lines = []
    for line in text_lines:
        line_words = line.split(" ")
        lemmatized_line = []
        for word in line_words:
            tokinized_word = word_tokenize(word)
            lemmatized_word = []
            for part in tokinized_word:
                lemmatized_word.append(lemmatizer.lemmatize(part))
            lemmatized_line.append("".join(lemmatized_word))
        lemmatized_lines.append(" ".join(lemmatized_line))
    return "\n".join(lemmatized_lines)


def remove_stopwords(text: str):
    return rs(text)


def apostrophe_removal(text):
    return text.replace("'", "").replace("'", "").replace("’", "")


def acronyms_removal(text):
    acr = set(re.findall(r"\w\.\w\.\w", text) + re.findall(r"\w+\"\w", text))
    for acr in acr:
        text = re.sub(acr, acr.replace(".", ""), text)
        text = re.sub(acr, acr.replace('"', ""), text)
    return text
