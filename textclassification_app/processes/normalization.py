from textclassification_app.classes.Experiment import Experiment
from textclassification_app.utils import print_message
import os
import re
import string
import shutil
import json
from pathlib import Path
import autocorrect
from nltk.corpus import stopwords
from autocorrect import Speller
from nltk import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.parsing.preprocessing import remove_stopwords as rs


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
                return True
    return False


def normalize(experiment: Experiment):
    path_parent = os.path.join(
        Path(__file__).parent.parent.parent, "corpus", experiment.language
    )
    path_original = os.path.join(path_parent, "originals")
    if not find_corpus_path(experiment):
        with open(
            os.path.join(path_original, "info.json"),
            "r+",
            encoding="utf8",
            errors="replace",
        ) as f:
            info = json.load(f)
            info["counter"] = info["counter"] + 1
            counter = info["counter"]
            json.dump(info, f)
        norm_path = os.path.join(path_parent, "norm" + str(counter))
        os.makedirs(norm_path)

        for file in os.listdir(path_original):
            if file.endswith(".json") and file != "info.json":
                shutil.copy(os.path.join(path_original, file), norm_path)
                txt_file = file.replace("json", "txt")
                with open(
                    os.path.join(path_original, txt_file),
                    "r",
                    encoding="utf8",
                    errors="replace",
                ) as f:
                    text = f.read()
                for normalization in experiment.preprocessing_functions:
                    text = normalization(text)
                with open(
                    os.path.join(norm_path, txt_file),
                    "w",
                    encoding="utf8",
                    errors="replace",
                ) as f:
                    f.write(text)
                write_info_file(norm_path, experiment)


def write_info_file(norm_path: str, experiment: Experiment):
    dic = {"normalizations": []}
    dic["normalizations"] = experiment.preprocessing_functions
    with open(
        os.path.join(norm_path, "info.json"), "w", encoding="utf8", errors="replace"
    ) as f:
        json.dump(dic, f)


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
