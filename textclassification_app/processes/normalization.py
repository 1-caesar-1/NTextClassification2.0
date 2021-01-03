import json
import os
import re
import shutil
import string
from pathlib import Path
from wordfreq import top_n_list
from autocorrect import Speller
from gensim.parsing.preprocessing import remove_stopwords as rs
from nltk import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
import hebrew_tokenizer as ht
from textclassification_app.classes.Experiment import Experiment
from textclassification_app.classes.stopwords_and_lists import hebrew_stopwords
from textclassification_app.rw_files.r_files import read_json_corpus
from textclassification_app.rw_files.w_files import write_json_corpus


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
            "r",
            encoding="utf8",
            errors="replace",
        ) as f:
            info = json.load(f)

        info["counter"] = info["counter"] + 1
        counter = info["counter"]
        with open(
            os.path.join(path_original, "info.json"),
            "w",
            encoding="utf8",
            errors="replace",
        ) as f:
            json.dump(info, f)
        norm_path = os.path.join(path_parent, "norm" + str(counter))
        os.makedirs(norm_path)

        corpus_files = read_json_corpus(path_original)
        normelized_corpus_files = []
        for text, data in corpus_files:
            for normalization in experiment.preprocessing_functions:
                text = normalization(text, experiment.language)
            normelized_corpus_files.append((text, data))
        write_json_corpus(norm_path, normelized_corpus_files)
        write_info_file(norm_path, experiment)


def write_info_file(norm_path: str, experiment: Experiment):
    dic = {
        "normalizations": [
            function.__name__ for function in experiment.preprocessing_functions
        ]
    }
    with open(
        os.path.join(norm_path, "info.json"), "w", encoding="utf8", errors="replace"
    ) as f:
        json.dump(dic, f)


def remove_html_tags(text: str, language: str = "english"):
    return re.sub(r"((<.*)(>|/>))", "", text)


def lowercase(text: str, language: str = "english"):
    return text.lower()


def spelling_correction(text: str, language: str = "english"):
    return Speller().autocorrect_sentence(text)


def remove_punctuation(text: str, language: str = "english"):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_repeated_characters(text: str, language: str = "english"):
    pass


def stemming(text: str, language: str = "english"):
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


def lemmatizing(text: str, language: str = "english"):
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


def remove_stopwords(text: str, language: str = "english"):
    if language == "hebrew":
        tokenized = [token[1] for token in ht.tokenize(text, with_whitespaces=True)]
        tokenized = [word for word in tokenized if word not in hebrew_stopwords]
        result = []
        for word in tokenized:
            for sword in hebrew_stopwords:
                if word.startswith(sword) and word[len(sword) :] in top_n_list(
                    "he", 100000
                ):
                    word = word[len(sword) :]
            result.append(word)
        text = "".join(result)

    if language == "english":
        text = rs(text)

    return text


def apostrophe_removal(text: str, language: str = "english"):
    return text.replace("'", "").replace("'", "").replace("’", "")


def acronyms_removal(text: str, language: str = "english"):
    acr = set(re.findall(r"\w\.\w\.\w", text) + re.findall(r"\w+\"\w", text))
    for acr in acr:
        text = re.sub(acr, acr.replace(".", ""), text)
        text = re.sub(acr, acr.replace('"', ""), text)
    return text


if __name__ == "__main__":
    pass

