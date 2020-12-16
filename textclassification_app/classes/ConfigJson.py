from textclassification_app.classes.StylisticFeatures import initialize_features_dict
import json
from os.path import dirname, abspath, exists
from os import mkdir
import os


class ConfigJson:
    def __init__(self, language=None, technique=""):
        self.language = language
        self.transformers = []
        self.preprocessing = []
        self.features_selection = []
        self.measurements = []
        self.classifiers = []
        self.classification_technique = technique + "()"

    def add_tfidf(
        self, use_idf: bool, max_features, analyzer, ngram, min_df=3, lowercase=False
    ):
        tfidf_parameters = {
            "max_features": "",
            "analyzer": "",
            "lowercase": "",
            "ngram_range": "",
            "use_idf": "",
            "min_df": "",
        }
        tfidf_parameters["max_features"] = max_features
        tfidf_parameters["analyzer"] = "'" + analyzer + "'"
        tfidf_parameters["lowercase"] = str(lowercase)
        tfidf_parameters["ngram_range"] = "(" + ngram + "," + ngram + ")"
        tfidf_parameters["use_idf"] = str(use_idf)
        tfidf_parameters["min_df"] = str(min_df)
        temp = ["=".join(i) for i in tfidf_parameters.items()]
        temp = ",".join(temp)
        tfidf_transfomer = "TfidfVectorizer(" + temp + ")"
        self.transformers.append(tfidf_transfomer)

    def add_stylistic(self, stylistic):
        self.transformers.append(
            "StylisticFeatures('"
            + "','".join(stylistic)
            + ",language='"
            + self.language
            + "')"
        )

    def add_transfomer(self, transfomer):
        self.transformers.append(transfomer)

    def add_selection(self, selection, k):
        self.features_selection.append((selection, k))

    def write_to_file(self):
        parent_dir = dirname(dirname(dirname(abspath(__file__)))) + "/configs"

        if not exists(parent_dir):
            mkdir(parent_dir)
        with open(os.path.join(parent_dir, "info.json"), "r") as f:
            dic = json.load(f)

        dic["counter"] = dic["counter"] + 1
        counter = dic["counter"]
        with open(os.path.join(parent_dir, "info.json"), "w") as f:
            f.write(json.dumps(dic, indent=4))
        with open(
            os.path.join(parent_dir, "config" + str(counter) + ".json"), "w"
        ) as f:
            f.write(json.dumps(self.__dict__, indent=4))
