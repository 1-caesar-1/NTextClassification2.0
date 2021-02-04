from textclassification_app.classes.StylisticFeatures import initialize_features_dict
import json
from os.path import dirname, abspath, exists
from os import mkdir
import os
from textclassification_app.rw_files.r_files import get_and_increase_info_counter


class ConfigJson:
    def __init__(self, language=None, technique=None, path=None):
        if not path and not technique and not language:
            pass
        elif path:
            with open(path, mode="r") as f:
                dic = json.load(f)
            self.language = dic["language"]
            self.transformers = dic["transformers"]
            self.preprocessing = dic["preprocessing"]
            self.features_selection = dic["features_selection"]
            self.measurements = dic["measurements"]
            self.classifiers = dic["classifiers"]
            self.classification_technique = dic["classification_technique"]
        elif not path and language and technique:
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
        tfidf_parameters["ngram_range"] = "(" + str(ngram) + "," + str(ngram) + ")"
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

    def read_from_file(self, path):
        with open(path, mode="r") as f:
            dic = json.load(f)
        self.language = dic["language"]
        self.transformers = dic["transformers"]
        self.preprocessing = dic["preprocessing"]
        self.features_selection = dic["features_selection"]
        self.measurements = dic["measurements"]
        self.classifiers = dic["classifiers"]
        self.classification_technique = dic["classification_technique"]

    def add_transfomer(self, transfomer):
        self.transformers.append(transfomer)

    def add_selection(self, selection, k):
        self.features_selection.append((selection, k))

    def write_to_file(self):
        parent_dir = dirname(dirname(dirname(abspath(__file__)))) + "/configs"

        if not exists(parent_dir):
            mkdir(parent_dir)
        counter = get_and_increase_info_counter(parent_dir)
        with open(
            os.path.join(parent_dir, "config" + str(counter) + ".json"), "w"
        ) as f:
            f.write(json.dumps(self.__dict__, indent=4))
