from flask import request
import json
from os.path import dirname, abspath, exists
from os import mkdir
import os

classifiers = ["mlp", "svc", "rf", "lr", "mnb"]


tfidf_parameters = {
    "max_features": 1,
    "analyzer": 2,
    "lowercase": 3,
    "ngram_range": 4,
    "use_idf": 5,
    "min_df": 6,
}

selection_type = {
    "chi2": "chi2",
    "mir": "mutual_info_regression",
    "mc": "mutual_info_classif",
    "fc": "f_classif",
    "rfecv": "RFECV",
    "sfm": "SelectFromModel",
}
measures = [
    "accuracy_score",
    "f1_score",
    "precision_score",
    "recall_score",
    "roc_auc_score",
    "confusion_matrix",
    "roc_curve_data",
    "precision_recall",
    "accuracy_confusion_matrix",
]
# do your logic as usual in Flask
data = {
    "transformers": [],
    "preprocessing": [],
    "language": "",
    "features_selection": [],
    "measurements": [],
    "classifiers": [],
    "classification_technique": "",
}

preprocessing = [
    "spelling_correction",
    "remove_html_tags",
    "lowercase",
    "remove_punctuation",
    "remove_repeated_characters",
    "remove_stopwords",
    "stemming",
    "lemmatizing",
    "acronyms_removal",
    "apostrophe_removal",
]


def data_parsing(request):
    parameters = dict(request.form.items())
    print(parameters)
    for key, value in parameters.items():
        if key in classifiers:
            data["classifiers"].append(key)
        if key in preprocessing:
            data["preprocessing"].append(key)
        if key in measures:
            data["measurements"].append(key)
        if key == "selection":
            data["features_selection"].append(
                (selection_type[value], parameters["selection_k"])
            )
        if key == "tf":
            tfidf_parameters["max_features"] = parameters["max"]
            tfidf_parameters["analyzer"] = "False"
            tfidf_parameters["lowercase"] = "False"
            tfidf_parameters["ngram_range"] = (
                "(" + parameters["grams"] + "," + parameters["grams"] + ")"
            )
            tfidf_parameters["use_idf"] = "False"
            tfidf_parameters["min_df"] = "3"
            temp = ["=".join(i) for i in tfidf_parameters.items()]
            temp = ",".join(temp)
            text = "TfidfVectorizer(" + temp + ")"
            data["transformers"].append(text)
        if key == "tfidf":
            tfidf_parameters["max_features"] = parameters["max"]
            tfidf_parameters["analyzer"] = "False"
            tfidf_parameters["lowercase"] = "False"
            tfidf_parameters["ngram_range"] = (
                "(" + parameters["grams"] + "," + parameters["grams"] + ")"
            )
            tfidf_parameters["use_idf"] = "True"
            tfidf_parameters["min_df"] = "3"
            temp = ["=".join(i) for i in tfidf_parameters.items()]
            temp = ",".join(temp)
            text = "TfidfVectorizer(" + temp + ")"
            data["transformers"].append(text)
        if key == "Language":
            data["language"] = value
        if key == "technique":
            data["classification_technique"] = value + "()"
    parent_dir = dirname(dirname(abspath(__file__))) + "/configs"

    if not exists(parent_dir):
        mkdir(parent_dir)
    with open(os.path.join(parent_dir, "info.json"), "r+") as f:
        dic = json.load(f)
        dic["counter"] = dic["counter"] + 1
        counter = dic["counter"]
        f.write(json.dumps(dic, indent=4))
    with open(os.path.join(parent_dir, "config" + str(counter) + ".json"), "w") as f:
        f.write(json.dumps(data, indent=4))
    print(data)


if __name__ == "__main__":
    pass
