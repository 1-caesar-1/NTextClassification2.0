from flask import request
import json
from os.path import dirname, abspath, exists
from os import mkdir
import os
from textclassification_app.classes.StylisticFeatures import initialize_features_dict

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
    "fr": "f_regression",
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
    stylistic = list(initialize_features_dict("en").keys())
    stylistic_use = []
    for key, value in parameters.items():
        if key in classifiers:
            data["classifiers"].append(key)
        elif key in preprocessing:
            data["preprocessing"].append(key)
        elif key in measures:
            data["measurements"].append(key)
        elif key == "selection":
            data["features_selection"].append(
                (selection_type[value], parameters["selection_k"])
            )
        elif key == "tf":
            tfidf_parameters["max_features"] = parameters["max"]
            tfidf_parameters["analyzer"] = "'" + parameters["Analyzer"] + "'"
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
        elif key == "tfidf":
            tfidf_parameters["max_features"] = parameters["max"]
            tfidf_parameters["analyzer"] = "'" + parameters["Analyzer"] + "'"
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
        elif key == "w2v":
            data["transformers"].append("W2VTransformer()")
        elif key == "d2v":
            data["transformers"].append("Doc2VecTransfomer()")
        elif key == "Language":
            data["language"] = value
        elif key == "technique":
            data["classification_technique"] = value + "()"
        elif key in stylistic:
            stylistic_use.append(key)
    data["transformers"].append(
        "StylisticFeatures('"
        + "','".join(stylistic_use)
        + ",language='"
        + data["language"]
        + "')"
    )
    parent_dir = dirname(dirname(abspath(__file__))) + "/configs"

    if not exists(parent_dir):
        mkdir(parent_dir)
    with open(os.path.join(parent_dir, "info.json"), "r") as f:
        dic = json.load(f)
    dic["counter"] = dic["counter"] + 1
    counter = dic["counter"]
    with open(os.path.join(parent_dir, "info.json"), "w") as f:
        f.write(json.dumps(dic, indent=4))
    with open(os.path.join(parent_dir, "config" + str(counter) + ".json"), "w") as f:
        f.write(json.dumps(data, indent=4))
    print(data)


if __name__ == "__main__":
    print("','".join(["2", "2"]))
