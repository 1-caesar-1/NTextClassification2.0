from flask import request
import json
from os.path import dirname, abspath, exists
from os import mkdir
import os
from textclassification_app.classes.StylisticFeatures import initialize_features_dict
from textclassification_app.classes.ConfigJson import ConfigJson

classifiers = ["mlp", "svc", "rf", "lr", "mnb"]


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
def init_data(request):
    parameters = dict(request.form.items())
    return ConfigJson(
        language=parameters["Language"], technique=parameters["technique"]
    )


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


def data_parsing_range(request, new_data):
    parameters = dict(request.form.items())
    data = ConfigJson(language=new_data.language, technique=new_data.technique)
    stylistic = []
    if new_data.language == "English":
        stylistic = list(initialize_features_dict("en").keys())
    else:
        stylistic = list(initialize_features_dict("he").keys())

    for i in range(
        int(parameters["min_grams"]),
        int(parameters["max_grams"]),
        int(parameters["jump_grams"]),
    ):
        data = ConfigJson(language=new_data.language, technique=new_data.technique)
        stylistic_use = []
        for key, value in parameters.items():
            if key in classifiers:
                data.classifiers.append(key)
            elif key in preprocessing:
                data.preprocessing.append(key)
            elif key in measures:
                data.measurements.append(key)
            elif key == "selection":
                data.add_selection(selection_type[value], parameters["selection_k"])

            elif key == "transfomer":
                new_data.add_tfidf(
                    use_idf=value == "tfidf",
                    max_features=parameters["max"],
                    analyzer=parameters["Analyzer"],
                    ngram=i,
                )

            elif key == "w2v":
                data.add_transfomer("W2VTransformer()")
            elif key == "d2v":
                data.add_transfomer("Doc2VecTransfomer()")
            elif key in stylistic:
                stylistic_use.append(key)
        if stylistic_use:
            new_data.add_stylistic(stylistic_use)
        data.write_to_file()


def data_parsing(request, new_data):
    parameters = dict(request.form.items())
    new_data
    stylistic = []
    if new_data.language == "English":
        stylistic = list(initialize_features_dict("en").keys())
    else:
        stylistic = list(initialize_features_dict("he").keys())

    stylistic_use = []
    for key, value in parameters.items():
        if key in classifiers:
            new_data.classifiers.append(key)
        elif key in preprocessing:
            new_data.preprocessing.append(key)
        elif key in measures:
            new_data.measurements.append(key)
        elif key == "selection":
            new_data.add_selection(
                selection=selection_type[value], k=parameters["selection_k"]
            )

        elif key == "tf":
            new_data.add_tfidf(
                use_idf=False,
                max_features=parameters["max"],
                analyzer=parameters["Analyzer"],
                ngram=parameters["grams"],
            )
        elif key == "tfidf":
            new_data.add_tfidf(
                use_idf=True,
                max_features=parameters["max"],
                analyzer=parameters["Analyzer"],
                ngram=parameters["grams"],
            )
        elif key == "w2v":
            new_data.add_transfomer("W2VTransformer()")
        elif key == "d2v":
            new_data.add_transfomer("Doc2VecTransfomer()")
        elif key in stylistic:
            stylistic_use.append(key)
    if stylistic_use:
        new_data.add_stylistic(stylistic_use)
    new_data.write_to_file()


if __name__ == "__main__":
    print("','".join(["2", "2"]))
