import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy
import xlsxwriter

from textclassification_app.classes.CrossValidation import CrossValidation
from textclassification_app.classes.Experiment import Experiment
from textclassification_app.classes.TrainTest import TrainTest
from textclassification_app.processes.classification import measures
from textclassification_app.processes.statistical_significance import differences_significance

_ = [CrossValidation, TrainTest]


def save_experiment_results(experiment: Experiment):
    """
    write the result of the experiment into JSON file
    :param experiment: the experiment to save
    """
    content = {"config": experiment.general_data, "results": experiment.classification_results}
    parent_folder = os.path.join(Path(__file__).parent.parent.parent, "results", "json")
    path = os.path.join(parent_folder, experiment.experiment_name + " results" + ".json")
    with open(path, "w+", encoding="utf8", errors="replace") as f:
        json.dump(content, f, indent=4)


def write_all_experiments():
    dir = os.path.join(Path(__file__).parent.parent.parent, "results", "json")
    for measure in measures:
        result = []
        for file in os.listdir(dir):
            with open(os.path.join(dir, file), "r", encoding="utf8", errors="replace") as f:
                json_file = json.load(f)
                if measure in json_file["results"]:
                    result += [(json_file["config"], json_file["results"][measure])]
        if result:
            write_xlsx(result, measure)


def write_xlsx(data: list, measure: str):
    # create an new Excel file and add a worksheet.
    file_path = os.path.join(Path(__file__).parent.parent.parent, "results", "excel", measure + ".xlsx")
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet()

    # create font styles
    regular = workbook.add_format({"font_name": "Times New Roman"})
    extra_big = workbook.add_format({"bold": True, "font_size": 17, "underline": True, "font_name": "Times New Roman"})
    big = workbook.add_format({"bold": True, "font_size": 12, "font_name": "Times New Roman"})
    bold_gray = workbook.add_format({"bold": True, "font_color": "gray", "font_name": "Times New Roman"})
    gray = workbook.add_format({"font_color": "gray", "font_name": "Times New Roman"})
    good = workbook.add_format({"font_color": "blue", "font_name": "Times New Roman"})
    best = workbook.add_format({"font_color": "red", "font_name": "Times New Roman"})
    centralized = workbook.add_format({'text_wrap': 'true', 'font_size': 10, 'align': 'center', 'valign': 'vcenter',
                                       "font_name": "Times New Roman"})

    # write titles
    worksheet.write("A1", "Classification results: " + measure.replace("_", " "), extra_big)
    worksheet.write("A2", "The classification results are shown in the table below in percentages", regular)

    # write general data
    worksheet.write("A4", "General information about the classification software", bold_gray)
    now = datetime.now()
    worksheet.write("A5", "Issue date: " + now.strftime("%d/%m/%Y %H:%M:%S"), gray)
    version = str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "." + str(sys.version_info[2])
    worksheet.write("A6", "Python version: Python " + version, gray)
    worksheet.write("A7", "Python classification libraries: keras, sklearn, tensorflow, VADAR from nltk, WordCloud",
                    gray)

    # write differences significance option
    worksheet.write("A9", "Statistical Significance Options", bold_gray)
    worksheet.write("A10", "V - Significantly larger than the baseline", gray)
    worksheet.write("A11", "* - Significantly smaller than the baseline", gray)

    # write colors description
    worksheet.write("F9", "Colors description", bold_gray)
    worksheet.write("F10", "The best result of the learning method", good)
    worksheet.write("F11", "The best result of all classification", best)

    # write the results
    worksheet.write("A14", "Results:", big)
    sizes = {1: [len("Language")], 2: [len("Features types")], 3: [len("Features selectors")],
             4: [len("Pre-processing")], 5: [len("Technique")]}
    bests = {"MLPClassifier": [], "LinearSVC": [], "LogisticRegression": [], "RandomForestClassifier": [],
             "MultinomialNB": [], "SVC": [], "RNNEstimator": []}
    indexes = {"MLPClassifier": 6, "LinearSVC": 7, "LogisticRegression": 8, "RandomForestClassifier": 9,
               "MultinomialNB": 10, "SVC": 7, "RNNEstimator": 11}
    row = 15
    for experiment, results in data:
        worksheet.write_number(row, 0, experiment["num_of_features"], centralized)

        worksheet.write(row, 1, experiment["language"], centralized)
        sizes[1] += [len(experiment["language"])]

        # pretty display of the transformers
        transformers = ""
        for transformer in experiment["transformers"]:
            name = transformer.split("(")[0]
            parms = transformer[::-1].replace(")", "", 1)[::-1].split("(", maxsplit=1)[1].split(",")
            named_parms = dict((x.split("=")[0], x.split("=")[1]) for x in parms if "=" in x)
            if name == "TfidfVectorizer":
                transformers += "TF"
                transformers += "-IDF: " if named_parms["use_idf"] == "True" else ": "
                transformers += named_parms["max_features"] + " "
                transformers += named_parms["analyzer"].replace("'", "").replace('"', '') + "s "
                transformers += {1: "unigrams", 2: "bigrmas", 3: "trigrams"}[int(named_parms["ngram_range"][-1])] + "\n"
            elif name == "StylisticFeatures":
                transformers += "Stylistic Features: "
                transformers += ", ".join(x.replace("'", "") for x in parms if "language" not in x) + "\n"
            else:
                transformers += name + "\n"
        worksheet.write(row, 2, transformers[:-1], centralized)
        sizes[2] += [max(len(x) for x in transformers.split('\n'))]

        # pretty display of the features selections
        selections = ", ".join(
            str(x[1]) + " " + x[0].replace("'", "").replace('"', '').replace("_", " ") for x in
            experiment["features_selection"])
        worksheet.write(row, 3, selections if selections else "None", centralized)
        sizes[3] += [len(selections if selections else 'None')]

        # pretty display of the pre-processing
        preprocessing = ", ".join(
            x.replace("'", "").replace('"', '').replace("_", " ") for x in experiment["preprocessing"])
        worksheet.write(row, 4, preprocessing if preprocessing else "None", centralized)

        # write the classification technique
        try:
            technique = str(eval(experiment["classification_technique"]))
        except:
            technique = str(CrossValidation())
        worksheet.write(row, 5, technique, centralized)
        sizes[5] += [len(technique)]

        # write the result
        for algo, arr in results.items():
            value = numpy.mean(arr) * 100
            k_folds = eval(experiment["classification_technique"]).k_fold if "CrossValidation" in experiment[
                "classification_technique"] else 1
            if k_folds > 1:
                value = str(float("{0:.4g}".format(value))) + differences_significance(arr, measure, k_folds,
                                                                                       experiment["language"])
                worksheet.write(row, indexes[algo], value, centralized)
            else:
                worksheet.write_number(row, indexes[algo], float("{0:.4g}".format(value)), centralized)
            bests[algo] += [(str(value), row)]

        row += 1

    # paint the bests results
    good.set_align("center")
    good.set_align("vcenter")
    good.set_font_size(10)
    best.set_align("center")
    best.set_align("vcenter")
    best.set_font_size(10)
    max_value = (_, _, 0)

    def convert(x):
        if str(x).lower() == "nan":
            return math.nan
        return float(str(x).replace('*', '').replace('V', ''))

    for algo, lst in bests.items():
        if lst:
            value = sorted(lst, key=lambda x: convert(x[0]))[-1]
            worksheet.write(value[1], indexes[algo], value[0], good)
            current_value_float = convert(value[0])
            max_value_float = convert(max_value[2])
            if current_value_float > max_value_float:
                max_value = (value[1], indexes[algo], value[0])
    worksheet.write(max_value[0], max_value[1], max_value[2], best)

    # enlarge the sizes of the columns
    for key in sizes:
        worksheet.set_column(key, key, max(sizes[key]))

    # mark the table
    worksheet.add_table(
        "A15:L" + str(row),
        {
            "columns": [
                {"header": "Number"},
                {"header": "Language"},
                {"header": "Features types"},
                {"header": "Features selectors"},
                {"header": "Pre-processing"},
                {"header": "Technique"},
                {"header": "MLP"},
                {"header": "SVC"},
                {"header": "LR"},
                {"header": "RF"},
                {"header": "MNB"},
                {"header": "RNN"}
            ],
            "style": "Table Style Light " + str(random.randint(9, 14))
        }
    )

    # close & save
    workbook.close()


if __name__ == '__main__':
    write_all_experiments()
