from flask import Flask, request, url_for
from flaskwebgui import FlaskUI  # get the FlaskUI class
from flask import render_template
from configuration_form.write_file import data_parsing
import os
from xlsx2html import xlsx2html
import json
from os.path import dirname, abspath, exists
import shutil
from textclassification_app.main import main
from textclassification_app.classes.StylisticFeatures import initialize_features_dict
import jpype
import asposecells

jpype.startJVM(convertStrings=False)
from asposecells.api import *

app = Flask(__name__)

# app.config["DEBUG"] = True
ui = FlaskUI(app)
ui.fullscreen = True
ui.maximized = True
# feed the parameters


@app.route("/")
def index():
    stylistic = list(initialize_features_dict("en").keys())
    featurs1 = stylistic[: round(len(stylistic) / 2)]
    featurs2 = stylistic[round(len(stylistic) / 2) :]
    return render_template("form.html", featurs1=featurs1, featurs2=featurs2)


@app.route("/data", methods=["POST"])
def get_data():
    data_parsing(request)
    return render_template("runFile.html")


@app.route("/run")
def run_file():
    parent_dir = dirname(dirname(abspath(__file__))) + "/configs"
    main(parent_dir)
    return render_template("form.html")


@app.route("/runFiles")
def runFiles():
    return render_template("runFile.html")


@app.route("/showConfigs")
def configTable():
    parent_dir = dirname(dirname(abspath(__file__))) + "/configs"
    configs = []
    for config in os.listdir(parent_dir):
        if config != "info.json":
            with open(
                os.path.join(parent_dir, config), "r", encoding="utf8", errors="replace"
            ) as f:
                con = json.load(f)
            con["name"] = config.replace(".json", "")
            configs.append(con)
    return render_template("configTable.html", configs=configs)


@app.route("/runFile/<name>", methods=["get"])
def runFile(name):
    parent_dir = os.path.join(dirname(dirname(abspath(__file__))), "configs")
    runfile_path = os.path.join(parent_dir, "config")
    os.mkdir(runfile_path)
    shutil.copy(os.path.join(parent_dir, name + ".json"), runfile_path)
    try:
        main(runfile_path)
    except:
        pass
    shutil.rmtree(runfile_path)


@app.route("/results")
def results():
    result_dir = os.path.join(dirname(dirname(abspath(__file__))), "results", "excel")

    for file in os.listdir(result_dir):
        wb = Workbook(os.path.join(result_dir, file))
        # save workbook as HTML file
        html_dir = os.path.join(dirname(abspath(__file__)), "static", "html")
        wb.save(os.path.join(html_dir, file.replace("xlsx", "html")))
        os.remove(
            os.path.join(html_dir, file.replace(".xlsx", "_files"), "sheet002.htm")
        )
    results = os.listdir(os.path.join(dirname(abspath(__file__)), "static", "html"))
    results = ["html/" + i for i in results if i.endswith(".html")]

    return render_template("results.html", results=results)


def run():
    ui.run()


if __name__ == "__main__":
    run()

