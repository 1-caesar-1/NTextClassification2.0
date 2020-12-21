from flask import Flask, request, url_for, redirect
from flaskwebgui import FlaskUI  # get the FlaskUI class
from flask import render_template
from configuration_form.parse_data import data_parsing, data_parsing_range, init_data
import os
from xlsx2html import xlsx2html
import json
from os.path import dirname, abspath, exists
import shutil
from textclassification_app.main import main
from textclassification_app.classes.StylisticFeatures import initialize_features_dict
from textclassification_app.classes.ConfigJson import ConfigJson


app = Flask(__name__)

# app.config["DEBUG"] = True
ui = FlaskUI(app)
ui.fullscreen = True
ui.maximized = True
data = ConfigJson()
# feed the parameters
@app.route("/")
def index():
    return render_template("home.html")


@app.route("/start_config", methods=["POST"])
def start_config():
    global data
    data = init_data(request)
    return render_template("chooseForm.html")


@app.route("/start", methods=["POST"])
def start_form():
    global data
    data = init_data(request)
    return render_template("chooseForm.html")


@app.route("/rangeForm")
def range():
    stylistic = []
    if data.language == "English":
        stylistic = list(initialize_features_dict("en").keys())
    else:
        stylistic = list(initialize_features_dict("he").keys())
    featurs1 = stylistic[: round(len(stylistic) / 2)]
    featurs2 = stylistic[round(len(stylistic) / 2) :]
    return render_template("rangeForm.html", featurs1=featurs1, featurs2=featurs2)


@app.route("/form")
def looser():
    stylistic = []
    if data.language == "English":
        stylistic = list(initialize_features_dict("en").keys())
    else:
        stylistic = list(initialize_features_dict("he").keys())
    featurs1 = stylistic[: round(len(stylistic) / 2)]
    featurs2 = stylistic[round(len(stylistic) / 2) :]
    return render_template("form.html", featurs1=featurs1, featurs2=featurs2)


@app.route("/data/<range>", methods=["POST"])
def get_data(range):
    if range == "false":
        data_parsing(request, data)
    elif range == "true":
        data_parsing_range(request, data)
    return render_template("runFile.html")


@app.route("/run")
def run_file():
    parent_dir = dirname(dirname(abspath(__file__))) + "/configs"
    main(parent_dir)
    return redirect("/results")


@app.route("/upload")
def upload_form():
    return render_template("upload.html")


@app.route("/upload_data")
def upload():
    return render_template("upload.html")


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
    return redirect("/results")


@app.route("/results")
def results():
    result_dir = os.path.join(dirname(dirname(abspath(__file__))), "results", "excel")
    html_dir = os.path.join(dirname(abspath(__file__)), "static", "html")
    for file in os.listdir(result_dir):
        xlsx2html(
            os.path.join(result_dir, file),
            os.path.join(html_dir, file.replace("xlsx", "html")),
        )

    results = os.listdir(html_dir)
    first = {"path": "html/" + results[0], "counter": str(1), "name": results[0]}
    results.pop(0)
    results = [
        {"path": "html/" + i, "counter": str(counter), "name": i}
        for counter, i in enumerate(results, start=2)
    ]

    return render_template("results.html", results=results, first=first)


def run():
    ui.run()


if __name__ == "__main__":
    run()

