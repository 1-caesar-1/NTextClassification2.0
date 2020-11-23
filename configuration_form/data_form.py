from flask import Flask, request, url_for
from flaskwebgui import FlaskUI  # get the FlaskUI class
from flask import render_template
from configuration_form.write_file import data_parsing
import os
import json
from os.path import dirname, abspath, exists
import shutil
from textclassification_app.main import main

app = Flask(__name__)
ui = FlaskUI(app)
ui.fullscreen = True
ui.maximized = True
# feed the parameters


@app.route("/")
def index():
    return render_template("form.html")


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
    main(runfile_path)
    shutil.rmtree(runfile_path)


def run():
    ui.run()


if __name__ == "__main__":
    run()

