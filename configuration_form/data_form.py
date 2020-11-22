from flask import Flask, request, url_for
from flaskwebgui import FlaskUI  # get the FlaskUI class
from flask import render_template
from configuration_form.write_file import data_parsing
import os
import json
from os.path import dirname, abspath, exists

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
    from textclassification_app.main import main

    parent_dir = dirname(dirname(abspath(__file__))) + "/configs"
    main(parent_dir)
    return render_template("form.html")


@app.route("/runFile")
def runFile():
    return render_template("runFile.html")


@app.route("/showConfigs")
def configTable():
    parent_dir = dirname(dirname(abspath(__file__))) + "/configs"
    configs = []
    for config in os.listdir(parent_dir):
        with open(
            os.path.join(parent_dir, config), "r", encoding="utf8", errors="replace"
        ) as f:
            con = json.load(f)
        con["name"] = config.replace(".json", "")
        configs.append(con)
    return render_template("configTable.html", configs=configs)


def run():
    ui.run()


if __name__ == "__main__":
    run()

