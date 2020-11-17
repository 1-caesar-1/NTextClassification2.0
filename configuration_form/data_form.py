from flask import Flask, request, url_for
from flaskwebgui import FlaskUI  # get the FlaskUI class
from flask import render_template
from write_file import data_parsing

from os.path import dirname, abspath, exists

app = Flask(__name__)
ui = FlaskUI(app)
ui.fullscreen = True
ui.maximized = True
# feed the parameters


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data", methods=["POST"])
def get_data():
    data_parsing(request)
    return render_template("runFile.html")


@app.route("/run")
def run_file():
    from textclassification_app.main import main

    parent_dir = dirname(dirname(abspath(__file__))) + "/configs"
    main(parent_dir)
    return render_template("index.html")


def run():
    ui.run()


if __name__ == "__main__":
    run()

