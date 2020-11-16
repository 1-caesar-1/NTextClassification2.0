from flask import Flask, request, url_for
from flaskwebgui import FlaskUI  # get the FlaskUI class
from flask import render_template
from write_file import data_parsing

app = Flask(__name__)
ui = FlaskUI(app)  # feed the parameters


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data", methods=["POST"])
def get_data():
    data_parsing(request)
    return render_template("index.html")


if __name__ == "__main__":
    ui.run()

