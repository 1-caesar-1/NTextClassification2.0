from flask import Flask, request, url_for
from flaskwebgui import FlaskUI  # get the FlaskUI class
from flask import render_template

app = Flask(__name__)
ui = FlaskUI(app)  # feed the parameters

classifiers = ["svc", "rf", "lr", "mnb"]

selection_type = {
    "chi2": "chi2",
    "mir": "mutual_info_regression",
    "mic": "mutual_info_classif",
    "fc": "f_classif",
    "rfecv": "RFECV",
    "sfm": "SelectFromModel",
}
# do your logic as usual in Flask
data = {
    "transformers": [],
    "preprocessing": [],
    "language": "",
    "features_selection": [],
    "measurements": [],
    "classifiers": [],
    "classification_technique": "TrainTest()",
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data", methods=["POST"])
def get_data():

    for key in request.form:
        if key in classifiers:
            data["classifiers"].append(key)
        if key in selection_type:
            data["features_selection"].append(selection_type[key])

    print(data)
    return render_template("index.html")


ui.run()

