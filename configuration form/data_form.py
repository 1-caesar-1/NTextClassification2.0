from flask import Flask, request, url_for
from flaskwebgui import FlaskUI  # get the FlaskUI class
from flask import render_template
import json
from os.path import dirname, abspath
from os import mkdir


app = Flask(__name__)
ui = FlaskUI(app)  # feed the parameters

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
    data_parsing(request)
    return render_template("index.html")


def data_parsing(request):
    parameters = dict(request.form.items())
    print(parameters)
    for key, _ in parameters.items():
        if key in classifiers:
            data["classifiers"].append(key)
        if key in selection_type:
            data["features_selection"].append(selection_type[key])
        if key == "tf":
            tfidf_parameters["max_features"] = parameters["max"]
            tfidf_parameters["analyzer"] = "False"
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

    mkdir(dirname(dirname(abspath(__file__))) + "/configs")
    with open(dirname(dirname(abspath(__file__))) + "/configs/config.json", "w") as f:
        f.write(json.dumps(data))
    print(data)


ui.run()

