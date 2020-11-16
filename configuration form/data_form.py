from flask import Flask, request, url_for
from flaskwebgui import FlaskUI  # get the FlaskUI class
from flask import render_template

app = Flask(__name__)
ui = FlaskUI(app)  # feed the parameters


# do your logic as usual in Flask


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data", methods=["POST"])
def get_data():
    print(request.get_data())
    return render_template("index.html")


ui.run()

