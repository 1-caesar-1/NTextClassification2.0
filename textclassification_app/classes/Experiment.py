import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from gensim.sklearn_api import D2VTransformer, W2VTransformer

from textclassification_app.classes.CrossValidation import CrossValidation
from textclassification_app.classes.StylisticFeatures import StylisticFeatures
from textclassification_app.classes.TrainTest import TrainTest
from textclassification_app.utils import print_error

# ignore this section
# this section is for not omit imports that come into use in 'eval'
_ = [TfidfVectorizer, TrainTest, StylisticFeatures]

classifiers_objects = {
    "svc": LinearSVC(),
    "rf": RandomForestClassifier(n_jobs=-1),
    "mlp": MLPClassifier(),
    "lr": LogisticRegression(n_jobs=-1),
    "mnb": MultinomialNB(),
}


class Experiment:
    """
    A class representing a single classification experiment

    Attributes
    ----------
    experiment_name: str
        The name of the experiment, for convenience of reading. Default: 'un-named experiment'.

    language: str
        Document language in this experiment, for advanced feature needs.

    preprocessing_functions: list[function]
        A collection of functions for pre-processing the text.

    features_extraction_transformers: FeatureUnion
        Sklearn's FeatureUnion to extract the features from the text.

    features_selection: SelectKBest
        Object of SelectKBest, None if not needed.

    measurements: list[str]
        List of measurement methods for displaying classification results (accuracy, precision etc.)

    classifiers: list[classifier]
        List of objects of classifiers to perform the classification itself (LinearSVC, RandomForestClassifier etc.)

    classification_technique: TrainTest / CrossValidation
        An instance of class TrainTest or CrossValidation that contains information about the classification technique.

    labels: nd_array
        The labels of the documents in this experiment.
        This variable will be None before extracting the features and should contain the labels after extracting.

    documents: list[str]
        # The features matrix of the documents in this experiment.
        # This variable will be None before extracting the features and should contain the features matrix after extracting.

    classification_results: dict
        The classification results of this experiment.
        This variable will be None before classification occurred and should contain the result afterwards.
    """

    def __init__(self, path: str, experiment_name: str = "un-named experiment"):
        # load the JSON file into config
        with open(path, "r", encoding="utf8", errors="replace") as file:
            config = json.load(file)

        # create name to the experiment
        self.experiment_name = experiment_name

        # get the language of the experiment
        self.language = config["language"].lower()

        # create a list of pre-processing functions
        self.preprocessing_functions = []
        for normalization in config["preprocessing"]:
            try:
                self.preprocessing_functions += [eval(normalization)]
            except Exception as e:
                print_error(
                    "cannot load pre-processing function "
                    + normalization
                    + ": "
                    + str(e),
                    num_tabs=1,
                )

        # create FeatureUnion for all the features transformers
        transformers = []
        counter = 1
        for transformer in config["transformers"]:
            try:
                transformers += [
                    (transformer.split("(")[0] + str(counter), eval(transformer))
                ]
                counter += 1
            except Exception as e:
                print_error(
                    "cannot create transformer "
                    + transformer.split("(")[0]
                    + ": "
                    + str(e),
                    num_tabs=1,
                )
        self.features_extraction_transformers = FeatureUnion(transformers, n_jobs=-1)

        # create variable with the name of the features selection
        try:
            if config["features_selection"]:
                model, k = zip(*config["features_selection"])
                self.features_selection = SelectKBest(model, k=k)
            else:
                self.features_selection = None
        except Exception as e:
            print_error(
                "cannot create features selection model" + ": " + str(e), num_tabs=1
            )

        # create a list of measurements names (accuracy, precision etc.)
        self.measurements = config["measurements"]

        # create a list of classifiers
        self.classifiers = []
        for classifier in config["classifiers"]:
            try:
                self.classifiers += [classifiers_objects[classifier]]
            except:
                print_error(
                    "cannot create classifiers "
                    + classifier
                    + ": "
                    + classifier
                    + " is not a recognized "
                    "abbreviation of a "
                    "classifier",
                    num_tabs=1,
                )

        # create a classification technique object
        try:
            self.classification_technique = eval(config["classification_technique"])
        except Exception as e:
            print_error(
                "cannot create classification technique object "
                + config["classification_technique"].split("(")[0]
                + ": "
                + str(e),
                num_tabs=1,
            )
            self.classification_technique = CrossValidation()

        # initialize the labels, the extracted feature and the results dict to be None
        self.labels = None
        self.documents = None
        self.classification_results = None

    def __str__(self):
        result = self.experiment_name + ": "
        result += (
            str(len(self.features_extraction_transformers.transformer_list))
            + " transformer, "
        )
        result += str(len(self.classifiers)) + " classifiers, "
        result += str(self.features_selection) if self.features_selection else "no"
        result += " features selection, using " + str(self.classification_technique)
        result += " in " + self.language
        return result

    def get_pipeline(self, clf):
        """
        :param clf: the model of classification
        :return: Pipeline of extraction, selection (if needed) and classification
        """
        lst = [("extraction", self.features_extraction_transformers)]
        if self.features_selection:
            lst.append(("selection", self.features_selection))
        lst.append(("classification", clf))
        return Pipeline(lst)


if __name__ == "__main__":
    experiment = Experiment("../configs/config.json")
    print(experiment)
