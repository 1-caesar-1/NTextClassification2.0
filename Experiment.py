import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from classification_technique import TrainTest, CrossValidation
from Utils import print_error

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

    features_selection: list[function]
        List of functions that can perform features selection.

    measurements: list[str]
        List of measurement methods for displaying classification results (accuracy, precision etc.)

    classifiers: list[classifier]
        List of objects of classifiers to perform the classification itself (LinearSVC, RandomForestClassifier etc.)

    classification_technique: TrainTest / CrossValidation
        An instance of class TrainTest or CrossValidation that contains information about the classification technique.

    labels: list[int]
        The labels of the documents in this experiment.
        This variable will be None before extracting the features and should contain the labels after extracting.

    extracted_features: np_array
        The features matrix of the documents in this experiment.
        This variable will be None before extracting the features and should contain the features matrix after extracting.
    """
    def __init__(self, path: str, experiment_name: str = 'un-named experiment'):
        # load the JSON file into config
        with open(path, "r", encoding="utf8", errors="replace") as file:
            config = json.load(file)

        # create name to the experiment
        self.experiment_name = experiment_name

        # get the language of the experiment
        self.language = config['language'].lower()

        # create a list of pre-processing functions
        self.preprocessing_functions = []
        for normalization in config['preprocessing']:
            try:
                self.preprocessing_functions += [eval(normalization)]
            except Exception as e:
                print_error('cannot load pre-processing function ' + normalization + ':', end=' ')
                print_error(str(e))

        # create FeatureUnion for all the features transformers
        transformers = []
        counter = 1
        for transformer in config['transformers']:
            try:
                transformers += [(transformer.split('(')[0] + str(counter), eval(transformer))]
                counter += 1
            except Exception as e:
                print_error('cannot create transformer ' + transformer.split('(')[0] + ':', end=' ')
                print_error(str(e))
        self.features_extraction_transformers = FeatureUnion(transformers, n_jobs=-1)

        # create a list of features selection functions
        self.features_selection = []
        for selection in config['features_selection']:
            try:
                self.features_selection += [eval(selection)]
            except Exception as e:
                print_error('cannot load features selection function ' + selection + ':', end=' ')
                print_error(str(e))

        # create a list of measurements names (accuracy, precision etc.)
        self.measurements = config['measurements']

        # create a list of classifiers
        self.classifiers = []
        for classifier in config['classifiers']:
            try:
                self.classifiers += [classifiers_objects[classifier]]
            except:
                print_error('cannot create classifiers ' + classifier + ': ' + classifier + ' is not a recognized '
                                                                                            'abbreviation of a '
                                                                                            'classifier')

        # create a classification technique object
        try:
            self.classification_technique = eval(config['classification_technique'])
        except Exception as e:
            print_error('cannot load features selection function ' + config['classification_technique'].split('(')[0] + ':', end=' ')
            print_error(str(e))
            self.classification_technique = CrossValidation()

        # initialize the labels and the extracted feature to be None
        self.labels = None
        self.extracted_features = None


if __name__ == '__main__':
    experiment = Experiment('test_config.json')
    print(bool)
