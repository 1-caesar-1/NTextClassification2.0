import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import FeatureUnion

from Utils import print_error

def normal(file):
    pass

class Test:
    def __init__(self, path: str, test_name: str = ''):
        # load the JSON file into config
        with open(path, "r", encoding="utf8", errors="replace") as file:
            config = json.load(file)

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

        # create a list of pre-processing functions
        self.preprocessing_functions = []
        for normalization in config['preprocessing']:
            try:
                self.preprocessing_functions += [eval(normalization)]
            except Exception as e:
                print_error('cannot load pre-processing function ' + normalization + ':', end=' ')
                print_error(str(e))



if __name__ == '__main__':
    test = Test('test.json')
    x = 3
