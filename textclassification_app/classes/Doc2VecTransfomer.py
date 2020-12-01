from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gensim.sklearn_api import D2VTransformer
import re


class Doc2VecTransfomer(BaseEstimator, TransformerMixin):
    """Tokenize input strings based on a simple word-boundary pattern."""

    def __init__(self):
        self.model = D2VTransformer()

    def fit(self, X, y=None):
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        parser = lambda doc: token_pattern.findall(doc)
        d2v_texts = [parser(doc) for doc in X]
        self.model.fit(d2v_texts, y)
        return self

    def transform(self, X):
        ## split on word-boundary. A simple technique, yes, but mirrors what sklearn does to preprocess:
        ## https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/feature_extraction/text.py#L261-L266
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        parser = lambda doc: token_pattern.findall(doc)
        d2v_texts = [parser(doc) for doc in X]
        return self.model.transform(d2v_texts)
