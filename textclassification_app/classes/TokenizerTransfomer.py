from sklearn.base import TransformerMixin, BaseEstimator, MetaEstimatorMixin
import re


class TokenizerTransfomer(BaseEstimator, MetaEstimatorMixin):
    """Tokenize input strings based on a simple word-boundary pattern."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ## split on word-boundary. A simple technique, yes, but mirrors what sklearn does to preprocess:
        ## https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/feature_extraction/text.py#L261-L266
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        parser = lambda doc: token_pattern.findall(doc)
        return [parser(doc) for doc in X]

