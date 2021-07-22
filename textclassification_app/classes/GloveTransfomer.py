import spacy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GloveTransfomer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="en_core_web_sm"):
        self._nlp = spacy.load("en_core_web_md")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.concatenate(
            [self._nlp(doc).vector.reshape(1, -1) for doc in X]
        )
