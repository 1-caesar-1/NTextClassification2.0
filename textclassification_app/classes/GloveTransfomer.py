import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
import en_core_web_md


class GloveTransfomer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="en_core_web_md"):
        self._nlp = en_core_web_md.load()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.concatenate([self._nlp(doc).vector.reshape(1, -1) for doc in X])

