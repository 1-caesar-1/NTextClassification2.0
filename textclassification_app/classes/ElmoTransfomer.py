import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

pd.set_option("display.max_colwidth", 200)

import tensorflow_hub as hub
import tensorflow as tf


class ElmoTransfomer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        return self

    def transform(self, X):
        """Given a list of original data, return a list of feature vectors."""

        return np.concatenate([self.elmo_vectors(i) for i in X], axis=0)

    def elmo_vectors(self, x):
        embeddings = self.elmo([x], signature="default", as_dict=True)["elmo"]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            # return average of ELMo features
            return sess.run(tf.reduce_mean(embeddings, 1))

