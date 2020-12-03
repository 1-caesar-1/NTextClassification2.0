import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
from sklearn.base import TransformerMixin, BaseEstimator

pd.set_option("display.max_colwidth", 200)

import tensorflow_hub as hub
import tensorflow as tf


class ElmoTransfomer(TransformerMixin, BaseEstimator):
    pass


elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


def elmo_vectors(x):
    embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings, 1))

