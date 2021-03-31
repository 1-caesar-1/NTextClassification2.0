
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

from textclassification_app.utils import print_message


class RNNClassifier(KerasClassifier):
    """
    Wraps class to the KerasClassifier for Keras RNN (Recurrent neural network) model
    """

    def __init__(self, dropout=0.5, hidden_layer=3, node=512, **sk_params):
        super().__init__(epochs=50, verbose=1)
        self.shape = 0
        self.dropout = dropout
        self.hidden_layer = hidden_layer
        self.node = node

    def fit(self, x, y, **kwargs):
        try:
            x = x.toarray()
        except Exception as ex:
            pass
        x = x[:, :, None]
        self.shape = x.shape[1:]
        return super(RNNClassifier, self).fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        try:
            x = x.toarray()
        except Exception as ex:
            pass
        x = x[:, :, None]
        return super(RNNClassifier, self).predict(x, **kwargs)

    def predict_proba(self, x, **kwargs):
        try:
            x = x.toarray()
        except Exception as ex:
            pass
        x = x[:, :, None]
        return super(RNNClassifier, self).predict_proba(x, **kwargs)

    def score(self, x, y, **kwargs):
        try:
            x = x.toarray()
        except Exception as ex:
            pass
        x = x[:, :, None]
        return super(RNNClassifier, self).score(x, y, **kwargs)

    def __call__(self, *args, **kwargs):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=500,
                    output_dim=32,
                    # Use masking to handle the variable sequence lengths
                    mask_zero=True,
                ),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        model.compile(
            loss="binary_crossentropy", optimizer="Adamax", metrics=["accuracy"]
        )

        model.summary()
        return model
