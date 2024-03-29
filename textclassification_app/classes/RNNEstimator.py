import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin


class RNNEstimator(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, epochs=50, validation_steps=10, vocab_size=1000):
        self.epochs = epochs
        self.validation_steps = validation_steps
        self.vocab_size = vocab_size
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=self.vocab_size,
                    output_dim=32,
                    # Use masking to handle the variable sequence lengths
                    mask_zero=True,
                ),

                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        self.model.compile(
            loss="binary_crossentropy", optimizer="Adamax", metrics=["accuracy"]
        )

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        self.model.fit(X, y, epochs=self.epochs, validation_steps=self.validation_steps)
        return self

    def predict(self, X, y=None):
        return (self.model.predict(X) > 0.5).astype("int32")

    def predict_proba(self, X, y=None):

        return self.model.predict_proba(X)

