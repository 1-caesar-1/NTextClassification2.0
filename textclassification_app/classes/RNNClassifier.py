from keras.layers import Dropout, Dense, GRU
from keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


class RNNClassifier(KerasClassifier):
    """
    Wraps class to the KerasClassifier for Keras RNN (Recurrent neural network) model
    """

    def __init__(self, dropout=0.5, hidden_layer=3, node=512, **sk_params):
        super().__init__(epochs=5, batch_size=10, verbose=1)
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
        model = Sequential()
        gru_node = 32
        for i in range(0, self.hidden_layer):
            model.add(GRU(gru_node, input_shape=self.shape, return_sequences=True, recurrent_dropout=0.2))
            model.add(Dropout(self.dropout))
        model.add(GRU(gru_node, recurrent_dropout=0.2))
        model.add(Dropout(self.dropout))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.n_classes_, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam')
        model.summary()
        return model
