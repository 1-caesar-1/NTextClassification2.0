import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import random
from textclassification_app.rw_files.r_files import read_json_corpus
import numpy as np


le = LabelEncoder()
k_fold = KFold()

tt_data = read_json_corpus(r"corpus\english\originals", onlyLabel=True)
random.Random(4).shuffle(tt_data)
# split the labels from the data
X, l = zip(*tt_data)
# encode the labels
y = le.fit_transform(l)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


def get_model(X):
    VOCAB_SIZE = 600
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=VOCAB_SIZE
    )
    encoder.adapt(X)
    model = tf.keras.Sequential(
        [
            encoder,
            tf.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()),
                output_dim=64,
                # Use masking to handle the variable sequence lengths
                mask_zero=True,
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="Adamax", metrics=["accuracy"])
    return model


cv = True
if cv:
    for train, test in k_fold.split(X, y):
        for i in range(20):
            model = get_model(X[train])
            history = model.fit(
                np.array(X[train]),
                y[train],
                epochs=75,
                validation_data=(np.array(X[test]), y[test]),
                validation_steps=10,
            )
            test_loss, test_acc = model.evaluate(np.array(X[test]), y[test])
else:
    model = get_model(X_train)
    history = model.fit(
        np.array(X_train),
        y_train,
        epochs=75,
        validation_data=(np.array(X_test), y_test),
        validation_steps=10,
    )

    test_loss, test_acc = model.evaluate(np.array(X_test), y_test)

print("Test Loss: {}".format(test_loss))
print("Test Accuracy: {}".format(test_acc))

