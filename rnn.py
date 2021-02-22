import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from textclassification_app.rw_files.r_files import read_json_corpus
import numpy as np


le = LabelEncoder()

tt_data = read_json_corpus(r"corpus\english\originals", onlyLabel=True)
random.Random(4).shuffle(tt_data)
# split the labels from the data
X, l = zip(*tt_data)
# encode the labels
y = le.fit_transform(l)
dataset_tt = list(zip(X, y))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
train_dataset = (X_train, y_train)
test_dataset = (X_test, y_test)
VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE
)
encoder.adapt(X_train)
model = tf.keras.Sequential(
    [
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=134,
            # Use masking to handle the variable sequence lengths
            mask_zero=True,
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(134)),
        tf.keras.layers.Dense(134, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"],
)

history = model.fit(np.array(X_train), y_train, epochs=10)

test_loss, test_acc = model.evaluate(test_dataset)

print("Test Loss: {}".format(test_loss))
print("Test Accuracy: {}".format(test_acc))

