import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

from textclassification_app.classes.Experiment import Experiment


def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_masks,
           }, label


@tf.autograph.experimental.do_not_convert
def encode_examples(x, y):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    max_length_test = 500

    ds = zip(x, y)

    for review, label in ds:
        bert_input = tokenizer.encode_plus(
            review,
            add_special_tokens=True,  # add [CLS], [SEP]
            max_length=max_length_test,  # max length of the text that can go to BERT
            pad_to_max_length=True,  # add [PAD] tokens
            return_attention_mask=True,  # add attention mask to not focus on pad tokens
        )

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)


def run_bert(experiment: Experiment):
    # for the convenience of reading
    X = experiment.documents
    y = experiment.labels

    # initialize the result dict
    result = dict()
    for measure in experiment.measurements:
        result[measure] = dict()
        for model in experiment.classifiers:
            result[measure][type(model).__name__] = list()

    for i in range(experiment.classification_technique.iteration):
        xy = list(zip(X, y))
        xy_split = np.array_split(xy, 5)
        for fold in xy_split:
            x_test, y_test = zip(*fold)
            y_test = [int(label) for label in y_test]
            x_train, y_train = [], []
            for other_fold in xy_split:
                if other_fold is not fold:
                    x, y = zip(*other_fold)
                    x_train += list(x)
                    y_train += [int(label) for label in y]

            batch_size = 150
            # train dataset
            ds_train_encoded = encode_examples(x_train, y_train).shuffle(10000).batch(batch_size)

            # test dataset
            ds_test_encoded = encode_examples(x_test, y_test).batch(batch_size)

            # recommended learning rate for Adam 5e-5, 3e-5, 2e-5
            learning_rate = 2e-5
            # we will do just 1 epoch for illustration, though multiple epochs might be better as long as we will not overfit the
            # model
            number_of_epochs = 5
            # model initialization
            model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
            # optimizer Adam
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
            # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
            model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
            bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)

            result["accuracy"]["RNNClassifier"] += bert_history.history["val_accuracy"]

    experiment.classification_results = result
    experiment.general_data["num_of_features"] = 0
