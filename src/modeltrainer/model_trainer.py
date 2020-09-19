import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

script_location = os.path.dirname(os.path.realpath(__file__))

def train_and_save_model(dataset,
                         model_filename,
                         scaler_filename,
                         tokenizer_filename,
                         encoder_filename,
                         epochs,
                         batch_size):

    dataset = dataset.filter(['retailerName', 'rawData'])
    dataset.retailerName = dataset.retailerName.str.replace('Boots.*', 'Boots', regex=True)
    cleaned_ds = dataset[dataset.rawData.apply(lambda x: isinstance(x, str))]
    parsed_raw_data = cleaned_ds.rawData.apply(json.loads).apply(pd.Series)
    joined = dataset.join(parsed_raw_data)
    dataset = joined.filter(['retailerName', 'result'])
    parsed_result = dataset.result.apply(pd.Series)
    joined = dataset.join(parsed_result)

    # filter so that we get only features and labels to train the model
    dataset = joined.filter(['establishment', 'retailerName'])
    dataset = dataset[dataset.establishment.apply(lambda x: isinstance(x, str))]
    dataset = dataset[dataset.retailerName.apply(lambda x: isinstance(x, str))]

    # use bag of words model
    train_size = int(len(dataset) * .8)
    train_ocr = dataset['establishment'][:train_size]
    train_tags = dataset['retailerName'][:train_size]
    test_ocr = dataset['establishment'][train_size:]
    test_tags = dataset['retailerName'][train_size:]

    max_words = 1000
    tokenize = Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(train_ocr)
    x_train = tokenize.texts_to_matrix(train_ocr)
    x_test = tokenize.texts_to_matrix(test_ocr)

    # use sklearn utility to convert label strings to numbered index
    encoder = LabelEncoder()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)

    # converts the labels to a one-hot representation
    num_classes = np.max(y_train) + 1
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Inspect the dimensions of our training and test data (this is helpful to debug)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    # build the model
    model = tf.keras.Sequential([
        InputLayer(input_shape=(max_words,)),
        Dense(512, activation='relu', name='hidden_layer'),
        Dense(num_classes, activation='softmax', name='output')
    ])
    lr = 1e-5  # Keep it small when transfer learning
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy'])
    model.run_eagerly = True
    # train the model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.1)
    # evaluate the accuracy
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print('Test accuracy: {0:.2f}%'.format(score[1] * 100))

    text_labels = encoder.classes_
    for i in range(10):
        feature = np.array([x_test[i]])
        prediction = model.predict(feature)
        predicted_label = text_labels[np.argmax(prediction)]
        print(test_ocr.iloc[i][:50], "...")
        print('correct:' + test_tags.iloc[i])
        print("predicted: " + predicted_label + "\n")
    # save the model
    model.save(model_filename)

    scaler = MinMaxScaler()
    scaler.fit(x_train)

    pickle.dump(scaler, open(scaler_filename, 'wb'))
    pickle.dump(tokenize, open(tokenizer_filename, 'wb'))
    pickle.dump(encoder, open(encoder_filename, 'wb'))

    print("Saved model!")


def save_model(args):
    model_filename = args.output_folder + '/model.h5'
    scaler_filename = args.output_folder + '/scaler.p'
    tokenizer_filename = args.output_folder + '/tokenizer.p'
    encoder_filename = args.output_folder + '/encoder.p'
    dataset = pd.read_csv(args.input_dataset)

    print("Training a prediction model with all the \"establishment\" as features and  \"retailerName\" as labels")
    train_and_save_model(dataset,
                         model_filename,
                         scaler_filename,
                         tokenizer_filename,
                         encoder_filename,
                         epochs=args.epochs,
                         batch_size=args.batch_size)


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=4, help="The number of epochs to train")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="The batch size to use for training")
parser.add_argument("-o", "--output_folder", type=str, default=f'{script_location}/../../resources', help="The folder to store the model in")
parser.add_argument("-i", "--input_dataset", type=str, help="The dataset to train with", required=True)
args = parser.parse_args()
save_model(args)

