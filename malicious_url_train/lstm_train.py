from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 64
EPOCHS = 20
VERBOSE = 1
NB_LSTM_CELLS = 256
NB_DENSE_CELLS = 256


def make_lstm_model(num_input_tokens):
    model = Sequential()
    model.add(LSTM(NB_LSTM_CELLS, input_shape=(None, num_input_tokens), return_sequences=False, return_state=False, dropout=0.2))
    model.add(Dense(NB_DENSE_CELLS))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    np.random.seed(42)

    data_dir_path = './data'
    model_dir_path = './models'
    model_name = 'lstm'
    weight_file_path = model_dir_path + '/' + model_name + '-weights.h5'
    architecture_file_path = model_dir_path + '/' + model_name + '-architecture.json'

    url_data = pd.read_csv(data_dir_path + os.path.sep + 'URL.txt', sep=',')
    url_data.columns = ['text', 'label']
    print(url_data.head())
    char2idx = dict()
    max_url_seq_length = 0
    for url in url_data['text']:
        max_url_seq_length = max(max_url_seq_length, len(url))
        for c in url:
            if c not in char2idx:
                char2idx[c] = len(char2idx)
    num_input_tokens = len(char2idx)
    idx2char = dict([(idx, c) for c, idx in char2idx.items()])

    config = dict()
    config['num_input_tokens'] = num_input_tokens
    config['char2idx'] = char2idx
    config['idx2char'] = idx2char
    config['max_url_seq_length'] = max_url_seq_length

    np.save(model_dir_path + '/' + model_name + '-config.npy', config)

    data_size = url_data.shape[0]
    X = np.zeros(shape=(data_size, max_url_seq_length, num_input_tokens))
    Y = np.zeros(shape=(data_size, 2))
    for i in range(data_size):
        url = url_data['text'][i]
        label = url_data['label'][i]
        for idx, c in enumerate(url):
            X[i, idx, char2idx[c]] = 1
        Y[i, label] = 1

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = make_lstm_model(num_input_tokens)
    open(architecture_file_path, 'w').write(model.to_json())

    checkpoint = ModelCheckpoint(weight_file_path)
    model.fit(x=Xtrain, y=Ytrain, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_data=(Xtest, Ytest), callbacks=[checkpoint])
    model.save_weights(weight_file_path)


if __name__ == '__main__':
    main()
