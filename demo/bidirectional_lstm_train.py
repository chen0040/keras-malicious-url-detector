from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, SpatialDropout1D, Bidirectional, Embedding
from keras.layers.recurrent import LSTM
from malicious_url_train.url_data_loader import load_url_data
import numpy as np
from sklearn.model_selection import train_test_split
from malicious_url_train.utils import plot_and_save_history




def make_bidirectional_lstm_model(num_input_tokens, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=num_input_tokens, output_dim=EMBEDDING_SIZE, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, input_shape=(max_len, EMBEDDING_SIZE))))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    np.random.seed(42)

    data_dir_path = './data'
    model_dir_path = './models'
    report_dir_path = './reports'

    model_name = 'bidirectional-lstm'
    weight_file_path = model_dir_path + '/' + model_name + '-weights.h5'
    architecture_file_path = model_dir_path + '/' + model_name + '-architecture.json'

    url_data = load_url_data(data_dir_path)

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
    X = np.zeros(shape=(data_size, max_url_seq_length))
    Y = np.zeros(shape=(data_size, 2))
    for i in range(data_size):
        url = url_data['text'][i]
        label = url_data['label'][i]
        for idx, c in enumerate(url):
            X[i, idx] = char2idx[c]
        Y[i, label] = 1

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = make_bidirectional_lstm_model(num_input_tokens, max_url_seq_length)
    open(architecture_file_path, 'w').write(model.to_json())

    checkpoint = ModelCheckpoint(weight_file_path, save_best_only=True)

    history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE,
                        validation_data=(Xtest, Ytest), callbacks=[checkpoint])

    model.save_weights(weight_file_path)
    plot_and_save_history(history, model_name, report_dir_path + '/' + model_name + '-history.png')


if __name__ == '__main__':
    main()
