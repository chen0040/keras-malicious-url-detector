import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, Activation
from sklearn.model_selection import train_test_split

from malicious_url_train.lstm_train import make_lstm_model


NB_LSTM_CELLS = 256
NB_DENSE_CELLS = 256


def make_lstm_model(num_input_tokens):
    model = Sequential()
    model.add(LSTM(NB_LSTM_CELLS, input_shape=(None, num_input_tokens), return_sequences=False, return_state=False, dropout=0.2))
    model.add(Dense(NB_DENSE_CELLS))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class LstmPredictor(object):

    model_name = 'lstm'

    def __init__(self):
        self.model = None
        self.num_input_tokens = None
        self.idx2char = None
        self.char2idx = None
        self.max_url_seq_length = None

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + LstmPredictor.model_name + '-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + LstmPredictor.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + LstmPredictor.model_name + '-architecture.json'

    def load_model(self, model_dir_path):
        config_file_path = self.get_config_file_path(model_dir_path)
        weight_file_path = self.get_weight_file_path(model_dir_path)

        config = np.load(config_file_path).item()
        self.num_input_tokens = config['num_input_tokens']
        self.max_url_seq_length = config['max_url_seq_length']
        self.idx2char = config['idx2char']
        self.char2idx = config['char2idx']

        self.model = make_lstm_model(self.num_input_tokens)
        self.model.load_weights(weight_file_path)

    def predict(self, url):
        data_size = 1
        X = np.zeros(shape=(data_size, self.max_url_seq_length, self.num_input_tokens))
        for idx, c in enumerate(url):
            if c in self.char2idx:
                X[0, idx, self.char2idx[c]] = 1
        predicted = self.model.predict(X)[0]
        predicted_label = np.argmax(predicted)
        return predicted_label

    def extract_training_data(self, url_data):
        data_size = url_data.shape[0]
        X = np.zeros(shape=(data_size, self.max_url_seq_length))
        Y = np.zeros(shape=(data_size, 2))
        for i in range(data_size):
            url = url_data['text'][i]
            label = url_data['label'][i]
            for idx, c in enumerate(url):
                X[i, idx] = self.char2idx[c]
            Y[i, label] = 1

        return X, Y

    def fit(self, text_model, url_data, model_dir_path, batch_size=None, epochs=None,
            test_size=None, random_state=None):
        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 50
        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = 42

        self.num_input_tokens = text_model['num_input_tokens']
        self.char2idx = text_model['char2idx']
        self.idx2char = text_model['idx2char']
        self.max_url_seq_length = text_model['max_url_seq_length']

        np.save(self.get_config_file_path(model_dir_path), text_model)

        weight_file_path = self.get_weight_file_path(model_dir_path)

        checkpoint = ModelCheckpoint(weight_file_path)

        X, Y = self.extract_training_data(url_data)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        self.model = make_lstm_model(self.num_input_tokens)

        with open(self.get_architecture_file_path(model_dir_path), 'wt') as f:
            f.write(self.model.to_json())

        history = self.model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                                 validation_data=(Xtest, Ytest), callbacks=[checkpoint])

        self.model.save_weights(weight_file_path)

        np.save(model_dir_path + '/' + LstmPredictor.model_name + '-history.npy', history.history)

        return history
