from keras_malicious_url_detector.library.lstm_embed import LstmEmbedPredictor
from keras_malicious_url_detector.library.utility.url_data_loader import load_url_data
import numpy as np
from keras_malicious_url_detector.library.utility.text_model_extractor import extract_text_model
from keras_malicious_url_detector.library.utility.plot_utils import plot_and_save_history


def main():

    random_state = 42
    np.random.seed(random_state)

    data_dir_path = './data'
    model_dir_path = './models'
    report_dir_path = './reports'

    url_data = load_url_data(data_dir_path)

    text_model = extract_text_model(url_data['text'])

    batch_size = 64
    epochs = 50

    classifier = LstmEmbedPredictor()

    history = classifier.fit(text_model=text_model,
                             model_dir_path=model_dir_path,
                             url_data=url_data, batch_size=batch_size, epochs=epochs)

    plot_and_save_history(history, LstmEmbedPredictor.model_name,
                          report_dir_path + '/' + LstmEmbedPredictor.model_name + '-history.png')


if __name__ == '__main__':
    main()
