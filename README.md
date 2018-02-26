# keras-malicious-url-detector

Malicious URL detector using char-level recurrent neural networks with Keras

The purpose of this project is to study malicious url detector that does not rely on any prior knowledge about urls

The training data is from this [link](https://github.com/vaseem-khan/URLcheck), can be found in "demo/data/URL.txt"

# Deep Learning Models

The following deep learning models have been implemented and studied:

* LSTM: this approach uses LSTM recurrent networks for classifier with categorical cross entropy loss function
    * training: [demo/lstm_train.py](demo/lstm_train.py) (one-hot encoding)
    * predictor: [demo/lstm_predict.py](demo/lstm_predict.py) (one-hot encoding)
    * training: [demo/lstm_embed_train.py](demo/lstm_embed_train.py) (word embedding)
    * predictor: [demo/lstm_embed_predict.py](demo/lstm_embed_predict.py) (word embedding)
    
* CNN + LSTM: this approach uses CNN + LSTM recurrent networks for classifier with categorical cross entropy loss function
    * training: [demo/cnn_lstm_train.py](demo/cnn_lstm_train.py) 
    * predictor: [demo/cnn_lstm_predict.py](demo/cnn_lstm_predict.py) 
    
* Bidirectional LSTM: this approach uses Bidirectional LSTM recurrent networks for classifier with categorical cross entropy loss function
    * training: [demo/bidirectional_lstm_train.py](demo/bidirectional_lstm_train.py) 
    * predictor: [demo/bidirectional_lstm_predict.py](demo/bidirectional_lstm_predict.py) 
    
# Usage

To run the training on Bidirectional LSTM:

```bash
cd demo
python bidirectional_lstm_train.py
```

Below is the code in [bidirectional_lstm_train.py](demo/bidirectional_lstm_train.py):

```python
from keras_malicious_url_detector.library.bidirectional_lstm import BidirectionalLstmEmbedPredictor
from malicious_url_train.url_data_loader import load_url_data
import numpy as np
from keras_malicious_url_detector.library.utility.text_model_extractor import extract_text_model
from malicious_url_train.utils import plot_and_save_history


def main():

    random_state = 42
    np.random.seed(random_state)

    data_dir_path = './data'
    model_dir_path = './models'
    report_dir_path = './reports'

    url_data = load_url_data(data_dir_path)

    text_model = extract_text_model(url_data['text'])

    batch_size = 64
    epochs = 30

    classifier = BidirectionalLstmEmbedPredictor()

    history = classifier.fit(text_model=text_model,
                             model_dir_path=model_dir_path,
                             url_data=url_data, batch_size=batch_size, epochs=epochs)

    plot_and_save_history(history, BidirectionalLstmEmbedPredictor.model_name,
                          report_dir_path + '/' + BidirectionalLstmEmbedPredictor.model_name + '-history.png')


if __name__ == '__main__':
    main()
```

After the training, the trained models are saved in the [demo/models](demo/models) folder.

To test the trained model,run:

```bash
cd demo
python bidirectional_lstm_predict.py
```

Below is the code in [bidrectional_lstm_predict.py](demo/bidirectional_lstm_predict.py):

```python
from keras_malicious_url_detector.library.bidirectional_lstm import BidirectionalLstmEmbedPredictor
from keras_malicious_url_detector.library.utility.url_data_loader import load_url_data


def main():

    data_dir_path = './data'
    model_dir_path = './models'

    predictor = BidirectionalLstmEmbedPredictor()
    predictor.load_model(model_dir_path)

    url_data = load_url_data(data_dir_path)
    count = 0
    for url, label in zip(url_data['text'], url_data['label']):
        predicted_label = predictor.predict(url)
        print('predicted: ' + str(predicted_label) + ' actual: ' + str(label))
        count += 1
        if count > 20:
            break


if __name__ == '__main__':
    main()
```

# Performance

Currently the bidirectional LSTM gives the best performance, with 75% - 80% accuracy after 30 to 40 epochs of training   

Below is the training history (loss and accuracy) for the bidirectional LSTM:

![bidirection-lstm-history](/demo/reports/bidirectional-lstm-history.png)

# Issues

* Currently the data size of the urls is small
* Class imbalances - the URL.txt contains class imbalances (more 0 than 1), ideally the problem should be an outlier 
or anomaly detection problem. To handle the class imabalances, currently a resampling method is used to make sure that 
there are more or less equal number of each classes