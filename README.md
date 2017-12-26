# keras-malicious-url-detector

Malicious URL detector using char-level recurrent neural networks with Keras

The purpose of this project is to study malicious url detector that does not rely on any prior knowledge about urls

The training data is from this [link](https://github.com/vaseem-khan/URLcheck), can be found in "malicious_url_train/data/URL.txt"

# Deep Learning Models

The following deep learning models have been implemented and studied:

* LSTM: this approach uses LSTM recurrent networks for classifier with categorical cross entropy loss function
    * training: malicious_url_train/lstm_train.py (one-hot encoding)
    * predictor: malicious_url_predict/lstm_predict.py (one-hot encoding)
    * training: malicious_url_train/lstm_embed_train.py (word embedding)
    * predictor: malicious_url_predict/lstm_embed_predict.py (word embedding)
    
* CNN + LSTM: this approach uses CNN + LSTM recurrent networks for classifier with categorical cross entropy loss function
    * training: malicious_url_train/cnn_lstm_train.py 
    * predictor: malicious_url_predict/cnn_lstm_predict.py 
    
* Bidirectional LSTM: this approach uses Bidirectional LSTM recurrent networks for classifier with categorical cross entropy loss function
    * training: malicious_url_train/bidirectional_lstm_train.py 
    * predictor: malicious_url_predict/bidirectional_lstm_predict.py 

# Performance

Currently the bidirectional LSTM gives the best performance, with 75% - 80% accuracy after 30 to 40 epochs of training   

Below is the training history (loss and accuracy) for the bidirectional LSTM:

![bidirection-lstm-history](/malicious_url_train/reports/bidirectional-lstm-history.png)

# Issues

* Currently the data size of the urls is small
* Class imbalances - the URL.txt contains class imbalances (more 0 than 1), ideally the problem should be an outlier 
or anomaly detection problem. To handle the class imabalances, currently a resampling method is used to make sure that 
there are more or less equal number of each classes