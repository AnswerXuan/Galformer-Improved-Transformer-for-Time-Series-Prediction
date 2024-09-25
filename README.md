# Galformer-Improved-Transformer-for-time-series-Prediction
Galformer: A Transformer with Generative Decoding and a Hybrid Loss Function for Multi-Step Stock Market Index Prediction(Tensorflow 2.9.1, Python 3.7)

We introduce an innovative transformer-based model with generative decoding and a hybrid loss function, named "Galformer," tailored for the multi-step prediction of stock market indices. Galformer possesses two distinctive characteristics: (1) a novel generative style decoder that predicts long time-series sequences in a single forward operation, significantly boosting the speed of predicting long sequences; (2) a novel loss function that combines quantitative error and trend accuracy of the predicted results, providing feedback and optimizing the transformer-based model.

The code model is based on LSTM neural network optimized by improved particle swarm optimization algorithm, which predicts the next day by Adj Close Price of the first 20 days.

Multi-dimensional and multi-step prediction can be achieved, seq_len in the code represents the length of the input sequence, mulpre represents the number of prediction steps.




Code coverage detailed comments, can be modified as needed.

Note that file_path and csv_path are the addresses of the folder where the results are saved, and filename is the address where the historical data set is stored. Please change it as required. l is the file name of each data set, using the for loop can achieve a run to predict multiple data sets, and save the results.

The introduction of its principle can be seen in the paper: ''PREDICTION OF THE STOCK ADJUSTED CLOSING PRICE BASED ON IMPROVED PSO-LSTM NEURAL NETWORK''
