# Galformer-Improved-Transformer-for-Time-Series-Prediction
Galformer: A Transformer with Generative Decoding and a Hybrid Loss Function for Multi-Step Stock Market Index Prediction(Tensorflow 2.9.1, Python 3.7)

![image](https://s2.loli.net/2024/10/12/boHp4vQFZjJI1lh.png)

We introduce an innovative transformer-based model with generative decoding and a hybrid loss function, named "Galformer," tailored for the multi-step prediction of stock market indices. 

Galformer possesses two distinctive characteristics: 

(1) a novel generative style decoder that predicts long time-series sequences in a single forward operation, significantly boosting the speed of predicting long sequences; 

(2) a novel loss function that combines quantitative error and trend accuracy of the predicted results, providing feedback and optimizing the transformer-based model.

The code model predicts the next 3 days by Adj Close Price of the previous 20 days.

Multi-dimensional and multi-step prediction can be achieved, src_len in the code represents the length of the Encoder input sequence, dec_len represents the length of the Decoder input sequence, tgt_len represents the number of prediction steps.

Code coverage detailed comments, can be modified as needed.

Note that file_path and csv_path are the addresses of the folder where the results are saved, and filename is the address where the historical data set is stored. Please change it as required. l is the file name of each data set, using the for loop can achieve a run to predict multiple data sets, and save the results.

The introduction of its principle, parameter settings and experimental results can be seen in the paper PDF: ''Galformer.pdf''
