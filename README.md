# btc-predictor
Testing some RNN's architecture to predict BTC price, given the history of prices.

**Remarks** : price history isn't sufficient to make any good predictions. Every models (GRU, LSTM) end up learning the identity function (just repeating the last known price). I have tried modifying the MSE error function, by penalizing the identity function but the model ends up learning some basic function close to identity.

In order to make the predictions more relevant it would probably need more complex data, for example analyzing some news feed from https://cryptopanic.com/ with NLP.
