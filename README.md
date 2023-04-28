# Approaching Globally Optimal Energy Efficiency in Interference Networks via Machine Learning

This repository is the source code and data for the paper

B. Peng, K.-L. Besser, R. Raghunath and E. A. Jorswieck, "Approaching Globally Optimal Energy Efficiency in Interference Networks via Machine Learning", IEEE Transactions on Wireless Communications (accepted).

Run `main.py` with the following arguments to train the model:

- `record`: `True` if you want to save the tensorboard log and trained models in a folder named after date and time of the beginning of training, `False` otherwise.
- `pmax`: maximum transmit power.

Run `test.py` to test the saved model on the validation data set.