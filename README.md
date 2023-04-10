# Ensemble Learning Example
This repository contains an example of ensemble learning using scikit-learn in Python. In this example, we use a combination of k-nearest neighbors regression, decision tree regression, and ridge regression as base learners. The meta-learner is linear regression.

The main script can be found in ensemble_stacking.py. The script consists of four main sections:

1. Defining the base learners and meta-learner.
2. Training the base learners on the training set and generating meta data from their predictions.
3. Testing the base learners on the test set and generating test meta data from their predictions.
4. Training the meta-learner on the meta data and evaluating the ensemble on the test meta data.

# Requirements
The script requires the following Python packages:

* scikit-learn
* numpy

# Data
The example uses the diabetes dataset from scikit-learn. The dataset contains 442 instances and 10 features. The target variable is a quantitative measure of disease progression one year after baseline.
