"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def check_if_real(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return y.dtype == 'float64' or y.dtype == 'int64'


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    # Count the occurrences of each unique value in the series
    value_counts = Y.value_counts()
    
    # Calculate the probabilities of each unique value
    probabilities = value_counts / len(Y) 
    
    # Calculate entropy using the formula: -sum(p_i * log2(p_i))
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy_value


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    # Count the occurrences of each unique value in the series
    value_counts = Y.value_counts()

    # Calculate the probabilities of each unique value
    probabilities = value_counts / len(Y)

    # Calculate gini index using the formula: 1 - sum(p_i^2)
    gini_index_value = 1 - np.sum(probabilities**2)

    return gini_index_value


def mse(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error
    """
    return np.mean((Y - np.mean(Y))**2)


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate Information Gain for both discrete and continuous cases
    """
    if check_if_real(attr):
        # For real-valued attributes, use MSE as the splitting criterion
        total_metric = mse(Y)
        weighted_avg_metric = 0
        unique_values = attr.unique()

        for value in unique_values:
            subset_Y = Y[attr == value]
            weight = len(subset_Y) / len(Y)
            weighted_avg_metric += weight * mse(subset_Y)

    else:
        # For discrete attributes, use entropy
        total_metric = entropy(Y)
        weighted_avg_metric = 0
        unique_values = attr.unique()

        for value in unique_values:
            subset_Y = Y[attr == value]
            weight = len(subset_Y) / len(Y)
            weighted_avg_metric += weight * entropy(subset_Y)

    # Information Gain is the reduction in metric (entropy or MSE)
    info_gain = total_metric - weighted_avg_metric

    return info_gain


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    pass
