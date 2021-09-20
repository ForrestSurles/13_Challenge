"""Neural Network Model Generators.

This script assists in generating multiple iterations
of a neural network model for comparison across differing criteria.

"""

# Imports
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def encode_categorical_variables(nn_data):
    """Encodes categorical variables into a list.

    Args:
        nn_data (DataFrame): The dataset for the neural net.

    Returns:
        A list of categorical variables. 

    """
    # 'categorical variables' = datatype 'object' (strings)
    categorical_variables = list(
        nn_data.dtypes[
            nn_data.dtypes == 'object'
        ].index
    )

    # display the categorical variables list
    print(categorical_variables)
    return categorical_variables