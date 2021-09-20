"""Neural Network Model Generators.

This script assists in generating multiple iterations
of a neural network model for comparison across differing criteria.

"""

# Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def encode_categorical_variables(nn_data,debug=False):
    """Encode categorical variables for neural network model.

    Encode neural network model categorical variables,
    and combine with numerical variables into a new DataFrame.

    Args:
        nn_data (DataFrame): Dataset for the neural net model

    Returns:
        New DataFrame with encoded categorical variables

    """

    # 'categorical variables' = datatype 'object' (strings)
    categorical_variables = list(
        nn_data.dtypes[
            nn_data.dtypes == 'object'
        ].index
    )

    if debug: print(categorical_variables)
    
    enc = OneHotEncoder(sparse=False)
    encoded_data = enc.fit_transform(nn_data[categorical_variables])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns = enc.get_feature_names(categorical_variables)
    )

    if debug: print(f'encoded_df datatypes:\n{encoded_df.dtypes}')
    if debug: print(f'encoded_df data:\n{encoded_df}')

    numerical_variables_df = nn_data.drop(columns=categorical_variables)

    if debug: print(f'numerical_variables_df:\n{numerical_variables_df}')
    
    # Concat categorical variables to numerical variables
    combined_encoded_df = pd.concat(
        [
            numerical_variables_df,
            encoded_df
        ],
        axis=1
    )

    if debug: print(f'combined_encoded_df:\n{combined_encoded_df}')

    return combined_encoded_df


def create_features_and_target(enc_data,target_name,debug=False):
    """Define encoded DataFrame features and target.

    This function accepts a single string for target_name
    to assign to the target dataset and assigns all remaining
    columns to the features dataset. 

    Args:
        enc_data (DataFrame): Preprocess data for the neural net model
        target_name (string): Name of the column for target set (y)

    Returns:
        X, y (list of DataFrames): Feature and target datasets

    """

    y = enc_data[target_name]
    X = enc_data.drop(columns=[target_name])

    if debug: print(f'Target Dataset (y):\n{y}')
    if debug: print(f'Features Dataset:\n{X}')

