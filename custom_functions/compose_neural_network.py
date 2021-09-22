"""Neural Network Model Generators.

This script assists in generating multiple iterations
of a neural network model for comparison across differing criteria.

"""

# Imports
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
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
    # 'categorical variables' are of datatype 'object' (strings)
    categorical_variables = list(
        nn_data.dtypes[
            nn_data.dtypes == 'object'
        ].index
    )

    if debug: print(f'categorical_variables:\n{categorical_variables}\n')
    
    enc = OneHotEncoder(sparse=False)
    encoded_data = enc.fit_transform(nn_data[categorical_variables])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns = enc.get_feature_names(categorical_variables)
    )

    print(f'Categorical Variables:\n{categorical_variables}\n')

    if debug: print(f'\nencoded_df datatypes:\n{encoded_df.dtypes}\n')
    if debug: print(f'encoded_df data:\n{encoded_df}\n')

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

    print(f'Combined Encoded DataFrame:\n{combined_encoded_df}')

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
        features, target (DataFrame(s)): Feature and target datasets
    """
    target = enc_data[target_name]
    features = enc_data.drop(columns=[target_name])

    if debug: print(f'Target Dataset (y):\n{target}\n')
    if debug: print(f'Features Dataset:\n{features}')

    return target, features

def scale_features_data(X_train, X_test):
    """Fit a StandardScaler instance and fit
    to features training dataset.

    Args:
        X_train (DataFrame): Features used to fit the scaler
        X_test  (DataFrame): Features used to test the neural net
    
    Returns:
        X_train_scaled (Array): Scaled training features dataset
        X_test_scaled (Array): Scaled testing features dataset
    """
    scaler = StandardScaler()

    # Fit the scaler to the features training dataset
    X_scaler = scaler.fit(X_train)

    # Scale 'X_train' and 'X_test' datasets to the fit X_scaler
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def binary_class_neural_net(
        number_input_features,
        hidden_node_layers,
        output_layer_activation):
    """Generate, populate, and run a
    binary classification neural network model.

    Args:
        number_input_features (Int): Number of inputs to the model
        hidden_node_layers (List of Lists): Number of hidden node layers
        output_layer_activation (Str): Output layer activation function

    Returns:
        neural_net (Object): The compiled model ready to be fit
    """
    nn = Sequential()

    # Add desired number of hidden node layers
    for idx, hidden_node_layer in enumerate(hidden_node_layers):
        if idx == 0:
            nn.add(Dense(
                units=hidden_node_layer[0],
                input_dim=number_input_features,
                activation=hidden_node_layer[1]
                )
            )
        else:
            nn.add(Dense(
                units=hidden_node_layer[0],
                activation=hidden_node_layer[1]
                )
            )
    
    # Add the output layer
    nn.add(Dense(
        units=1,
        activation=output_layer_activation
        )
    )

    nn.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    nn.summary()
    
    return nn
