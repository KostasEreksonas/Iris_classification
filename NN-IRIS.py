#!/usr/bin/env python3

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

def load_data():
    """
    Load data for the model
    """
    columns = pd.read_csv("data/iris names.txt", names=['names'])
    df = pd.read_csv("data/iris2.data", sep="\t", names=columns["names"])
    return df

def prepare_data(df):
    """
    Prepare data for the model
    """
    X = df.drop("class", axis=1)
    y = pd.get_dummies(df["class"])
    return X, y

def split_data(df):
    """
    Split data into training, validation and testing subsets
    """
    X, y = prepare_data(df)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_model(df):
    """
    Create a Sequential model
    """
    model = keras.Sequential()
    model.add(keras.layers.Dense(16, activation='relu', input_shape=(4,)))
    model.add(keras.layers.Dense(3, activation='softmax'))
    return model

def evaluate(model, X_test, y_test):
    """
    Evaluate the model on the test data
    """
    _, FN, FP, TN, TP, accuracy, precision, recall = model.evaluate(X_test, y_test)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print(f"Test accuracy: {accuracy*100:.2f}%")
    print(f"Test recall: {recall*100:.2f}%")
    print(f"Test precision: {precision*100:.2f}%")
    print(f"TP: {TP}")
    print(f"TN: {TN}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"Test Matthew coefficient: {mcc * 100:.2f}%")

def main():
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    model = create_model(df)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'recall', 'precision', 'TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives']) # Compile the model
    model.fit(X_train, y_train,
            batch_size=12,
            epochs=200,
            validation_data=(X_val, y_val)) # Train the model
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()
