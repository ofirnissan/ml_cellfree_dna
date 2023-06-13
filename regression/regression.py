import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt


# GLOBALS:
TRAIN_DATA_PATH = r"C:\Users\ofirn\PycharmProjects\project1\venv\Scripts\Machine_Learning\DNABERT\examples\sample_data\ft\6_age\train.tsv"
DEV_DATA_PATH = r"C:\Users\ofirn\PycharmProjects\project1\venv\Scripts\Machine_Learning\DNABERT\examples\sample_data\ft\6_age\dev.tsv"
PREDICTIONS_OUTPUT = r"C:\Users\ofirn\PycharmProjects\project1\venv\Scripts\Machine_Learning\ml_cellfree_dna\regression\simple_regression_res.csv"
# Set vocabulary to represent each subsequence by its index
VOCAB_PATH = r"C:\Users\ofirn\PycharmProjects\project1\venv\Scripts\Machine_Learning\DNABERT\examples\sample_data\ft\6_age\vocab.txt"
df_vocab = pd.read_csv(VOCAB_PATH, sep=' ')
VOCAB_DICT = {value:i for i, value in enumerate(df_vocab["vocab"])}


def parse_sequence_from_data(path):
    df_data = pd.read_csv(path, sep='\t', header=0)
    mat = []
    for i in range(len(df_data["sequence"])):
        seq = []
        cur = df_data["sequence"][i].split()
        for j in range(len(cur)):
            seq.append(VOCAB_DICT[cur[j]] / 4096)
        mat.append(seq)

    X_numpy = np.array(mat, dtype=np.float32)
    y_numpy = df_data["label"].to_numpy(dtype=np.float32)
    # cast to float Tensor
    X = torch.from_numpy(X_numpy.astype(np.float32))
    y = torch.from_numpy(y_numpy.astype(np.float32))
    y = y.view(y.shape[0], 1)
    return X, y


def exist_linear_model(features_num):
    input_size = features_num
    output_size = 1
    model = nn.Linear(input_size, output_size)
    return model


def train_model(model, criterion, optimizer, training_data, labels):
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass and loss
        y_predicted = model(training_data)
        loss = criterion(y_predicted, labels)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(loss)
            print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')


def main():
    # get tensors training data
    training_data, labels = parse_sequence_from_data(TRAIN_DATA_PATH)

    samples_num, features_num = training_data.shape
    # Linear model:
    model = exist_linear_model(features_num)

    # Set loss and optimizer
    learning_rate = 0.01
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training
    train_model(model, criterion, optimizer, training_data, labels)

    # Get prediction on test data
    X, y = parse_sequence_from_data(DEV_DATA_PATH)
    predicted = model(X)
    # print loss:
    print("TEST MSE LOSS: " + str(criterion(predicted, y).item()))
    y = y.numpy()
    predicted = predicted.detach().numpy()
    # plot as data frame table
    predicted = np.squeeze(np.asarray(predicted))
    print(f"Prediction Mean: {np.average(predicted)}")
    y = np.squeeze(np.asarray(y))
    df = pd.DataFrame({'predicted':predicted, 'y':y})
    df.to_csv(PREDICTIONS_OUTPUT)





if __name__ == "__main__":
    main()