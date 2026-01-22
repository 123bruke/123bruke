import pandas as pd
from tensorflow.keras.layers import Normalization

def load_and_preprocess_data(csv_path):
    dataset = pd.read_csv(csv_path)
    dataset = pd.get_dummies(dataset)
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_labels = train_dataset.pop('expenses')
    test_labels = test_dataset.pop('expenses')
    normalizer = Normalization(axis=-1)
    normalizer.adapt(train_dataset)

    return train_dataset, test_dataset, train_labels, test_labels, normalizer
