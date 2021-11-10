from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

class Dataset:

    def __init__(self):
        self.inputs = None
        self.labels = None
        self.feature_names = None

    def _load(self):
        dataset = datasets.load_breast_cancer()
        data = dataset['data']
        target = dataset['target']
        self.feature_names = dataset['feature_names']
        inputs = []
        labels = []
        for input, label in zip(data, target):
            inputs.append(input)
            labels.append(label)
        return np.asarray(inputs), np.asarray(labels)

    def load(self, format, batch_size, num_examples=None):
        inputs, labels = self._load()
        data = train_test_split(inputs, labels, test_size=0.5, shuffle=False)
        if format == 'tf':
            train_dataset = tf.data.Dataset.from_tensor_slices((data[0], data[2]))
            if batch_size is None:
                train_dataset = train_dataset.batch(batch_size=len(data[0]))
            else:
                train_dataset = train_dataset.batch(batch_size=batch_size)
            
            train_dataset = train_dataset.shuffle(buffer_size=100000,
                                              seed=1337,
                                              reshuffle_each_iteration=True)
                                        
            val_dataset = tf.data.Dataset.from_tensor_slices((data[1], data[3]))
            # No batches for validation
            val_dataset = val_dataset.batch(batch_size=len(data[1]))
            return train_dataset, val_dataset
        elif format == 'np':
            return data[0], data[2], data[1], data[3]
        else:
            raise NotImplementedError


if __name__ == '__main__':
    dataset = Dataset()
    train_dataset, test_dataset = dataset.load('tf', batch_size=None)

    for x, y in train_dataset:
        print(y)
        break

