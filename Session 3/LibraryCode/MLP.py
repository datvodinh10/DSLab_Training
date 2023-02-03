import tensorflow as tf
import numpy as np
import random
# disable eager execution
tf.compat.v1.disable_eager_execution()

class MLP:
    def __init__(self, vocab_size, hidden_size, num_classes=10):
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self.num_classes = num_classes

    def buildGraph(self):
        self._X = tf.compat.v1.placeholder(tf.float32, shape=(None, self._vocab_size))
        self._real_Y = tf.compat.v1.placeholder(tf.int32, shape=(None,))

        weights_1 = tf.compat.v1.get_variable(
            name="weights_input_hidden",
            shape=(self._vocab_size, self._hidden_size),
            initializer=tf.random_normal_initializer(seed=42)
            )
        biases_1 = tf.compat.v1.get_variable(
            name="biases_output_hidden",
            shape=(self._hidden_size),
            initializer=tf.random_normal_initializer(seed=42)
            )
        weights_2 = tf.compat.v1.get_variable(
            name="weights_output_hidden",
            shape=(self._hidden_size, self.num_classes),
            initializer=tf.random_normal_initializer(seed=42)
            )
        biases_2 = tf.compat.v1.get_variable(
            name="biases_input_hidden",
            shape=(self.num_classes),
            initializer=tf.random_normal_initializer(seed=42)
            )

        hidden = tf.matmul(self._X, weights_1) + biases_1
        hidden = tf.sigmoid(hidden)
        logits = tf.matmul(hidden, weights_2) + biases_2

        labels_one_hot = tf.one_hot(
            indices=self._real_Y, 
            depth=self.num_classes,
            dtype=tf.float32
            )
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot,
            logits=logits
            )
        loss = tf.reduce_mean(loss)

        probs = tf.nn.softmax(logits)
        pred_labels = tf.argmax(probs, axis=1)
        pred_labels = tf.squeeze(pred_labels)

        return pred_labels, loss

    def trainer(self, loss, learning_rate):
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op

class DataReader:
    def __init__(self, data_path, batch_size, vocab_size, size=(0, 0.8)):
        self._batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.read().splitlines()
        
        self._data = []
        self._labels = []
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split("<fff>")
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            for token in tokens:
                index, value = int(token.split(":")[0]), \
                               float(token.split(":")[1])
                vector[index] = value
            self._data.append(vector)
            self._labels.append(label)

        start = int(size[0]*len(self._data))
        end = int(size[1]*len(self._data))
        self._data = np.array(self._data[start:end])
        self._labels = np.array(self._labels[start:end])

        self._num_epoch = 0
        self._batch_id = 0

    def nextBatch(self):
        start = self._batch_id + self._batch_size
        end = start + self._batch_size
        self._batch_id += 1

        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch += 1
            self._batch_id += 0
            indices = list(range(len(self._data)))
            random.seed(42)
            random.shuffle(indices)
            self._data, self._labels = self._data[indices], self._labels[indices]
        
        return self._data[start:end], self._labels[start:end]