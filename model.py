# preprocess returns train_texts, train_labels, test_texts, test_labels, max_length, max_features
import tensorflow as tf
class Model(tf.keras.Model):
    def __init__ (self, max_length, max_features):
        super(Model, self).__init__()
        self.embedding = tf.keras.layers.Embedding(max_features, 128)
        self.lstm = tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)
        self.dense = tf.keras.layers.Dense(1, activation='softmax')
        
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

    def call(self, texts):
        embed = self.embedding(texts)
        output = self.lstm(embed)
        output = self.dense(output)
        return output
    
    def loss(self, labels, predictions):
        # might want to do sparse softmax cross entropy to get 3 types of sentiment
        return tf.keras.losses.binary_crossentropy(labels, predictions)

    def accuracy(self, labels, predictions):
        # TODO: update to consider 3 different sentiment instead of binary
        return tf.keras.metrics.binary_accuracy(labels, predictions)