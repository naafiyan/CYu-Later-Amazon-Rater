# preprocess returns train_texts, train_labels, test_texts, test_labels, max_length, max_features
import tensorflow as tf
class Model(tf.keras.Model):
    def __init__ (self, max_length, max_features, batch_size, lr):
        super(Model, self).__init__()
        self.embedding = tf.keras.layers.Embedding(max_features, 128)
        self.lstm_1 = tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(1, activation='sigmoid')
        
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.loss_list = []

        self.batch_size = batch_size

    def call(self, texts):
        # embedding
        embeddings = self.embedding(texts)

        # lstm
        lstm_out = self.lstm_1(embeddings)
        lstm_out = self.lstm_2(lstm_out)

        # dense
        dense_out = self.dense_1(lstm_out)
        dense_out = self.dense_2(dense_out)

        return dense_out
    
    def loss(self, labels, predictions):
        # might want to do sparse softmax cross entropy to get 3 types of sentiment
        labels = tf.reshape(labels, (-1, 1))
        preds = tf.reshape(predictions, (-1, 1))
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, preds, from_logits=True))
        # append loss to list so that we can visualize it
        self.loss_list.append(loss)
        return loss

    def accuracy(self, labels, predictions):
        # TODO: update to consider 3 different sentiment instead of binary
        return tf.reduce_mean(tf.keras.metrics.binary_accuracy(labels, predictions))