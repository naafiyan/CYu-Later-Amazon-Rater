# preprocess returns train_texts, train_labels, test_texts, test_labels, max_length, max_features
import tensorflow as tf
class Model(tf.keras.Model):
    def __init__ (self, max_length, max_features, batch_size, lr):
        super(Model, self).__init__()

        # using sequential
        self.model_call = tf.keras.Sequential([
            tf.keras.layers.Embedding(max_features, 128, batch_input_shape=[batch_size, max_length]),
            tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.loss_list = []

        self.batch_size = batch_size

    def call(self, texts):
        return self.model_call(texts)
    
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