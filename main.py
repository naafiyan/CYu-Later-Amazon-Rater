from preprocess import preprocess
from model import Model
import tensorflow as tf

def train(model, train_texts, train_labels):
    print("Model training...")
    batch_size = 128
    epochs = 1
    # train for 10 epochs
    for i in range(epochs):
        for j in range(batch_size, len(train_texts), batch_size):
            # get batch
            batch_texts = train_texts[j-batch_size:j]
            batch_labels = train_labels[j-batch_size:j]
            
            with tf.GradientTape() as tape:
                probs = model.call(batch_texts)
                loss = model.loss(batch_labels, probs)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print("Epoch: {}, Batch: {}, Loss: {}".format(i, j, loss))

def test(model, test_texts, test_labels):
    # test model
    probs = model.call(test_texts)
    accuracy = model.accuracy(test_labels, probs)
    print("Accuracy: {}".format(accuracy))


def main():
    
    # get data from preprocess
    train_texts, train_labels, test_texts, test_labels, max_length, max_features = preprocess()

    model = Model(max_length, max_features)

    # train model
    train(model, train_texts, train_labels)
    test(model, test_texts, test_labels)
    
    # TODO: Visualize the data

if __name__ == '__main__':
    main()