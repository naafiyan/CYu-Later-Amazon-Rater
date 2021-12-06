from preprocess import preprocess
from model import Model
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm


def train(model, train_texts, train_labels):
    print("Model training...")
    batch_size = model.batch_size
    epochs = 1
    # train for 10 epochs
    for i in range(epochs):
        print("Num Examples: {}".format(len(train_texts)))

        for j in tqdm(range(batch_size, len(train_texts), batch_size)):
            # get batch
            batch_texts = train_texts[j-batch_size:j]
            batch_labels = train_labels[j-batch_size:j]

            # shuffle data
            indices = tf.range(len(batch_texts))
            shuffled_indices = tf.random.shuffle(indices)
            batch_texts = tf.gather(batch_texts, shuffled_indices)
            batch_labels = tf.gather(batch_labels, shuffled_indices)
            
            with tf.GradientTape() as tape:
                probs = model.call(batch_texts)
                loss = model.loss(batch_labels, probs)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print("Epoch: {}, Batch: {}, Loss: {}".format(i, int(j/batch_size), loss))

def test(model, test_texts, test_labels):
    # testing model

    # batch and get accuracy
    batch_size = model.batch_size

    acc = 0
    steps = 0
    for i in tqdm(range(batch_size, len(test_texts), batch_size)):
        # get batch
        batch_texts = test_texts[i-batch_size:i]
        batch_labels = test_labels[i-batch_size:i]

        # shuffle
        indices = tf.range(len(batch_texts))
        shuffled_indices = tf.random.shuffle(indices)
        batch_texts = tf.gather(batch_texts, shuffled_indices)
        batch_labels = tf.gather(batch_labels, shuffled_indices)

        probs = model.call(batch_texts)
        probs = tf.cast(probs, tf.int32)
        curr_accuracy = model.accuracy(batch_labels, probs)
        acc += curr_accuracy
        steps += 1     
    print("Test Accuracy: {}".format(acc/steps))

def visualize_loss_batch(loss_list):
    """
    Visualize the loss per batch using matplotlib to plot loss_list
    """
    x = [i for i in range(len(loss_list))]
    plt.plot(x, loss_list)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  

def main():
    
    # get data from preprocess
    train_texts, train_labels, test_texts, test_labels, max_length, max_features = preprocess()

    model = Model(max_length, max_features)

    # train model
    train(model, train_texts, train_labels)
    test(model, test_texts, test_labels)
    
    # TODO: Visualize the data
    visualize_loss_batch(model.loss_list)

if __name__ == '__main__':
    main()