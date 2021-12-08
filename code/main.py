from preprocess import preprocess, get_all_data
from glove import glove_embed
from model import Model
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
from os.path import exists


def train(model, train_texts, train_labels, epochs):
    print("Model training...")
    batch_size = model.batch_size
    # train for 10 epochs
    print("Num Examples: {}".format(len(train_texts)))
    for i in range(epochs):

        for j in tqdm(range(batch_size, len(train_texts), batch_size), desc="Epoch {}".format(i)):
            # get batch
            batch_texts = train_texts[j-batch_size:j]
            batch_labels = train_labels[j-batch_size:j]

            # shuffle data
            indices = tf.range(len(batch_texts))
            shuffled_indices = tf.random.shuffle(indices)
            batch_texts = tf.gather(batch_texts, shuffled_indices)
            batch_labels = tf.gather(batch_labels, shuffled_indices)
            
            with tf.GradientTape() as tape:
                preds = model.call(batch_texts)
                loss = model.loss(batch_labels, preds)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print("Batch: {}, Loss: {}".format(int(j/batch_size), tf.reduce_sum(loss)))

def test(model, test_texts, test_labels):
    # testing model
    print("Testing model...")
    # batch and get accuracy
    batch_size = model.batch_size

    acc = 0
    steps = 0
    for i in tqdm(range(batch_size, len(test_texts), batch_size), desc="Testing"):
        # get batch
        batch_texts = test_texts[i-batch_size:i]
        batch_labels = test_labels[i-batch_size:i]

        # shuffle
        indices = tf.range(len(batch_texts))
        shuffled_indices = tf.random.shuffle(indices)
        batch_texts = tf.gather(batch_texts, shuffled_indices)
        batch_labels = tf.gather(batch_labels, shuffled_indices)

        """
        # TODO: Fix this
        The issue here is that probabilities all get rounded up to 1
        That could be an issue with softmax and maybe we should use sigmoid or ReLU instead
        It could also be an issue with the loss function
        Maybe we need to somehow use one_hot encoding for the labels
        """
        probs = model.call(batch_texts)
        # not sure if this rounding is needed?
        probs = tf.round(probs)
        # but if i dont round then all the values get truncated to 0 when we cast here
        # but we need to cast for the accuracy function to work
        # sigmoid is centred around 0.5 with range [0,1]
        probs = tf.cast(probs, tf.int32)
        curr_accuracy = model.accuracy(batch_labels, probs)
        print("Accuracy: {}".format(curr_accuracy))
        acc += curr_accuracy
        steps += 1     
    print("Test Accuracy: {}".format(acc/max(steps, 1)))

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

def visualize_avg_loss_epoch(loss_list, batch_size):
    """
    Visualize the average loss per epoch using matplotlib to plot loss_list against epoch
    """
    pass

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='File Name')
    parser.add_argument('--num_examples', type=int, default=10000, help='Number of examples to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')    
    parser.add_argument('--sentiment_threshold', type=float, default=3.5, help='Sentiment threshold')
    parser.add_argument('--load_weights', type=bool, default=False, help='Load weights')
    parser.add_argument('--is_glove', type=bool, default=False, help='Use Glove')

    args = parser.parse_args()
    file_name = args.file_name
    num_examples = args.num_examples
    batch_size = args.batch_size
    # makes sure that we have a suitable batch size and number of examples
    if batch_size > int(num_examples*0.2):
        parser.error("Batch size cannot be greater than 20% of the number of examples")
    epochs = args.epochs
    lr = args.lr
    sentiment_threshold = args.sentiment_threshold
    load_weights = args.load_weights
    is_glove = args.is_glove

    # file_path
    file_path = "../data/{}.json.gz".format(file_name)
    if not exists(file_path):
        raise Exception("File does not exist")
    # get data from preprocess
    review_list, labels_list = get_all_data(file_path)

    # get data from preprocess
    preprocess_data = preprocess(review_list, labels_list, num_examples, sentiment_threshold, is_glove)
    train_texts, train_labels, test_texts, test_labels, max_length, max_features = preprocess_data

    model = Model(max_length, max_features, batch_size, lr, is_glove)

    if load_weights:
        model.load_weights("../models/{}_weights.h5".format(file_name))

    # train model
    train(model, train_texts, train_labels, epochs)
    test(model, test_texts, test_labels)
    
    # TODO: Visualize the data
    # TODO: this should visualize the loss per batch rather whole loss
    visualize_loss_batch(model.loss_list)
    model.save_weights("../models/{}_weights.h5".format(file_name))

if __name__ == '__main__':
    main()