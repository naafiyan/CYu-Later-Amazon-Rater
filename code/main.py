from preprocess import preprocess
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
    for i in range(epochs):
        print("Num Examples: {}".format(len(train_texts)))

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
    for i in tqdm(range(batch_size, len(test_texts), batch_size), desc="Testing"):
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
    
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='File Name')
    parser.add_argument('--num_examples', type=int, default=10000, help='Number of examples to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')    

    args = parser.parse_args()
    file_name = args.file_name
    num_examples = args.num_examples
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    # file_path
    file_path = "../data/{}.json.gz".format(file_name)
    if not exists(file_path):
        raise Exception("File does not exist")
    # get data from preprocess
    preprocess_data = preprocess(file_path, num_examples)
    train_texts, train_labels, test_texts, test_labels, max_length, max_features = preprocess_data

    model = Model(max_length, max_features, batch_size, lr)

    # train model
    train(model, train_texts, train_labels, epochs)
    test(model, test_texts, test_labels)
    
    # TODO: Visualize the data
    visualize_loss_batch(model.loss_list)

if __name__ == '__main__':
    main()