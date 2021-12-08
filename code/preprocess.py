import json
import gzip
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re 
import matplotlib
import matplotlib.pyplot as plt
from glove import glove_embed
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

def get_all_data(path):
    """
    Gets all the reviewText and Overall Sentiment from the json file
    """
    dict = {}
    generator = parse(path)
    i = 0
    for line in generator:
        dict[i] = line
        i+=1

    data = {"reviewText": [], "overall": []}
    for d in dict:
        data["reviewText"].append(dict[d]["reviewText"])
        data["overall"].append(dict[d]["overall"])
    review_list = data["reviewText"]
    label_list = data["overall"]
    return review_list, label_list

def classify_labels(labels, sentiment_threshold):
    classified_labels = []
    for i in range(len(labels)):
        if(labels[i] > sentiment_threshold):
            classified_labels.append(1)
        else:
            classified_labels.append(0)
    return classified_labels

def stop_words(word_tokens, sent_tokens):
    stop_words = set(stopwords.words('english'))
    filtered_word = []
    for i in range(len(sent_tokens)):
        filtered_word.append([w for w in word_tokens[i] if not w.lower() in stop_words])
    return filtered_word

def porter_stemmer(filtered_word):
    ps = PorterStemmer()
    stemmed_word = []
    for i in range(len(filtered_word)):
        stemmed_word.append([ps.stem(w) for w in filtered_word[i]])
    return stemmed_word

def lemmatize_word(stemmed_word):
    lemmatize_word = []
    lemmatizer = nltk.WordNetLemmatizer()
    for i in range(len(stemmed_word)):
        lemmatize_word.append([lemmatizer.lemmatize(w) for w in stemmed_word[i]])
    return lemmatize_word

def normalize_text(lemmatized_word):
    NON_ALPHANUM = re.compile(r'[\W]')
    NON_ASCII = re.compile(r'[^a-z0-1\s]')
    normalized_word = []
    for review in lemmatized_word:
        new_review = []
        for word in review:
            lower = word.lower()
            no_punctuation = NON_ALPHANUM.sub(r'', lower)
            no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
            if no_non_ascii != '':
                new_review.append(no_non_ascii)
        
        normalized_word.append(new_review)
    return normalized_word

def tokenize_text(text):
    num_words = len(text)
    tokenizer = Tokenizer(num_words = num_words)
    tokenizer.fit_on_texts(text)

    tokenized_words = tokenizer.texts_to_sequences(text)
    return tokenized_words

def pad_text(vec_words):
    return pad_sequences(vec_words)

def preprocess(review_list, label_list, num_examples, sentiment_threshold, is_glove):
    sent_tokens = []
    word_tokens = []
    for review in review_list[0:num_examples]:
        sent_tokens.append(sent_tokenize(review))
        word_tokens.append(word_tokenize(review))

    # Classify labels 
    labels = label_list[0:num_examples]

    # UNCOMMENT TO VISUALIZE RATING FREQUENCY
    # visualize_rating_frequency(labels)

    classified_labels = classify_labels(labels, sentiment_threshold)
    filtered_word = stop_words(word_tokens, sent_tokens)
    
    # make plural words singular (stemming)
    # stemmed_word = porter_stemmer(filtered_word)

    # lemmatize words
    # lemmatized_word = lemmatize_word(stemmed_word)
    lemmatized_word = lemmatize_word(filtered_word)

    # Normalize text (omit punctuation and non-ascii characters)
    normalized_word = normalize_text(lemmatized_word)
    
    # Tokenization
    if is_glove:
        tokenized_words = glove_embed(normalized_word)
    else:
        tokenized_words = tokenize_text(normalized_word)

    # Padding 
    padded = pad_text(tokenized_words)

    train_test_split = int(0.8*len(padded))
    train_padded = padded[:train_test_split]
    test_padded = padded[train_test_split:]
    train_labels = classified_labels[:train_test_split]
    test_labels = classified_labels[train_test_split:]

    MAX_LENGTH = max(len(x) for x in train_padded)
    MAX_FEATURES = 20000

    # shuffle train data
    indices = tf.range(len(train_padded))
    shuffled_indices = tf.random.shuffle(indices)
    train_padded = tf.gather(train_padded, shuffled_indices)
    train_labels = tf.gather(train_labels, shuffled_indices)

    # shuffle test data
    indices = tf.range(len(test_padded))
    shuffled_indices = tf.random.shuffle(indices)
    test_padded = tf.gather(test_padded, shuffled_indices)
    test_labels = tf.gather(test_labels, shuffled_indices)

    print("Preprocessing finished")
    
    return train_padded, train_labels, test_padded, test_labels, MAX_LENGTH, MAX_FEATURES

def visualize_rating_frequency(labels):
    ratings = {}

    for label in labels:
        if label not in ratings:
            ratings[label] = 1

        else:
            ratings[label] = ratings[label] + 1
 
    keys = ratings.keys()
    values = ratings.values()
 
    plt.bar(keys,values)
    plt.show()
