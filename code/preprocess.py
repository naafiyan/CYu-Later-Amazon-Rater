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
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
# from wordcloud import WordCloud, STOPWORDS

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

def preprocess(file_path, num_examples, sentiment_threshold):
    review_list, label_list = get_all_data(file_path)
    sent_tokens = []
    word_tokens = []
    for review in review_list[0:num_examples]:
        sent_tokens.append(sent_tokenize(review))
        word_tokens.append(word_tokenize(review))

    # Classify labels 
    labels = label_list[0:num_examples]

    # visualize_rating_frequency(labels)

    classified_labels = []
    for i in range(len(labels)):
        if(labels[i] > sentiment_threshold):
            classified_labels.append(1)
        else:
            classified_labels.append(0)
    
    # TODO: remove symbols from word_tokens
    stop_words = set(stopwords.words('english'))
    filtered_word = []
    for i in range(len(sent_tokens)):
        filtered_word.append([w for w in word_tokens[i] if not w.lower() in stop_words])
    
    # make plural words singular (stemming)
    ps = PorterStemmer()
    stemmed_word = []
    for i in range(len(filtered_word)):
        stemmed_word.append([ps.stem(w) for w in filtered_word[i]])

    # lemmatize words
    lemmatized_word = []
    lemmatizer = nltk.WordNetLemmatizer()
    for i in range(len(stemmed_word)):
        lemmatized_word.append([lemmatizer.lemmatize(w) for w in stemmed_word[i]])
    

    # Normalize text (omit punctuation and non-ascii characters)
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

    visualize_word_cloud(lemmatized_word)

    # map each word to its frequency
    word_freq = {}
    for i in range(len(lemmatized_word)):
        for word in lemmatized_word[i]:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    # Tokenization
    num_words = len(lemmatized_word)
    tokenizer = Tokenizer(num_words = num_words)
    tokenizer.fit_on_texts(lemmatized_word)

    tokenized_words = tokenizer.texts_to_sequences(lemmatized_word)

    # Padding 
    padded = pad_sequences(tokenized_words)

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

# def visualize_word_cloud(words):
#     wordcloud = WordCloud(words, background_color="white", max_words=100, min_word_length=5).generate(text)

#     fig=plt.figure(figsize=(15, 8))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.title('Total Reviews Word Clowd')
#     plt.show()
