import json
import gzip
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

def main():
    dict = {}
    generator = parse('../data/reviews_Baby_5.json.gz')
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
    sent_tokens = []
    word_tokens = []
    for review in review_list[0:10]:
        sent_tokens.append(sent_tokenize(review))
        word_tokens.append(word_tokenize(review))

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

    print("Tokenized Words")
    print(tokenized_words)

    # Padding 
    padded = pad_sequences(tokenized_words)

    print("Padded Words")
    print(padded)
                    
if __name__ == '__main__':
    main()