import json
import gzip
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

def main():
    dict = {}
    generator = parse('./data/reviews_Baby_5.json.gz')
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
if __name__ == '__main__':
    main()