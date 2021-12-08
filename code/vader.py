# preprocess returns train_texts, train_labels, test_texts, test_labels, max_length, max_features
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

class VADER_Model(tf.keras.Model):
    def __init__ (self, max_length, max_features, batch_size, lr):
        super(Model, self).__init__()

    def call(self, reviews, sentiment_threshold):
        polarity_scores = [sid.polarity_scores(review) for review in reviews]
        sentiments = []

        for score in polarity_scores:
            if score >= sentiment_threshold:
                sentiments.append(1)

            else:
                sentiments.append(0)

        return sentiments