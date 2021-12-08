import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def glove_embed(review_list):
    embeddings_dict = {}
    with open("data/glove/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector


    vectorized_reviews = []
    for review in review_list:
        vectorized_review = []
        for word in review:
            vectorized_word = embeddings_dict[word]
            vectorized_review.append(vectorized_word)
        
        vectorized_reviews.append(vectorized_review)
        
    return vectorized_reviews
