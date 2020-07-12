import pickle
from typing import List

from src.features.tokenize import lemmatization
import gensim.corpora as corpora

def predict(reviews: List[str]):
    predictions = []
    reviews = lemmatization(reviews)

    with open('models/model.pkl', 'rb') as input_file:
        lda_model = pickle.load(input_file)

    id2word = corpora.Dictionary(reviews)

    print(reviews)
    corpus = [id2word.doc2bow(text) for text in reviews]

    print(list(lda_model[corpus])[0])
    return {
        'predictions': [[str(value) for value in pre[0]] for pre in list(lda_model[corpus])],
        'topics': lda_model.print_topics()
    }
