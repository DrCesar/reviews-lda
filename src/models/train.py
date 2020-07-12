import logging
import pickle
import gensim.corpora as corpora
import numpy as np
import gensim

from src.data.prepare_data import read_data
from src.features.tokenize import lemmatization



def train(read=True):
    data_lemmatized  = []
    if read:
        data = read_data()

        data_lemmatized = lemmatization(data)

        with open("models/data.pkl", "wb") as output_file:
            pickle.dump(data_lemmatized, output_file)
    else:
        with open('models/data.pkl', 'rb') as input_file:
            data_lemmatized = pickle.load(input_file)



    id2word = corpora.Dictionary(data_lemmatized)
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]
    
    # print(data_lemmatized)
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=5, 
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    with open("models/model.pkl", "wb") as output_file:
        pickle.dump(lda_model, output_file)
