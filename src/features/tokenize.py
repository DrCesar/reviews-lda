

import gensim
from gensim.utils import simple_preprocess

from src.features import nlp, stop_words
# from src.features.bigrams import apply_bigrams, build_bigrams


def build_bigrams(reviews):
    bigram = gensim.models.Phrases(reviews, min_count=5, threshold=100)

    return gensim.models.phrases.Phraser(bigram)


def apply_bigrams(reviews, bigram_mod):
    return [bigram_mod[doc] for doc in reviews]

def sent_to_words(reviews):
    review_words = []
    for sentence in reviews:
        review_words.append(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    
    return review_words

def remove_stopwords(reviews):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in reviews]

def lemmatization(reviews, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    bigrams_mod = build_bigrams(reviews)

    reviews = apply_bigrams(remove_stopwords(sent_to_words(reviews)), bigrams_mod)

    texts_out = []
    for sent in reviews:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out