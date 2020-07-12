
import gensim



def build_bigrams(reviews):
    bigram = gensim.models.Phrases(reviews, min_count=5, threshold=100)

    return gensim.models.phrases.Phraser(bigram)


def apply_bigrams(reviews, bigram_mod):
    return [bigram_mod[doc] for doc in reviews]