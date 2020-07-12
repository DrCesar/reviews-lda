
import spacy
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

stop_words = stopwords.words('english')