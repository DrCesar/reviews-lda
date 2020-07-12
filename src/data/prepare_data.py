

import pandas as pd
from langdetect import detect
import re

def read_data():
    data = pd.read_csv('data/tripadvisor_reviews.csv')
    raw_reviews = data.review.values.tolist()
    raw_reviews = [review for review in raw_reviews if detect(review) == 'en']

    return [re.sub(r'\'|\(|\)', '', review) for review in raw_reviews]