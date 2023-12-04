import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

english_stopwords = set(stopwords.words('english'))

def process_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()] #remove punctuation
    tokens = [word for word in tokens if word not in english_stopwords] #remove stopwords
    return tokens

# read json file to pd data chunk list, using line and chunksize because file is
# too big and causes memeory error
data_chunks = pd.read_json('yelp_academic_dataset_reviews - Copy.json', lines=True, chunksize=1000)

comment_token_collector_by_star = {1: [],2: [],3: [],4: [],5: []}


# for each star number, join the list from all trunks and put in list

for chunk in data_chunks:
    for star_number in range(1,6):
        reviews = chunk[chunk['stars'] == star_number]['text'].tolist()
        for review in reviews:
            tokens = process_text(review)
            if tokens is None: 
                tokens = []
            comment_token_collector_by_star[star_number].extend(tokens)

#dictionary of word frequency
word_freq_by_start_dict = {star: nltk.FreqDist(tokens) for star, tokens in comment_token_collector_by_star.items()}
# item in dict will be word: number of appearance 
# example { 1: {like: 1000}}

# print the most common 10 words for comments with various stars
for star, freq_dist in word_freq_by_start_dict.items():
    print(f"Most common words for {star}-star reviews:")
    print(freq_dist.most_common(10))

# make it to be a histgram

# store this in a file, json or some other format.
with open('word_freq_by_star.json', 'w') as f:
    json.dump({star: freq_dist.most_common(10) for star, freq_dist in word_freq_by_start_dict.items()}, f, ensure_ascii=False)


