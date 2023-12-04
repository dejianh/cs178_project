import json
import pandas as pd

# read json file to pd data chunk list, using line and chunksize because file is
# too big and causes memeory error
data_chunks = pd.read_json('src/yelp_academic_dataset_review.json', lines=True, chunksize=1000)

comment_token_collector_by_star = {1:dict(),2:dict(),3:dict(),4:dict(),5:dict()}
# item in dict will be word: number of appearance 
# example { 1: {like: 1000}}

# for each star number, join the list from all trunks and put in list

for chunk in data_chunks:
    for star_number in range(1,6):
         comment_token_collector_by_star[star_number].extend(chunk[chunk['stars'] ==\
             star_number]['text'].tolist())
        # cut out stopping words, puntunations
        #using nlkt here to pasre the comment in to words and put them in to
        #dict


# print the most common 10 words for comments with various stars

# make it to be a histgram

# store this in a file, json or some other format.
print(len(comment_token_collector_by_star[1]))


