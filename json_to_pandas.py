import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

english_stopwords = set(stopwords.words('english'))
more_stopwords = {'would', 'really', 'food', 'us', 'one', 'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 
                  'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 
                  'by', 'can', 'could', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 
                  'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 
                  'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 
                  'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 
                  'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 
                  'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 
                  'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 
                  'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 
                  'yours', 'yourself', 'yourselves', 'get', 'got', 'told', 'said', 'even', 'also', 'two'
}

def process_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()] #remove punctuation
    tokens = [word for word in tokens if word not in english_stopwords.union(more_stopwords)] #Add 'would' to stopwords set in case it's missing
    return tokens


# read json file to pd data chunk list, using line and chunksize because file is
# too big and causes memeory error
data_chunks = pd.read_json('yelp_academic_dataset_review.json', lines=True, chunksize=1000)

comment_token_collector_by_star = {1: [], 2: [], 3: [], 4: [], 5: []}
star_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
total_word_freq = Counter()


# for each star number, join the list from all trunks and put in list

for chunk in data_chunks:
    for star_number in range(1,6):
        reviews = chunk[chunk['stars'] == star_number]['text'].tolist()
        for review in reviews:
            if review and not pd.isna(review):
                tokens = process_text(review)
                if tokens is None: 
                    tokens = []
                comment_token_collector_by_star[star_number].extend(tokens)
                # Update the total word frequency Counter           
                total_word_freq.update(tokens)
                star_count[star_number] += 1

#dictionary of word frequency
word_freq_by_star_dict = {star: nltk.FreqDist(tokens) for star, tokens in comment_token_collector_by_star.items()}
# item in dict will be word: number of appearance 
# example { 1: {like: 1000}}

# print the most common 10 words for comments with various stars
for star, freq_dist in word_freq_by_star_dict.items():
    print(f"Most common words for {star}-star reviews:")
    print(freq_dist.most_common(10))

#print number of reviews of each star
for start, count in star_count.items():
    print(start , ": " , count, " reviews")
    

colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray']

# Print the most common 10 words for comments with various stars
for star, freq_dist in word_freq_by_star_dict.items():
    print(f"Most common words for {star}-star reviews:")
    common_words = freq_dist.most_common(20)
    print(common_words)

    # Plotting
    words = [word for word, freq in common_words]
    frequencies = [freq for word, freq in common_words]
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies, color=colors[:len(words)])  # Apply different color to each bar
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Word Frequency in {star}-Star Reviews')
    plt.xticks(rotation=45)
    plt.show()

#Plot the total word frequency distribution
plt.figure(figsize=(10, 5))
most_common_words = total_word_freq.most_common(20)
words, counts = zip(*most_common_words)
plt.bar(words, counts)
plt.title('Total Word Frequency Across All Star Ratings')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Plot the distribution of comments
plt.figure(figsize=(10, 5))
stars, comment_counts = zip(*star_count.items())
plt.bar(stars, comment_counts, color=['red', 'orange', 'yellow', 'green', 'blue'])  # Use different colors for clarity
plt.title('Distribution of Comments by Star Rating')
plt.xlabel('Star Rating')
plt.ylabel('Number of Comments')
plt.xticks(range(1, 6))
plt.show()

# store this in a file, json or some other format.
with open('word_freq_by_star.json', 'w') as f:
    json.dump({star: freq_dist.most_common(10) for star, freq_dist in word_freq_by_star_dict.items()}, f, ensure_ascii=False)



