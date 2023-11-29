import json
import pandas as pd

data = pd.read_json('yelp_dataset/yelp_academic_dataset_review.json', lines=True)
#print(data.columns)

star1 = data[data['stars'] == 1.0]['text'].tolist()
#star2 = data[data['stars'] == 2.0]['text'].tolist()
#star3 = data[data['stars'] == 3.0]['text'].tolist()
#star4 = data[data['stars'] == 4.0]['text'].tolist()
#star5 = data[data['stars'] == 5.0]['text'].tolist()

# Print 3 comments from the 1 star review
for i in range(3):
    print(star1[i])
