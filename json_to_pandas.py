import json
import pandas as pd

# Use your own file address
data_file = open("yelp_dataset/yelp_academic_dataset_checkin.json") 
data = []

for line in data_file:
    data.append(json.loads(line))

checkin_df = pd.DataFrame(data)
data_file.close()
