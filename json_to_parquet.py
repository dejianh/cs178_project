import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Read JSON into DataFrame
df_list = []
with open('src/yelp_academic_dataset_review.json', encoding='utf-8') as f:
    df_reader = pd.read_json(f, lines=True, chunksize=100)
    for chunk in df_reader:
        df_list.append(chunk)

# Concatenate the list of DataFrames into a single DataFrame
df = pd.concat(df_list, ignore_index=True)

# Convert DataFrame to Arrow Table
table = pa.Table.from_pandas(df)

# Write Arrow Table to Parquet file
# pq.write_table(table, 'output.parquet')
