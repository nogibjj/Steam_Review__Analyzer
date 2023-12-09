"""
In this python script we convert ONE game dictionary into a dataframe and then into a csv.

The json file is a dictionary of dictionary of dictionaries... 

We convert the first dictionaries keys into columns then go into
the second inner dicitionary to make those keys columns as well. 

The output of the parquet file of every review with any information attached review 
which is scraped in the *get_steam_data.py* 
"""
import json
import pandas as pd

# Use a raw string for the file path

with open(r"ReviewAnalyzer\data\review_1172620.json", "r") as file:
    data = json.load(file)

# Use the `get` method to handle the case when "reviews" is not present in data
reviews_data = data.get("reviews", {})
df = pd.DataFrame.from_dict(reviews_data, orient="index")

# get the second inner dictionary as column headers as well
for index, row in df.iterrows():
    # Extract the "author" dictionary
    author_dict = row.get("author", {})

    # iterate over each key in the "author" dictionary
    for key, value in author_dict.items():
        df.at[index, key] = value

# Drop the original "author" column
df = df.drop(columns=["author"])
df["weighted_vote_score"] = (
    df["weighted_vote_score"]
    .astype(str)
    .apply(lambda x: x.encode("utf-8", "ignore").decode("utf-8"))
)
file_name = "sea_of_thieves"
# CHANGE CSV NAME AND OUTPUT
df.to_parquet(f"ReviewAnalyzer/data/Parquet_Files/{file_name}.parquet")
