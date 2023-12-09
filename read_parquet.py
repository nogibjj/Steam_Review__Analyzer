import pandas as pd


df = pd.read_parquet("ReviewAnalyzer\data\Parquet_Files\stardew_valley_neg.parquet")
print(df.head(10))
