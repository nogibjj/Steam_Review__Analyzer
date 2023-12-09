import os
import pandas as pd
from langdetect import detect


input_folder = "ReviewAnalyzer\data\Parquet_Files"


def load_parquet(file_path):
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def detect_language(text):
    try:
        return detect(text)
    except:
        return None


def convert_timestamps(df):
    df["timestamp_created"] = pd.to_datetime(df["timestamp_created"], unit="s")
    df["timestamp_updated"] = pd.to_datetime(df["timestamp_updated"], unit="s")
    return df


def convert_author_last_played(df):
    if "author_last_played" in df.columns:
        df["author_last_played"] = pd.to_datetime(
            df["author_last_played"], unit="s", errors="coerce"
        )
    return df


def convert_last_played(df):
    if "last_played" in df.columns:
        df["last_played"] = pd.to_datetime(df["last_played"], unit="s", errors="coerce")
    return df


def apply_language_detection(df):
    df["detected_language"] = df["review"].apply(detect_language)
    return df


def filter_english_reviews(df):
    return df[df["detected_language"] == "en"].copy()


def drop_columns(df):
    columns_to_drop = [
        "language",
        "hidden_in_steam_china",
        "steam_china_location",
        "detected_language",
    ]
    return df.drop(columns=columns_to_drop, errors="ignore")


parquet_files = [
    os.path.join(input_folder, file)
    for file in os.listdir(input_folder)
    if file.endswith(".parquet")
]


output_folder = "ReviewAnalyzer\data\CleanParquets"

for idx, file_path in enumerate(parquet_files, start=1):
    print(
        f"Processing file {idx} out of {len(parquet_files)}: {file_path}"
    )  # Add this line
    game_df = load_parquet(file_path)

    if game_df is not None:
        game_df = convert_timestamps(game_df)
        game_df = convert_author_last_played(game_df)
        game_df = convert_last_played(game_df)
        game_df = apply_language_detection(game_df)
        df_en = filter_english_reviews(game_df)
        df_en = drop_columns(df_en)

        # Save the cleaned DataFrame to a Parquet file with "_clean" appended to the original filename
        output_file_name = (
            os.path.splitext(os.path.basename(file_path))[0] + "_clean.parquet"
        )
        output_file_path = os.path.join(output_folder, output_file_name)
        df_en.to_parquet(output_file_path, index=False)
