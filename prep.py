import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import gc
import re

print("Starting VADER sentiment labeling script...")

try:
    analyzer = SentimentIntensityAnalyzer()
    print("VADER sentiment analyzer loaded.")
except LookupError:
    print("VADER lexicon not found. Downloading...")
    nltk.download("vader_lexicon")
    analyzer = SentimentIntensityAnalyzer()
    print("VADER lexicon downloaded and analyzer loaded.")

# Configuration
INPUT_CSV_PATH = "reddit_wsb.csv"
OUTPUT_CSV_PATH = "reddit_wsb_with_sentiments.csv"
COLUMNS_TO_LOAD = [
    "id",
    "title",
    "body",
    "created",
]


def preprocess(text):
    # remove links
    text = re.sub(r"https?://\S+", "", text)
    # remove username
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_vader_compound(text):
    if pd.isna(text) or not text.strip():
        return 0.0
    try:
        return analyzer.polarity_scores(str(text))["compound"]
    except Exception as e:
        print(f"Error in VADER scoring: {e} for text: {text[:100]}...")
        return 0.0


def vader_sentiment_label(compound_score, positive_thresh=0.05, negative_thresh=-0.05):
    if compound_score >= positive_thresh:
        return "positive"
    elif compound_score <= negative_thresh:
        return "negative"
    else:
        return "neutral"


def encode_labels(label):
    if label == "positive":
        return 2
    elif label == "neutral":
        return 1
    elif label == "negative":
        return 0
    return 1


# Processing
print(f"Loading data from {INPUT_CSV_PATH}...")
try:
    df = pd.read_csv(INPUT_CSV_PATH, usecols=COLUMNS_TO_LOAD)

except FileNotFoundError:
    print(f"Error: Input CSV file not found at {INPUT_CSV_PATH}")
    exit()
except ValueError as ve:
    print(
        f"Error: Columns {COLUMNS_TO_LOAD} might not all exist in {INPUT_CSV_PATH}. Details: {ve}"
    )
    try:
        temp_df = pd.read_csv(INPUT_CSV_PATH, nrows=1)
        print(f"Available columns are: {temp_df.columns.tolist()}")
    except Exception as e_temp:
        print(f"Could not read the header of the CSV. Error: {e_temp}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during CSV loading: {e}")
    exit()

# Combine title and selftext
print("Combining 'title' and 'body' into 'full_text'...")
df["title"] = df["title"].fillna("")
df["body"] = df["body"].fillna("")
df["full_text"] = df["title"] + " " + df["body"]
df["full_text"] = df["full_text"].str.strip()

# Preprocessing
print("Applying pre-VADER text cleaning (e.g., removing URLs/HTML)...")
df["full_text"] = df["full_text"].apply(preprocess)


# Apply VADER sentiment analysis
print("Applying VADER sentiment analysis...")
df["vader_compound"] = df["full_text"].apply(get_vader_compound)
df["vader_label"] = df["vader_compound"].apply(
    lambda score: vader_sentiment_label(score)
)
print("VADER sentiment analysis complete.")

print("Encoding sentiment labels to integers (0: neg, 1: neu, 2: pos)...")
df["sentiment_label_encoded"] = df["vader_label"].apply(encode_labels)

COLUMNS_TO_SAVE = [
    "id",
    "created",
    "full_text",
    "vader_compound",
    "vader_label",
    "sentiment_label_encoded",
]
final_columns_in_df = [col for col in COLUMNS_TO_SAVE if col in df.columns]
if len(final_columns_in_df) != len(COLUMNS_TO_SAVE):
    print(f"Warning: Not all columns intended for saving are present in the DataFrame.")
    print(f"Columns that will be saved: {final_columns_in_df}")
    print(
        f"Columns that were requested but missing: {list(set(COLUMNS_TO_SAVE) - set(final_columns_in_df))}"
    )

df_output = df[final_columns_in_df]


print(f"Saving processed data to {OUTPUT_CSV_PATH}...")
try:
    df_output.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Successfully saved.")
except Exception as e:
    print(f"Error saving output CSV: {e}")

del df
del df_output
gc.collect()
