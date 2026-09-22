import re
import pandas as pd
import numpy as np

RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}

def clean_price(price_val):
    """Strip currency symbols and convert to float."""
    if pd.isna(price_val):
        return np.nan
    clean_str = re.sub(r"[^\d.]", "", str(price_val))
    try:
        return float(clean_str)
    except ValueError:
        return np.nan

def clean_rating(rating_val):
    """Convert text rating ('One'..'Five') to integer (1..5)."""
    if pd.isna(rating_val):
        return np.nan
    val = str(rating_val).strip().lower()
    return RATING_MAP.get(val, np.nan)

def clean_availability(avail_val):
    """Parse availability text to boolean in_stock."""
    if pd.isna(avail_val):
        return False
    val = str(avail_val).lower()
    return "in stock" in val

def clean_books_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw catalog data:
    1. Drops rows without titles.
    2. Parses price_gbp, rating, and in_stock.
    3. Handles missing/unparseable numeric values using median imputation.
    """
    cleaned_df = df.copy()

    # Drop records missing critical product identifiers
    cleaned_df = cleaned_df.dropna(subset=["title"])
    cleaned_df = cleaned_df[cleaned_df["title"].str.strip() != ""]

    # 1. Strip currency and convert to float
    cleaned_df["price_gbp"] = cleaned_df["price"].apply(clean_price)

    # 2. Convert text rating to integer 1-5
    cleaned_df["rating"] = cleaned_df["star_rating"].apply(clean_rating)

    # 3. Parse availability to boolean
    cleaned_df["in_stock"] = cleaned_df["availability"].apply(clean_availability)

    # 4. Handle failed parses with median imputation
    if cleaned_df["price_gbp"].isna().any():
        median_price = cleaned_df["price_gbp"].median()
        cleaned_df["price_gbp"] = cleaned_df["price_gbp"].fillna(round(median_price, 2))
        print(f"[Imputation] Filled missing prices with median: {median_price:.2f}")

    if cleaned_df["rating"].isna().any():
        median_rating = int(cleaned_df["rating"].median())
        cleaned_df["rating"] = cleaned_df["rating"].fillna(median_rating).astype(int)
        print(f"[Imputation] Filled missing ratings with median: {median_rating}")
    else:
        cleaned_df["rating"] = cleaned_df["rating"].astype(int)

    # Select and reorder finalized clean columns
    cols = ["title", "category", "price_gbp", "rating", "in_stock"]
    return cleaned_df[cols]

if __name__ == "__main__":
    raw_path = "data_pipeline/scraped_raw_books.csv"
    try:
        raw_df = pd.read_csv(raw_path)
        print("Raw data loaded successfully.")
    except FileNotFoundError:
        print("Raw file not found. Running scraper first...")
        from scraper import run_task1_scraper
        data = run_task1_scraper()
        raw_df = pd.DataFrame(data)

    clean_df = clean_books_data(raw_df)
    
    print("\n--- Cleaned Data Sample (First 5 records) ---")
    print(clean_df.head())
    
    print("\n--- Data Types and Null Count ---")
    print(clean_df.info())

    clean_df.to_csv("data_pipeline/cleaned_books.csv", index=False)
    print("\nSaved cleaned data to data_pipeline/cleaned_books.csv")