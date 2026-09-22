import os
import pandas as pd

# Required fixed baseline conversion rate (graded)
FIXED_GBP_TO_INR = 105.50

def enrich_with_currency(df: pd.DataFrame, rate: float = FIXED_GBP_TO_INR) -> pd.DataFrame:
    """Enriches dataset with price_inr column using the fixed baseline rate."""
    enriched_df = df.copy()
    enriched_df["price_inr"] = (enriched_df["price_gbp"] * rate).round(2)
    return enriched_df

if __name__ == "__main__":
    cleaned_path = "data_pipeline/cleaned_books.csv"
    output_path = "data_pipeline/enriched_books.csv"

    # Load cleaned data (or clean raw data if cleaned file doesn't exist yet)
    if os.path.exists(cleaned_path):
        clean_df = pd.read_csv(cleaned_path)
    else:
        from cleaner import clean_books_data
        raw_df = pd.read_csv("data_pipeline/scraped_raw_books.csv")
        clean_df = clean_books_data(raw_df)

    # Apply the fixed conversion rate
    enriched_df = enrich_with_currency(clean_df, rate=FIXED_GBP_TO_INR)

    # Save output
    enriched_df.to_csv(output_path, index=False)
    print(f"Successfully converted prices at rate {FIXED_GBP_TO_INR} and saved to {output_path}")
    print(enriched_df[["title", "price_gbp", "price_inr", "rating", "in_stock"]].head())