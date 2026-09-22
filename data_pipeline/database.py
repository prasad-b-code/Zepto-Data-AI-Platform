import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data_pipeline/catalog.db")

def setup_and_load(df: pd.DataFrame):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        
        # 1. Reset tables with normalized foreign-key schema
        conn.executescript("""
            DROP TABLE IF EXISTS books;
            DROP TABLE IF EXISTS categories;

            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL,
                in_stock INTEGER NOT NULL,
                category_id INTEGER NOT NULL REFERENCES categories(category_id) ON DELETE CASCADE
            );
        """)

        # 2. Extract and insert unique categories
        categories = pd.DataFrame({"category_name": df["category"].dropna().unique()})
        categories.to_sql("categories", conn, if_exists="append", index=False)

        # 3. Map category names to their generated database IDs
        cat_map = pd.read_sql("SELECT category_id, category_name FROM categories", conn)
        df_books = df.merge(cat_map, left_on="category", right_on="category_name")

        # 4. Prepare and insert books
        cols = ["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]
        df_books["in_stock"] = df_books["in_stock"].astype(int)
        df_books[cols].to_sql("books", conn, if_exists="append", index=False)

        print(f"Loaded {len(categories)} categories and {len(df_books)} books into {DB_PATH}.")

if __name__ == "__main__":
    file_path = Path("data_pipeline/enriched_books.csv")
    
    if not file_path.exists():
        from enricher import enrich_with_currency
        df = enrich_with_currency(pd.read_csv("data_pipeline/cleaned_books.csv"))
    else:
        df = pd.read_csv(file_path)

    setup_and_load(df)