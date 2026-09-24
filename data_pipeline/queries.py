import sqlite3
import pandas as pd

conn = sqlite3.connect("data_pipeline/catalog.db")

# ---  SQL Queries ---
queries = {
    "1. SELECT, WHERE, ORDER BY, LIMIT": 
        "SELECT title, price_inr FROM books WHERE in_stock = 1 ORDER BY price_inr DESC LIMIT 5;",
    "2. DISTINCT, ORDER BY": 
        "SELECT DISTINCT rating FROM books ORDER BY rating ASC;",
    "3. BETWEEN, ORDER BY": 
        "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 2000 AND 3500 ORDER BY price_inr LIMIT 5;",
    "4. IN, WHERE": 
        "SELECT title, rating FROM books WHERE rating IN (1, 5) LIMIT 5;",
    "5. JOIN, ORDER BY, LIMIT": 
        "SELECT b.title, c.category_name, b.price_inr FROM books b JOIN categories c ON b.category_id = c.category_id ORDER BY b.price_inr DESC LIMIT 5;"
}

with open("data_pipeline/query_results.txt", "w", encoding="utf-8") as f:
    for name, sql in queries.items():
        df = pd.read_sql_query(sql, conn)
        block = f"--- {name} ---\nSQL: {sql}\nResults:\n{df.to_string(index=False)}\n\n"
        print(block)
        f.write(block)


# 1. Read the SQL join result via pd.read_sql
sql_join_df = pd.read_sql_query(queries["5. JOIN, ORDER BY, LIMIT"], conn)

# 2. Read raw tables into in-memory DataFrames
books_df = pd.read_sql_query("SELECT title, price_inr, category_id FROM books;", conn)
categories_df = pd.read_sql_query("SELECT category_id, category_name FROM categories;", conn)

# 3. Reproduce join using pd.merge in pure pandas (no SQL)
merged_df = (
    pd.merge(books_df, categories_df, on="category_id")
    [["title", "category_name", "price_inr"]]
    .sort_values(by="price_inr", ascending=False)
    .head(5)
    .reset_index(drop=True)
)

print("\n[SQL JOIN Output]:")
print(sql_join_df)

print("\n[Pandas pd.merge() Output]:")
print(merged_df)

# 4. Prove equivalence
is_equivalent = sql_join_df.equals(merged_df)
print(f"\nAre both results strictly equivalent? {is_equivalent}")

# Save verification to file
with open("data_pipeline/query_results.txt", "a", encoding="utf-8") as f:
    f.write(f"--- Task 6 Equivalence Check ---\n")
    f.write(f"SQL Join and Pandas Merge Output Equivalence: {is_equivalent}\n")

conn.close()