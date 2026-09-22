import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"

def get_categories():
    """Extract category names and URLs from the homepage sidebar."""
    response = requests.get(BASE_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    categories = []
    category_menu = soup.select("ul.nav-list > li > ul > li > a")
    for cat in category_menu:
        name = cat.get_text(strip=True)
        url = urljoin(BASE_URL, cat["href"])
        categories.append({"name": name, "url": url})
    return categories

def scrape_category_books(category_name, start_url, max_books_per_category=30):
    """Scrape books from a given category URL, handling pagination if present."""
    books = []
    current_url = start_url

    while current_url and len(books) < max_books_per_category:
        response = requests.get(current_url)
        if response.status_code != 200:
            break
        soup = BeautifulSoup(response.text, "html.parser")
        product_cards = soup.select("article.product_pod")

        for card in product_cards:
            # 1. Title
            title_tag = card.select_one("h3 a")
            title = title_tag.get("title", "").strip() if title_tag else ""

            # 2. Price (raw text in GBP, e.g., '£51.77')
            price_tag = card.select_one("p.price_color")
            price = price_tag.get_text(strip=True) if price_tag else ""

            # 3. Star Rating (extract word rating: One, Two, Three, Four, Five)
            rating_tag = card.select_one("p.star-rating")
            star_rating = ""
            if rating_tag:
                classes = rating_tag.get("class", [])
                star_rating = [c for c in classes if c != "star-rating"][0] if len(classes) > 1 else ""

            # 4. Availability
            avail_tag = card.select_one("p.instock.availability")
            availability = avail_tag.get_text(strip=True) if avail_tag else ""

            books.append({
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category_name
            })

            if len(books) >= max_books_per_category:
                break

        # Check for next page inside category
        next_button = soup.select_one("li.next a")
        if next_button:
            current_url = urljoin(current_url, next_button["href"])
        else:
            current_url = None

    return books

def run_task1_scraper(target_category_count=4, min_books=60):
    """Scrapes across categories until we meet both category and book count thresholds."""
    all_categories = get_categories()
    collected_books = []
    categories_used = 0

    print("--- Starting Scraping Task 1 ---")
    for cat in all_categories:
        print(f"Scraping category: {cat['name']}...")
        category_books = scrape_category_books(cat["name"], cat["url"], max_books_per_category=25)
        collected_books.extend(category_books)
        categories_used += 1
        time.sleep(0.3)  # Polite crawling delay

        if categories_used >= target_category_count and len(collected_books) >= min_books:
            break

    print(f"\nScraping complete!")
    print(f"Total books captured: {len(collected_books)}")
    print(f"Total categories scraped: {categories_used}")
    return collected_books

if __name__ == "__main__":
    import pandas as pd

    data = run_task1_scraper()
    df = pd.DataFrame(data)
    
    # Preview results
    print("\n--- Sample Scraped Data (First 5 records) ---")
    print(df.head())
    
    # Save a raw copy for validation
    df.to_csv("data_pipeline/scraped_raw_books.csv", index=False)
    print("\nSaved raw output to data_pipeline/scraped_raw_books.csv")