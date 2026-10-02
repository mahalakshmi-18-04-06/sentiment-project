import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}

# Predefined news sources
NEWS_SOURCES = {
    "BBC World":       "https://www.bbc.com/news",
    "The Hindu":       "https://www.thehindu.com/",
    "Times of India":  "https://timesofindia.indiatimes.com/",
    "NDTV":            "https://www.ndtv.com/latest",
    "Reuters":         "https://www.reuters.com/",
    "Al Jazeera":      "https://www.aljazeera.com/",
}


def scrape_from_url(url, limit=20):
    """Scrape headlines from a single URL."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except Exception as e:
        return [f"[Error scraping {url}]: {e}"]

    soup = BeautifulSoup(r.text, "html.parser")
    tags = soup.find_all(['h2', 'h3'])

    headlines = []
    seen = set()
    for t in tags:
        text = t.get_text().strip()
        if len(text) > 20 and text not in seen:
            headlines.append(text)
            seen.add(text)
        if len(headlines) >= limit:
            break
    return headlines


def scrape_multiple(sources, per_source=10):
    """
    Scrape from multiple sources.
    sources: list of source names (keys from NEWS_SOURCES)
    Returns: list of dicts [{headline, source}, ...]
    """
    all_items = []
    for source_name in sources:
        url = NEWS_SOURCES.get(source_name)
        if not url:
            continue
        headlines = scrape_from_url(url, limit=per_source)
        for h in headlines:
            if not h.startswith("[Error"):
                all_items.append({
                    "headline": h,
                    "source": source_name
                })
    return all_items


# Backward-compatible single-URL scraper
def scrape_headlines(url="https://www.bbc.com/news", limit=20):
    return scrape_from_url(url, limit=limit)


if __name__ == "__main__":
    print("=== Testing multi-source scrape ===")
    items = scrape_multiple(["BBC World", "The Hindu"], per_source=5)
    for i, item in enumerate(items, 1):
        print(f"{i:2d}. [{item['source']}] {item['headline']}")