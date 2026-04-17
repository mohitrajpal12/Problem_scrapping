"""
web_search.py - Scrapes Google search results and extracts text content from result pages.
"""

import logging
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

GOOGLE_SEARCH_URL = "https://www.google.com/search"
REQUEST_TIMEOUT = 10
MAX_CONTENT_LENGTH = 2000
DEFAULT_NUM_RESULTS = 10


class WebSearchTool:
    """Scrapes Google search results and extracts visible text from each result page."""

    def __init__(
        self,
        num_results: int = DEFAULT_NUM_RESULTS,
        timeout: int = REQUEST_TIMEOUT,
        max_content_length: int = MAX_CONTENT_LENGTH,
    ):
        self.num_results = num_results
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def search(self, query: str) -> list[str]:
        """
        Perform a Google search and return a list of result URLs.

        Args:
            query: The search query string.

        Returns:
            A list of URLs from the search results page.
        """
        params = {"q": query, "num": self.num_results}
        urls = []

        try:
            response = self.session.get(
                GOOGLE_SEARCH_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup.select("a[href]"):
                href = tag["href"]
                # Google wraps result links in /url?q=<actual_url>
                if href.startswith("/url?q="):
                    url = href.split("/url?q=")[1].split("&")[0]
                    if url.startswith("http") and "google.com" not in url:
                        urls.append(url)

            logger.info("Found %d URLs for query: '%s'", len(urls), query)
        except requests.RequestException as e:
            logger.error("Search failed for query '%s': %s", query, e)

        return urls[: self.num_results]

    def extract_text(self, url: str) -> str:
        """
        Fetch a webpage and return its visible text content.

        Args:
            url: The URL of the page to scrape.

        Returns:
            Cleaned visible text, truncated to max_content_length characters.
            Returns an empty string on failure.
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()

            text = " ".join(soup.get_text(separator=" ").split())
            return text[: self.max_content_length]

        except requests.RequestException as e:
            logger.warning("Skipping %s — %s", url, e)
            return ""

    def get_problem_data(self, query: str) -> list[dict]:
        """
        Search for a query and extract text content from each result page.

        Args:
            query: The search query string.

        Returns:
            A list of dicts with keys 'url' and 'content' for each successfully
            scraped page.
        """
        urls = self.search(query)
        results = []

        for url in urls:
            logger.info("Extracting content from: %s", url)
            content = self.extract_text(url)
            if content:
                results.append({"url": url, "content": content})

        logger.info("Extracted content from %d/%d pages.", len(results), len(urls))
        return results
