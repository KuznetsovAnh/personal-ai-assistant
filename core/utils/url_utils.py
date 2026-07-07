"""
Google News URL Resolver

Resolves Google News RSS redirect URLs to canonical article URLs.
Uses title + source domain search strategy (Strategy B) since Google no longer redirects.

Discovery from debugging:
- Base64 segment is actually protobuf-encoded internal ID, NOT the original URL
- HTTP redirect no longer works - Google returns Google News page instead
- RSS feed has `source.href` field with source domain (e.g., https://tienphong.vn)
- Solution: Search "title site:domain" on Google to find original article URL
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class GoogleNewsResolver:
    """Resolve Google News URLs to canonical article URLs."""

    def __init__(self, timeout: float = 2.5, max_workers: int = 8):
        """
        Initialize resolver.

        Args:
            timeout: HTTP request timeout in seconds (default: 2.5s)
            max_workers: Max parallel workers for batch resolution
        """
        self.timeout = timeout
        self.max_workers = max_workers

    def is_google_news_url(self, url: str) -> bool:
        """Check if URL is a Google News RSS URL."""
        return bool(url and "news.google.com/rss/articles" in url)

    def decode_base64_url(self, url: str) -> Optional[str]:
        """
        Attempt to decode Google News URL via base64.

        Google News URLs format: https://news.google.com/rss/articles/CBMi...
        The segment after /articles/ may contain base64-encoded data.

        Args:
            url: Google News URL

        Returns:
            Decoded canonical URL if successful, None otherwise
        """
        if not self.is_google_news_url(url):
            return None

        try:
            # Extract base64 segment
            parsed = urlparse(url)
            path = parsed.path  # e.g., /rss/articles/CBMi...

            if "/rss/articles/" not in path:
                return None

            # Get segment after /rss/articles/
            segments = path.split("/rss/articles/")
            if len(segments) < 2:
                return None

            encoded = segments[1]
            if not encoded:
                return None

            # Try base64 decode
            # Google News uses URL-safe base64 without padding
            # Add padding if needed
            padding = 4 - (len(encoded) % 4)
            if padding != 4:
                encoded += "=" * padding

            # Decode (URL-safe base64)
            decoded = base64.urlsafe_b64decode(encoded)

            # Try to extract URL from decoded bytes
            decoded_str = decoded.decode('utf-8', errors='ignore')

            # Look for URL pattern in decoded string
            # Common patterns: full URL, or URL with metadata
            if decoded_str.startswith('http'):
                # Direct URL
                return decoded_str.strip()

            # Try to find URL in decoded content
            # Pattern: https://domain.com/...
            url_match = re.search(r'https?://[^\s\x00-\x1f]+', decoded_str)
            if url_match:
                return url_match.group(0).rstrip('/')

            logger.debug(f"Base64 decoded but no URL found: {decoded_str[:100]}")
            return None

        except Exception as e:
            logger.debug(f"Base64 decode failed for {url}: {e}")
            return None

    def resolve_via_redirect(self, url: str, timeout: Optional[float] = None) -> str:
        """
        Resolve Google News URL by following HTTP redirects.

        Args:
            url: Google News URL
            timeout: Request timeout (default: use instance timeout)

        Returns:
            Canonical URL if successful, original URL otherwise
        """
        if not self.is_google_news_url(url):
            return url

        timeout = timeout or self.timeout

        # Use strong browser User-Agent to avoid bot detection
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }

        try:
            # Try HEAD request first (faster)
            response = requests.head(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers=headers
            )

            final_url = response.url

            # Verify it's not still a Google News URL
            if self.is_google_news_url(final_url):
                logger.debug("Strategy A: HEAD redirect still points to Google News")
                return url

            logger.debug(f"Strategy A: HEAD redirect success -> {final_url}")
            return final_url

        except requests.exceptions.Timeout:
            logger.debug("Strategy A: HEAD timeout")
            return url

        except Exception as e:
            # HEAD failed, try GET with stream
            try:
                response = requests.get(
                    url,
                    timeout=timeout,
                    allow_redirects=True,
                    stream=True,
                    headers=headers
                )
                response.close()

                final_url = response.url
                if self.is_google_news_url(final_url):
                    logger.debug("Strategy A: GET redirect still points to Google News")
                    return url

                logger.debug(f"Strategy A: GET redirect success -> {final_url}")
                return final_url

            except Exception as e2:
                logger.debug(f"Strategy A: Failed with {type(e2).__name__}")
                return url

    def resolve_via_google_search(self, title: str, source_domain: str, timeout: Optional[float] = None) -> Optional[str]:
        """
        Strategy B: Search for original article using title + source domain.

        Args:
            title: Article title
            source_domain: Source domain (e.g., "tienphong.vn")
            timeout: Request timeout in seconds

        Returns:
            Original article URL if found, None otherwise
        """
        timeout = timeout or self.timeout

        # Clean title for search
        clean_title = re.sub(r'[^\w\s]', '', title).strip()

        # Build search query: "title site:domain.com"
        query = f"{clean_title} site:{source_domain}"

        try:
            # Use DuckDuckGo HTML search (free, no API key needed)
            search_url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(search_url, params=params, headers=headers, timeout=timeout)
            html = response.text

            # Parse search results - look for first result from source domain
            # DuckDuckGo HTML structure: <a class="result__url" href="URL">
            result_pattern = rf'<a[^>]+class="result__url"[^>]+href="([^"]*{re.escape(source_domain)}[^"]*)"'
            match = re.search(result_pattern, html)

            if match:
                found_url = match.group(1)
                # Clean up URL
                found_url = found_url.split('&')[0]  # Remove tracking params
                logger.debug(f"Google search found: {query} -> {found_url}")
                return found_url

            # Fallback: look for any URL from source domain in results
            domain_pattern = rf'href="(https?://[^"]*{re.escape(source_domain)}[^"]*)"'
            matches = re.findall(domain_pattern, html)

            for url in matches:
                # Skip search engine URLs
                if 'duckduckgo.com' not in url and source_domain in url:
                    logger.debug(f"Google search fallback found: {query} -> {url}")
                    return url

            logger.debug(f"Google search found no results: {query}")
            return None

        except Exception as e:
            logger.debug(f"Google search failed for {query}: {e}")
            return None

    def resolve_via_tavily(self, title: str, source_domain: str, timeout: Optional[float] = None) -> Optional[str]:
        """
        Strategy C: Use Tavily API to search for original article.

        Args:
            title: Article title
            source_domain: Source domain (e.g., "tienphong.vn")
            timeout: Request timeout in seconds

        Returns:
            Original article URL if found, None otherwise
        """
        timeout = timeout or self.timeout

        try:
            from ..config import settings
            tavily_api_key = settings.tavily_api_key

            if not tavily_api_key:
                logger.debug("Tavily API key not configured")
                return None

            # Build search query with domain filter
            query = f"{title} {source_domain}"

            # Tavily Search API
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_api_key,
                "query": query,
                "max_results": 5,
                "search_depth": "basic"
            }

            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            # Find first result from source domain
            for result in results:
                result_url = result.get("url", "")
                if source_domain in result_url and not self.is_google_news_url(result_url):
                    logger.debug(f"Tavily found: {query} -> {result_url}")
                    return result_url

            logger.debug(f"Tavily found no matching results for: {query}")
            return None

        except Exception as e:
            logger.debug(f"Tavily search failed for {title}: {e}")
            return None

    def resolve(self, url: str, title: str = "", source_domain: str = "") -> str:
        """
        Resolve Google News URL using hybrid strategy.

        Strategy:
        A. HTTP redirect (may fail if Google changed behavior)
        B. DuckDuckGo search (may be blocked)
        C. Tavily API search (reliable fallback)

        Args:
            url: Google News URL or any URL
            title: Article title (for strategies B, C)
            source_domain: Source domain like "tienphong.vn" (for strategies B, C)

        Returns:
            Canonical URL if resolution successful, original URL otherwise
        """
        if not self.is_google_news_url(url):
            return url

        # Strategy A: Try HTTP redirect
        resolved = self.resolve_via_redirect(url)
        if resolved != url:
            logger.debug(f"Strategy A success: {url} -> {resolved}")
            return resolved

        # Need title and source_domain for strategies B, C
        if not title or not source_domain:
            logger.debug(f"Cannot resolve without title and source_domain: {url}")
            return url

        # Strategy B: DuckDuckGo search (may be blocked)
        found = self.resolve_via_google_search(title, source_domain)
        if found and not self.is_google_news_url(found):
            logger.debug(f"Strategy B success: {url} -> {found}")
            return found

        # Strategy C: Tavily API search (reliable)
        found = self.resolve_via_tavily(title, source_domain)
        if found and not self.is_google_news_url(found):
            logger.debug(f"Strategy C success: {url} -> {found}")
            return found

        logger.debug(f"All strategies failed: {url}")
        return url

    def resolve_batch(self, urls: List[str], titles: List[str] = None, source_domains: List[str] = None) -> Dict[str, str]:
        """
        Resolve multiple URLs in parallel.

        Args:
            urls: List of URLs to resolve
            titles: Optional list of article titles (for Strategy B)
            source_domains: Optional list of source domains (for Strategy B)

        Returns:
            Dict mapping original URL -> resolved URL
        """
        if not urls:
            return {}

        # Filter Google News URLs
        google_news_urls = [url for url in urls if self.is_google_news_url(url)]
        other_urls = [url for url in urls if not self.is_google_news_url(url)]

        if not google_news_urls:
            # No Google News URLs, return as-is
            return {url: url for url in urls}

        results = {}

        # Add non-Google News URLs as-is
        for url in other_urls:
            results[url] = url

        # Prepare metadata for Google News URLs
        if titles is None:
            titles = [""] * len(urls)
        if source_domains is None:
            source_domains = [""] * len(urls)

        # Create mapping from URL to metadata
        url_metadata = {}
        for i, url in enumerate(urls):
            if self.is_google_news_url(url):
                title = titles[i] if i < len(titles) else ""
                domain = source_domains[i] if i < len(source_domains) else ""
                url_metadata[url] = (title, domain)

        # Resolve Google News URLs in parallel
        def resolve_single(url: str) -> tuple[str, str]:
            title, domain = url_metadata.get(url, ("", ""))
            return url, self.resolve(url, title=title, source_domain=domain)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(resolve_single, url) for url in google_news_urls]

            for future in as_completed(futures):
                try:
                    original, resolved = future.result()
                    results[original] = resolved
                except Exception as e:
                    logger.error(f"Failed to resolve URL: {e}")

        return results


# Singleton instance for convenience
_default_resolver = GoogleNewsResolver()


def resolve_google_news_url(url: str) -> str:
    """
    Convenience function to resolve Google News URL.

    Uses default resolver instance.

    Args:
        url: Google News URL

    Returns:
        Canonical URL
    """
    return _default_resolver.resolve(url)


def resolve_google_news_urls(urls: List[str]) -> Dict[str, str]:
    """
    Convenience function to resolve multiple Google News URLs.

    Args:
        urls: List of URLs

    Returns:
        Dict mapping original -> resolved URLs
    """
    return _default_resolver.resolve_batch(urls)
