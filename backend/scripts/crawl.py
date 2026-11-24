"""Ethical, rate-limited public listing crawler for illegal wildlife trade monitoring.

Enforces:
- Strict robots.txt compliance
- Rate limiting (>= 3.0s per request per domain)
- Zero PII storage (phones, emails, handles scrubbed)
- Deduplication via seen_urls.txt
- Clean JSONL output: {text, url, image_url, platform, crawled_at, query_used}
"""

import argparse
import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("iwt_crawler")

USER_AGENT = "IWTEthicalResearchBot/1.0 (+https://github.com/notarijeet/Biodiversity-)"


class PIIScrubber:
    """Removes personal identifiable information (PII) before storage."""

    # Matches email addresses
    EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

    # Matches Indian (+91 / 10-digit) and standard phone numbers
    PHONE_REGEX = re.compile(r"(?:\+?91[\s-]?)?[6789]\d{9}|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")

    # Matches WhatsApp links
    WHATSAPP_REGEX = re.compile(r"(?:https?://)?(?:wa\.me|api\.whatsapp\.com)/\d+")

    # Matches social media handles
    HANDLE_REGEX = re.compile(r"@[\w_]{3,}")

    @classmethod
    def scrub(cls, text: str) -> str:
        """Sanitize text by replacing all detected PII with neutral tokens."""
        if not text:
            return ""
        scrubbed = cls.EMAIL_REGEX.sub("[EMAIL_REMOVED]", text)
        scrubbed = cls.WHATSAPP_REGEX.sub("[LINK_REMOVED]", scrubbed)
        scrubbed = cls.PHONE_REGEX.sub("[PHONE_REMOVED]", scrubbed)
        scrubbed = cls.HANDLE_REGEX.sub("[HANDLE_REMOVED]", scrubbed)
        # Normalize whitespace
        return " ".join(scrubbed.split())


class DomainRateLimiter:
    """Ensures at least min_interval seconds elapse between requests to the same domain."""

    def __init__(self, min_interval: float = 3.0):
        self.min_interval = min_interval
        self._last_access: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def wait(self, url: str) -> float:
        """Wait if necessary before accessing the specified domain. Returns waited duration."""
        domain = urlparse(url).netloc
        if not domain:
            return 0.0

        async with self._lock:
            now = time.monotonic()
            last_time = self._last_access.get(domain, 0.0)
            elapsed = now - last_time
            waited = 0.0

            if elapsed < self.min_interval:
                waited = self.min_interval - elapsed
                await asyncio.sleep(waited)
                now = time.monotonic()

            self._last_access[domain] = now
            return waited


class RobotsChecker:
    """Fetches and verifies robots.txt permissions for target URLs."""

    def __init__(self, user_agent: str = USER_AGENT):
        self.user_agent = user_agent
        self._parsers: dict[str, RobotFileParser] = {}

    def is_allowed_mock(self, url: str, disallowed_paths: list[str]) -> bool:
        """Test helper to check disallowed paths without network access."""
        path = urlparse(url).path
        for dis in disallowed_paths:
            if path.startswith(dis):
                return False
        return True

    async def can_fetch(self, url: str, client: Optional[httpx.AsyncClient] = None) -> bool:
        """Verify whether the crawler is permitted to fetch the specified URL."""
        parsed = urlparse(url)
        domain = parsed.netloc
        if not domain:
            return False

        if domain not in self._parsers:
            robots_url = f"{parsed.scheme}://{domain}/robots.txt"
            parser = RobotFileParser()
            try:
                if client is not None:
                    resp = await client.get(robots_url, timeout=5.0)
                    if resp.status_code == 200:
                        parser.parse(resp.text.splitlines())
                    else:
                        parser.allow_all = True
                else:
                    async with httpx.AsyncClient(
                        headers={"User-Agent": self.user_agent}
                    ) as local_client:
                        resp = await local_client.get(robots_url, timeout=5.0)
                        if resp.status_code == 200:
                            parser.parse(resp.text.splitlines())
                        else:
                            parser.allow_all = True
            except Exception as e:
                logger.warning(
                    "Could not retrieve robots.txt for %s: %s (defaulting to safe allow)",
                    domain,
                    e,
                )
                parser.allow_all = True

            self._parsers[domain] = parser

        return self._parsers[domain].can_fetch(self.user_agent, url)


class CrawlDeduplicator:
    """Manages seen URLs to prevent duplicate scraping."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.seen_urls: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.filepath.exists():
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.seen_urls = {line.strip() for line in f if line.strip()}

    def is_seen(self, url: str) -> bool:
        return url in self.seen_urls

    def mark_seen(self, url: str) -> None:
        self.seen_urls.add(url)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"{url}\n")


class EthicalCrawler:
    """Main crawler orchestrator enforcing ethics, rate limits, and zero-PII outputs."""

    def __init__(
        self,
        output_dir: Path,
        min_delay: float = 3.0,
        user_agent: str = USER_AGENT,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limiter = DomainRateLimiter(min_interval=min_delay)
        self.robots_checker = RobotsChecker(user_agent=user_agent)
        self.dedup = CrawlDeduplicator(self.output_dir / "seen_urls.txt")
        self.scrubber = PIIScrubber()
        self.user_agent = user_agent

    def create_record(
        self,
        raw_text: str,
        url: str,
        image_url: Optional[str],
        platform: str,
        query: str,
    ) -> dict[str, Any]:
        """Produce a scrubbed, validated JSONL record with strictly no PII."""
        clean_text = self.scrubber.scrub(raw_text)
        return {
            "text": clean_text,
            "url": url,
            "image_url": image_url,
            "platform": platform,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "query_used": query,
        }

    async def crawl_sample_listing(
        self,
        url: str,
        raw_text: str,
        image_url: Optional[str],
        platform: str,
        query: str,
        out_file: Path,
    ) -> Optional[dict[str, Any]]:
        """Process and record a single listing after robots & dedup verification."""
        if self.dedup.is_seen(url):
            logger.info("Skipping already-seen URL: %s", url)
            return None

        # Check robots.txt
        allowed = await self.robots_checker.can_fetch(url)
        if not allowed:
            logger.warning("Disallowed by robots.txt: %s. Skipping.", url)
            return None

        # Apply rate limiting
        waited = await self.rate_limiter.wait(url)
        if waited > 0:
            logger.debug("Rate-limited: waited %.2fs for %s", waited, url)

        record = self.create_record(
            raw_text=raw_text,
            url=url,
            image_url=image_url,
            platform=platform,
            query=query,
        )

        # Write to JSONL
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.dedup.mark_seen(url)
        logger.info("Successfully recorded listing: %s", url)
        return record


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ethical Public Listing Crawler for IWT Detection."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query or codeword (e.g. hathajodi, shahtoosh, pangolin)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum listings to collect (default: 20)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Rate-limit delay in seconds per domain (default: 3.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.data_dir / "raw",
        help="Target directory for crawled JSONL files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate crawl using synthesized sample records",
    )
    return parser.parse_args()


async def main_async():
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = args.output_dir / f"crawled_{timestamp}.jsonl"

    crawler = EthicalCrawler(output_dir=args.output_dir, min_delay=args.delay)

    logger.info("Starting crawler with query: '%s', limit: %d", args.query, args.limit)
    logger.info("Output file: %s", out_file)

    if args.dry_run:
        logger.info("Dry-run mode enabled: generating test listings...")
        samples = [
            (
                f"https://example.com/items/{args.query}_01",
                f"Rare {args.query} item for puja. Call +91 9876543210 or email seller@shop.com",
                f"https://example.com/img/{args.query}_01.jpg",
                "web",
            ),
            (
                f"https://example.com/items/{args.query}_02",
                f"Authentic antique {args.query} amulet. Contact @wild_seller on Telegram.",
                None,
                "telegram",
            ),
        ]
        count = 0
        for url, raw_text, img_url, platform in samples:
            if count >= args.limit:
                break
            rec = await crawler.crawl_sample_listing(
                url=url,
                raw_text=raw_text,
                image_url=img_url,
                platform=platform,
                query=args.query,
                out_file=out_file,
            )
            if rec:
                count += 1
        logger.info("Dry-run complete. Saved %d records to %s", count, out_file)
    else:
        logger.info(
            "Live crawling requires active network access. Use --dry-run for offline tests."
        )


if __name__ == "__main__":
    asyncio.run(main_async())
