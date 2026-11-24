import asyncio
import json
import time
from pathlib import Path
from urllib.robotparser import RobotFileParser

from scripts.crawl import (
    CrawlDeduplicator,
    DomainRateLimiter,
    EthicalCrawler,
    PIIScrubber,
    RobotsChecker,
)


def test_pii_scrubber_emails_and_phones():
    text = (
        "Genuine ivory pendant for sale. Call +91 9876543210 or 9876543210. "
        "Email dealer_wild@domain.com or visit wa.me/919876543210. "
        "Telegram @wild_trade_guy."
    )
    scrubbed = PIIScrubber.scrub(text)

    # Must NOT contain phone number or email or handle
    assert "9876543210" not in scrubbed
    assert "dealer_wild@domain.com" not in scrubbed
    assert "@wild_trade_guy" not in scrubbed
    assert "[PHONE_REMOVED]" in scrubbed
    assert "[EMAIL_REMOVED]" in scrubbed
    assert "[HANDLE_REMOVED]" in scrubbed
    assert "[LINK_REMOVED]" in scrubbed


def test_pii_scrubber_clean_text():
    clean = "Authentic carved wooden craft item from India"
    assert PIIScrubber.scrub(clean) == clean


def test_domain_rate_limiter():
    async def _run():
        limiter = DomainRateLimiter(min_interval=0.05)

        t0 = time.monotonic()
        waited1 = await limiter.wait("https://example.com/item1")
        waited2 = await limiter.wait("https://example.com/item2")
        t1 = time.monotonic()

        assert waited1 == 0.0
        assert waited2 >= 0.02
        assert (t1 - t0) >= 0.04

    asyncio.run(_run())


def test_robots_checker_mock_rules():
    checker = RobotsChecker()

    # Disallow /private/
    disallowed = ["/private/", "/admin/"]
    assert checker.is_allowed_mock("https://example.com/public/items", disallowed) is True
    assert checker.is_allowed_mock("https://example.com/private/deal", disallowed) is False
    assert checker.is_allowed_mock("https://example.com/admin/login", disallowed) is False


def test_robots_parser_integration():
    robots_txt = """
User-agent: *
Disallow: /restricted/
Allow: /public/
"""
    parser = RobotFileParser()
    parser.parse(robots_txt.splitlines())

    assert parser.can_fetch("*", "https://example.com/public/item1") is True
    assert parser.can_fetch("*", "https://example.com/restricted/secret") is False


def test_crawl_deduplicator(tmp_path: Path):
    seen_file = tmp_path / "seen_urls.txt"
    dedup = CrawlDeduplicator(seen_file)

    test_url = "https://example.com/listing/456"
    assert dedup.is_seen(test_url) is False

    dedup.mark_seen(test_url)
    assert dedup.is_seen(test_url) is True

    # Re-instantiate from file
    dedup2 = CrawlDeduplicator(seen_file)
    assert dedup2.is_seen(test_url) is True


def test_ethical_crawler_record_generation(tmp_path: Path):
    async def _run():
        crawler = EthicalCrawler(output_dir=tmp_path, min_delay=0.01)
        out_file = tmp_path / "crawled_test.jsonl"

        record = await crawler.crawl_sample_listing(
            url="https://example.com/listing/999",
            raw_text="Rare hathajodi root. Call 9876543210 for price",
            image_url="https://example.com/img.jpg",
            platform="web",
            query="hathajodi",
            out_file=out_file,
        )

        assert record is not None
        assert record["url"] == "https://example.com/listing/999"
        assert record["platform"] == "web"
        assert record["query_used"] == "hathajodi"
        assert "9876543210" not in record["text"]
        assert "[PHONE_REMOVED]" in record["text"]

        # Verify JSONL content on disk
        assert out_file.exists()
        with open(out_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["url"] == "https://example.com/listing/999"

        # Second crawl of same URL must be deduplicated
        record_dup = await crawler.crawl_sample_listing(
            url="https://example.com/listing/999",
            raw_text="Duplicate post",
            image_url=None,
            platform="web",
            query="hathajodi",
            out_file=out_file,
        )
        assert record_dup is None

    asyncio.run(_run())
