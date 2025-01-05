# Standard Library Imports
import argparse
import asyncio
import logging
import re
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse
from functools import wraps

# Third-Party Library Imports
import aiohttp
import aiofiles
from aiohttp import ClientTimeout, ClientError
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm, tqdm_asyncio
from playwright.async_api import async_playwright

# Typing
from typing import Tuple, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)

USER_AGENT = "Mozilla/5.0"
HOST = "www.erome.com"
CHUNK_SIZE = 1024
ALLOWED_CONTENT_TYPES = {"video", "image"}

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (ClientError, asyncio.TimeoutError) as e:
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (backoff ** attempt))
                        logging.warning(f"Retry {attempt + 1}/{max_attempts} for {func.__name__} due to {e}")
                    else:
                        logging.error(f"Max retries reached for {func.__name__}")
                        raise
        return wrapper
    return decorator

def _clean_album_title(title: str, default_title="temp") -> str:
    """Remove illegal characters from the album title"""
    illegal_chars = r'[\\/:*?"<>|]'
    title = re.sub(illegal_chars, "_", title)
    title = title.strip(". ")
    return title if title else default_title


def _get_final_download_path(album_title: str) -> Path:
    """Create a directory with the title of the album"""
    final_path = Path("downloads") / album_title
    if not final_path.exists():
        final_path.mkdir(parents=True)
    return final_path

def normalize_url(url: str) -> str:
    """Normalize and validate the URL."""
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
    return urljoin(url, "/")

async def dump(url: str, max_connections: int, skip_videos: bool, skip_images: bool):
    """Collect album data and download the album"""
    if urlparse(url).hostname != HOST:
        raise ValueError(f"Host must be {HOST}")

    title, urls = await _collect_album_data(
        url=url, skip_videos=skip_videos, skip_images=skip_images
    )
    download_path = _get_final_download_path(album_title=title)

    await _download(
        album=url,
        urls=urls,
        max_connections=max_connections,
        download_path=download_path,
    )

async def _download(
    album: str,
    urls: list[str],
    max_connections: int,
    download_path: Path,
):
    """Download the album and display a summary using Playwright."""
    # Remove leftover temporary files
    for temp_file in download_path.glob("*.part"):
        logging.warning(f"Removing leftover temporary file: {temp_file}")
        temp_file.unlink()

    # Use Playwright for downloading
    tasks = [
        _download_file_with_playwright(
            url=url,
            download_path=download_path,
        )
        for url in urls
    ]
    results = await tqdm_asyncio.gather(
        *tasks,
        colour="MAGENTA",
        desc="Album Progress",
        unit="file",
        leave=True,
    )

    # Summarize results
    summary = {"completed": [], "skipped": [], "failed": []}
    for status, url in results:
        summary[status].append(url)

    logging.info("Download Summary:")
    logging.info(f"Completed: {len(summary['completed'])} files")
    logging.info(f"Skipped: {len(summary['skipped'])} files")
    logging.info(f"Failed: {len(summary['failed'])} files")
    if summary["failed"]:
        logging.error(f"Failed Downloads:\n{summary['failed']}")

@retry(max_attempts=3, delay=2)
async def _download_file_with_playwright(
    url: str,
    download_path: Path,
):
    """Download the file using Playwright's fetch API with browser-like headers."""
    temp_file_path = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        try:
            # Prepare file paths
            file_name = Path(urlparse(url).path).name
            temp_file_path = Path(download_path, f"{file_name}.part")
            final_file_path = Path(download_path, file_name)

            # Check if file is already downloaded
            if final_file_path.exists():
                logging.info(f"Skipping {url} [already downloaded]")
                return "skipped", url

            logging.info(f"Starting Playwright fetch for {url}")

            # Add headers to mimic a browser request
            headers = {
                "User-Agent": USER_AGENT,
                "Referer": "https://www.erome.com",  # Use a valid referer
            }

            # Fetch the resource with custom headers
            response = await context.request.get(url, headers=headers)

            # Validate response status
            if response.status != 200:
                logging.error(f"Failed to fetch {url}, HTTP status: {response.status}")
                return "failed", url

            # Validate content-type
            content_type = response.headers.get("content-type", "")
            if not any(ct in content_type for ct in ALLOWED_CONTENT_TYPES):
                logging.warning(f"Skipped {url} (Invalid Content-Type: {content_type})")
                return "skipped", url

            # Get content length for progress tracking
            content_length = response.headers.get("content-length")
            total_size_in_bytes = int(content_length) if content_length else None

            # Download the file in one go
            file_content = await response.body()

            # Write to temporary file
            async with aiofiles.open(temp_file_path, "wb") as f:
                await f.write(file_content)

            # Rename temp file to final file
            temp_file_path.rename(final_file_path)
            logging.info(f"Successfully downloaded {url} -> {final_file_path}")
            return "completed", url

        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")
            if temp_file_path and temp_file_path.exists():
                temp_file_path.unlink()  # Remove partial file
            return "failed", url

        finally:
            await browser.close()

@retry(max_attempts=3, delay=2)
async def _collect_album_data(
    url: str, skip_videos: bool, skip_images: bool
) -> Tuple[str, List[str]]:
    """Collect videos and images from the album."""
    headers = {"User-Agent": USER_AGENT}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            if response.status != 200:
                logging.error(f"Failed to fetch URL {url}, HTTP status: {response.status}")
                raise ValueError(f"Failed to fetch URL {url}, HTTP status: {response.status}")

            # Fetch the HTML content
            html_content = await response.text()
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract album title
            try:
                meta_tag = soup.find("meta", property="og:title")
                album_title = _clean_album_title(meta_tag["content"])
                logging.info(f"Extracted album title: {album_title}")
            except Exception as e:
                logging.error(f"Error extracting album title: {e}")
                raise ValueError("Album title could not be extracted from the metadata.")

            # Extract videos and images
            try:
                videos = (
                    [video["src"] for video in soup.find_all("source")]
                    if not skip_videos
                    else []
                )
                images = (
                    [
                        image["data-src"]
                        for image in soup.find_all("img", {"class": "img-back"})
                    ]
                    if not skip_images
                    else []
                )
                album_urls = list({*videos, *images})
                logging.info(f"Videos: {len(videos)} found, Images: {len(images)} found.")
            except Exception as e:
                logging.error(f"Error extracting media URLs: {e}")
                album_urls = []

            if not album_urls:
                logging.warning(f"No videos or images found for album: {album_title}.")

            return album_title, album_urls



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", help="URL to download", type=str, required=True)
    parser.add_argument(
        "-c",
        "--connections",
        help="Maximum number of simultaneous connections",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-sv",
        "--skip-videos",
        action=argparse.BooleanOptionalAction,
        help="Skip downloading videos",
    )
    parser.add_argument(
        "-si",
        "--skip-images",
        action=argparse.BooleanOptionalAction,
        help="Skip downloading images",
    )
    args = parser.parse_args()
    asyncio.run(
        dump(
            url=args.url,
            max_connections=args.connections,
            skip_videos=args.skip_videos,
            skip_images=args.skip_images,
        )
    )
