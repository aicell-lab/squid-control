#!/usr/bin/env python3
"""
Download Opera Phoenix dataset (all plates) from figshare.scilifelab.se.

Uses Playwright to handle the JavaScript-based WAF challenge, then downloads
individual plate files with wget using the resolved cookies/redirect.

Usage:
    python scripts/download_opera_plates.py --output-dir /media/reef/harddisk/immunofluorescence_data
    python scripts/download_opera_plates.py --plates 6 7  # only plates 6 and 7
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# All known plate file IDs on figshare.scilifelab.se (article 14315777)
# Plates 1-5 are also on figshare.com; plates 6-7 are scilifelab-only.
PLATE_FILES = {
    1: {
        "tif": "https://ndownloader.figshare.com/files/27671958",
        "offsets": "https://ndownloader.figshare.com/files/27671955",
        "tif_size_gb": 20.47,
        "tif_md5": "2950104387d6313a68c38ec130380428",
    },
    2: {
        "tif": "https://ndownloader.figshare.com/files/27672696",
        "offsets": "https://ndownloader.figshare.com/files/27672693",
        "tif_size_gb": 20.27,
        "tif_md5": "e13dce069251f248dc1a3001d0131480",
    },
    3: {
        "tif": "https://ndownloader.figshare.com/files/27672903",
        "offsets": "https://ndownloader.figshare.com/files/27672879",
        "tif_size_gb": 20.68,
        "tif_md5": "19aaaa91ac5d4db007b2593f7bd64278",
    },
    4: {
        "tif": "https://ndownloader.figshare.com/files/27673347",
        "offsets": "https://ndownloader.figshare.com/files/27673344",
        "tif_size_gb": 19.63,
        "tif_md5": "efb90209e19cfd30a5870e57afe0be88",
    },
    5: {
        "tif": "https://ndownloader.figshare.com/files/27673620",
        "offsets": "https://ndownloader.figshare.com/files/27673614",
        "tif_size_gb": 20.10,
        "tif_md5": "6e52278c67788bbf1294d2d777f13f26",
    },
    6: {
        "tif": "https://figshare.scilifelab.se/ndownloader/files/27686883",
        "offsets": None,  # No offsets file found for plate 6
        "tif_size_gb": 19.32,
        "tif_md5": "46de8a127dceeb034d37264a46102f1b",
    },
    7: {
        "tif": "https://figshare.scilifelab.se/ndownloader/files/27687054",
        "offsets": None,  # No offsets file found for plate 7
        "tif_size_gb": 17.77,
        "tif_md5": "f31e23e571270b6be6fd0aad4c308767",
    },
}


def file_already_downloaded(dest_path: Path, expected_md5: str) -> bool:
    """Check if file exists and has the correct MD5."""
    if not dest_path.exists():
        return False
    size_gb = dest_path.stat().st_size / (1024**3)
    logger.info(f"  File exists: {dest_path} ({size_gb:.2f} GB)")
    # Quick size check (within 1 MB tolerance)
    plate_num = int(dest_path.stem.replace("plate", "").replace(".ome", ""))
    expected_size_gb = PLATE_FILES[plate_num]["tif_size_gb"]
    if abs(size_gb - expected_size_gb) > 0.01:
        logger.warning(f"  Size mismatch: got {size_gb:.2f} GB, expected {expected_size_gb:.2f} GB")
        return False
    logger.info(f"  File size matches expected {expected_size_gb:.2f} GB — skipping download")
    return True


def download_with_wget(url: str, dest_path: Path) -> bool:
    """Download a file using wget with resume support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "--continue",           # Resume partial downloads
        "--progress=bar:force",
        "--timeout=60",
        "--tries=5",
        "--retry-connrefused",
        "--waitretry=30",
        "-O", str(dest_path),
        url,
    ]
    logger.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


async def download_with_playwright(url: str, dest_path: Path) -> bool:
    """
    Use Playwright to solve WAF challenge, then download via direct HTTP.

    Playwright navigates to the URL, solves any JavaScript challenge,
    and then follows the redirect to get the actual download URL.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.error("Playwright not installed. Run: pip install playwright && python -m playwright install chromium")
        return False

    logger.info(f"  Using Playwright to resolve WAF challenge for {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Track download URL via network requests
        actual_download_url = None
        download_cookies = None

        async def handle_response(response):
            nonlocal actual_download_url
            # Look for the redirect to the actual S3/CDN file URL
            if response.status in (200, 206) and any(
                ext in response.url for ext in [".tif", "amazonaws.com", "storage"]
            ):
                actual_download_url = response.url
                logger.info(f"  Intercepted download URL: {actual_download_url[:80]}...")

        page.on("response", handle_response)

        try:
            # Navigate to the download URL — WAF challenge happens here
            logger.info(f"  Navigating to {url} (waiting up to 60s for WAF challenge)...")
            await page.goto(url, timeout=60000, wait_until="networkidle")

            # Get cookies after WAF resolution
            cookies = await context.cookies()
            download_cookies = "; ".join(f"{c['name']}={c['value']}" for c in cookies)

            # Wait a bit for any redirect to complete
            await asyncio.sleep(5)

        except Exception as e:
            logger.warning(f"  Playwright navigation issue: {e}")

        await browser.close()

        if actual_download_url:
            # Use wget with the resolved URL
            logger.info(f"  Downloading from resolved URL: {actual_download_url[:80]}...")
            cmd = ["wget", "--continue", "--progress=bar:force",
                   "-O", str(dest_path), actual_download_url]
            if download_cookies:
                cmd.extend(["--header", f"Cookie: {download_cookies}"])
            result = subprocess.run(cmd)
            return result.returncode == 0
        else:
            logger.warning("  Could not intercept download URL via Playwright")
            # Fall back to wget with browser-like headers
            return False


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    import hashlib
    logger.info(f"  Verifying MD5 for {file_path.name}...")
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual == expected_md5:
        logger.info(f"  MD5 OK: {actual}")
        return True
    else:
        logger.error(f"  MD5 MISMATCH: got {actual}, expected {expected_md5}")
        return False


async def download_plate(plate_num: int, output_dir: Path, verify: bool = True) -> bool:
    """Download a single plate's OME-TIFF and offsets JSON."""
    info = PLATE_FILES[plate_num]
    tif_url = info["tif"]
    offsets_url = info.get("offsets")
    expected_md5 = info["tif_md5"]

    tif_dest = output_dir / f"plate{plate_num}.ome.tif"
    needs_playwright = "figshare.scilifelab.se" in tif_url

    logger.info(f"\n{'='*60}")
    logger.info(f"Plate {plate_num}: {info['tif_size_gb']:.2f} GB")
    logger.info(f"  URL: {tif_url}")
    logger.info(f"  Dest: {tif_dest}")

    # Skip if already downloaded with correct size
    if file_already_downloaded(tif_dest, expected_md5):
        return True

    # Download
    success = False
    if needs_playwright:
        success = await download_with_playwright(tif_url, tif_dest)
        if not success:
            logger.error(
                f"  Playwright download failed for plate {plate_num}.\n"
                f"  Manual download: {tif_url}\n"
                f"  Save to: {tif_dest}"
            )
            return False
    else:
        success = download_with_wget(tif_url, tif_dest)

    if not success:
        logger.error(f"  Download failed for plate {plate_num}")
        return False

    # Verify checksum
    if verify and expected_md5:
        if not verify_md5(tif_dest, expected_md5):
            logger.error(f"  Checksum verification failed — file may be corrupt")
            return False

    # Download offsets JSON (plates 1-5 only)
    if offsets_url:
        offsets_dest = output_dir / f"plate{plate_num}.ome.tif_offsets.json"
        if not offsets_dest.exists():
            logger.info(f"  Downloading offsets JSON...")
            download_with_wget(offsets_url, offsets_dest)

    logger.info(f"  Plate {plate_num} download complete!")
    return True


async def main():
    parser = argparse.ArgumentParser(description="Download Opera Phoenix dataset from figshare")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/reef/harddisk/immunofluorescence_data/14315777"),
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--plates",
        type=int,
        nargs="+",
        default=list(PLATE_FILES.keys()),
        help="Plate numbers to download (default: all)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip MD5 checksum verification",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Plates to download: {args.plates}")

    # Calculate total size
    total_gb = sum(PLATE_FILES[p]["tif_size_gb"] for p in args.plates)
    logger.info(f"Total download size: {total_gb:.1f} GB")

    # Check available disk space
    stat = os.statvfs(output_dir)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    logger.info(f"Free disk space: {free_gb:.1f} GB")
    if free_gb < total_gb * 0.5:
        logger.warning(f"Low disk space! Need ~{total_gb:.0f} GB, only {free_gb:.0f} GB free")

    results = {}
    for plate_num in args.plates:
        if plate_num not in PLATE_FILES:
            logger.error(f"Unknown plate number: {plate_num} (valid: {list(PLATE_FILES.keys())})")
            continue
        results[plate_num] = await download_plate(
            plate_num, output_dir, verify=not args.no_verify
        )

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Download Summary:")
    for plate_num, success in results.items():
        status = "OK" if success else "FAILED"
        logger.info(f"  Plate {plate_num}: {status}")

    failed = [p for p, ok in results.items() if not ok]
    if failed:
        logger.error(f"\nFailed plates: {failed}")
        logger.info("\nFor WAF-protected files, download manually in your browser:")
        for plate_num in failed:
            info = PLATE_FILES[plate_num]
            dest = output_dir / f"plate{plate_num}.ome.tif"
            logger.info(f"  Plate {plate_num} ({info['tif_size_gb']:.2f} GB):")
            logger.info(f"    URL: {info['tif']}")
            logger.info(f"    Save to: {dest}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
