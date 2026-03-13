"""
Upload raw U2OS time-lapse datasets to the Hypha Artifact Manager.

Scans a base directory for time-lapse experiment folders and uploads each as a
child dataset under the 'u2os-timelapse' gallery collection.

Usage:
    python scripts/upload_u2os_timelapse.py
    python scripts/upload_u2os_timelapse.py --base-dir ~/europa_disk/u2os-treatment
    python scripts/upload_u2os_timelapse.py --dry-run
    python scripts/upload_u2os_timelapse.py --folders 20250619-U2OS-Eto-Tracker 20250704-U2OS-Eto-ER
    python scripts/upload_u2os_timelapse.py --skip static 001
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path

import dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    dotenv.load_dotenv(_ENV_PATH)

logger = logging.getLogger(__name__)

GALLERY_ALIAS = "u2os-timelapse"
WORKSPACE = "reef-imaging"
SERVER_URL = "https://hypha.aicell.io"

GALLERY_MANIFEST = {
    "name": "U2OS Cell Treatment Time-lapse",
    "description": (
        "Raw time-lapse microscopy datasets from U2OS cell treatment experiments. "
        "Includes various conditions: Etoposide treatment, ER stress, FUCCI cell-cycle "
        "reporters, Chromobody, and drug screens."
    ),
    "type": "collection",
    "created_by": "reef-imaging",
    "tags": ["u2os", "time-lapse", "microscopy", "cell-treatment", "raw"],
}

# Folders that are not time-lapse experiment datasets
DEFAULT_SKIP = {"static"}


def _sanitize(name: str) -> str:
    """Lowercase alphanumeric + hyphens, start/end alphanumeric."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9\-]", "-", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    if not s:
        s = "dataset"
    if not s[0].isalnum():
        s = "d" + s
    if not s[-1].isalnum():
        s = s + "0"
    return s


def _parse_folder_metadata(folder_name: str) -> dict:
    """
    Extract date, cell line, and treatment from a folder name.

    Examples:
        20250619-U2OS-Eto-Tracker  → date=2025-06-19, tags=[u2os, etoposide, tracker]
        20250912-U2OS_U2OS-FUICCI  → date=2025-09-12, tags=[u2os, fucci]
        001                        → date=unknown
    """
    meta: dict = {"date": None, "tags": ["u2os", "time-lapse", "raw"]}

    # Parse leading date: YYYYMMDD or YYYYMM (at least 6 digits)
    date_match = re.match(r"^(\d{4})(\d{2})(\d{2})?", folder_name)
    if date_match:
        year, month = date_match.group(1), date_match.group(2)
        day = date_match.group(3) or "01"
        meta["date"] = f"{year}-{month}-{day}"

    name_lower = folder_name.lower()
    if "eto" in name_lower:
        meta["tags"].append("etoposide")
    if "er" in name_lower:
        meta["tags"].append("er-stress")
    if "fucci" in name_lower or "fuicci" in name_lower:
        meta["tags"].append("fucci")
    if "chromobody" in name_lower:
        meta["tags"].append("chromobody")
    if "tracker" in name_lower:
        meta["tags"].append("tracker")
    if "drug" in name_lower:
        meta["tags"].append("drug-test")

    return meta


def discover_folders(base_dir: Path, skip: set[str]) -> list[dict]:
    """
    Scan base_dir for subdirectories that look like time-lapse datasets.

    Returns a list of dicts with folder info and extracted metadata.
    """
    found = []
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in skip:
            logger.info(f"Skipping (excluded): {folder.name}")
            continue

        meta = _parse_folder_metadata(folder.name)
        alias = "u2os-" + _sanitize(folder.name)
        found.append(
            {
                "folder": folder,
                "alias": alias,
                "meta": meta,
            }
        )
        logger.info(f"  Found: {folder.name} → alias={alias}, date={meta['date']}")

    return found


async def ensure_gallery(svc, dry_run: bool) -> None:
    """Create the gallery collection if it doesn't exist."""
    artifact_id = f"{WORKSPACE}/{GALLERY_ALIAS}"
    if dry_run:
        logger.info(f"[dry-run] Would ensure gallery: {artifact_id}")
        return

    try:
        await svc.read(artifact_id=artifact_id)
        logger.info(f"Gallery already exists: {artifact_id}")
        return
    except Exception as e:
        if not any(k in str(e).lower() for k in ("not found", "does not exist", "keyerror")):
            raise

    logger.info(f"Creating gallery: {artifact_id}")
    await svc.create(
        alias=artifact_id,
        type="collection",
        manifest=GALLERY_MANIFEST,
        config={"permissions": {"*": "r", "@": "r+"}},
    )
    logger.info(f"Gallery created: {artifact_id}")


async def upload_dataset(
    svc,
    info: dict,
    token: str,
    dry_run: bool,
) -> dict:
    """Create (or overwrite) a child dataset under the gallery and upload its contents."""
    from hypha_artifact import AsyncHyphaArtifact

    folder: Path = info["folder"]
    alias: str = info["alias"]
    meta: dict = info["meta"]
    dataset_id = f"{WORKSPACE}/{alias}"
    gallery_id = f"{WORKSPACE}/{GALLERY_ALIAS}"

    manifest = {
        "name": folder.name,
        "description": f"U2OS time-lapse dataset: {folder.name}",
        "date": meta.get("date", ""),
        "source_folder": folder.name,
        "tags": meta.get("tags", []),
    }

    logger.info(f"[{folder.name}] alias={alias}, path={folder}")

    if dry_run:
        logger.info(f"[dry-run] Would upload {folder.name} → {dataset_id}")
        return {"success": True, "dry_run": True, "dataset_id": dataset_id}

    # Create or stage artifact
    try:
        await svc.create(
            alias=dataset_id,
            type="generic",
            manifest=manifest,
            parent_id=gallery_id,
            overwrite=True,
        )
        logger.info(f"[{folder.name}] Dataset artifact created.")
    except Exception as exc:
        err = str(exc).lower()
        if "already" in err or "exist" in err:
            logger.info(f"[{folder.name}] Dataset exists — staging for overwrite.")
            await svc.edit(artifact_id=dataset_id, stage=True)
        else:
            raise

    # Upload folder contents
    logger.info(f"[{folder.name}] Uploading {folder} ...")
    async with AsyncHyphaArtifact(
        artifact_id=alias,
        workspace=WORKSPACE,
        token=token,
        server_url=SERVER_URL,
    ) as artifact:
        await artifact.edit(stage=True)
        await artifact.put(
            str(folder),
            folder.name,
            recursive=True,
            multipart_config={
                "enable": True,
                "max_parallel_uploads": 8,
                "chunk_size": 20 * 1024 * 1024,  # 20 MB
            },
        )

    await svc.commit(
        artifact_id=dataset_id,
        comment=f"Uploaded time-lapse dataset: {folder.name}",
    )
    logger.info(f"[{folder.name}] Done → {dataset_id}")
    return {"success": True, "dataset_id": dataset_id, "folder": folder.name}


async def _async_main(args) -> None:
    from hypha_rpc import connect_to_server

    token = args.token or os.environ.get("REEF_WORKSPACE_TOKEN")
    if not token and not args.dry_run:
        logger.error("No auth token. Set REEF_WORKSPACE_TOKEN or pass --token.")
        sys.exit(1)

    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.is_dir():
        logger.error(f"Base directory not found: {base_dir}")
        sys.exit(1)

    skip = DEFAULT_SKIP | set(args.skip or [])
    logger.info(f"Scanning {base_dir} for time-lapse folders...")
    datasets = discover_folders(base_dir, skip=skip)

    if not datasets:
        logger.error("No dataset folders found.")
        sys.exit(1)

    # Filter to requested folders if --folders was given
    if args.folders:
        requested = set(args.folders)
        datasets = [d for d in datasets if d["folder"].name in requested]
        if not datasets:
            logger.error(f"None of {args.folders} found in {base_dir}.")
            sys.exit(1)

    logger.info(f"Datasets to upload: {[d['folder'].name for d in datasets]}")

    if args.dry_run:
        logger.info("[dry-run] No uploads will be performed.")
        await ensure_gallery(None, dry_run=True)
        for info in datasets:
            await upload_dataset(None, info, token="", dry_run=True)
        return

    logger.info(f"Connecting to {SERVER_URL} (workspace={WORKSPACE})...")
    server = await connect_to_server(
        {"server_url": SERVER_URL, "token": token, "workspace": WORKSPACE}
    )
    svc = await server.get_service("public/artifact-manager")

    await ensure_gallery(svc, dry_run=False)

    results = []
    for info in datasets:
        try:
            result = await upload_dataset(svc, info, token, dry_run=False)
            results.append(result)
        except Exception as exc:
            logger.error(f"[{info['folder'].name}] Upload failed: {exc}", exc_info=True)
            results.append(
                {"success": False, "folder": info["folder"].name, "error": str(exc)}
            )

    succeeded = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    logger.info("=" * 60)
    logger.info(f"Done. {len(succeeded)}/{len(results)} datasets uploaded.")
    for r in succeeded:
        logger.info(f"  OK  {r['dataset_id']}")
    for r in failed:
        logger.info(f"  ERR {r.get('folder')} — {r.get('error')}")
    logger.info(f"Gallery: {SERVER_URL}/artifacts/{WORKSPACE}/{GALLERY_ALIAS}")
    logger.info("=" * 60)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Upload U2OS time-lapse datasets to Hypha.\n\n"
            f"Creates/updates gallery '{GALLERY_ALIAS}' under workspace '{WORKSPACE}',\n"
            "then uploads each discovered folder as a child dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        default="~/europa_disk/u2os-treatment",
        help="Directory containing time-lapse experiment folders.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        metavar="FOLDER_NAME",
        help="Upload only these folder names. Uploads all if omitted.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="FOLDER_NAME",
        default=[],
        help="Additional folder names to skip (e.g. --skip static 001).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hypha auth token (falls back to REEF_WORKSPACE_TOKEN env var).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and print what would be uploaded without doing anything.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(_async_main(_parse_args()))
