"""
Upload simulated microscope sample zarr datasets to the Hypha Artifact Manager.

Scans a base directory for zarr data folders, cross-references them with
SIMULATION_SAMPLES metadata, creates a gallery collection, and uploads each
sample as a child dataset.

Gallery name: squid-sim-samples
  Alternatives considered:
    - virtual-slide-library  (more generic, discoverable)
    - reef-sim-library       (workspace-specific)
    - squid-virtual-samples  (verbose but descriptive)

Usage:
    conda run -n squid python scripts/upload_sim_samples.py
    conda run -n squid python scripts/upload_sim_samples.py --base-dir /mnt/shared_documents
    conda run -n squid python scripts/upload_sim_samples.py --dry-run
    conda run -n squid python scripts/upload_sim_samples.py --samples HPA_FULL_SCAN U2OS_FUCCI
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

GALLERY_ALIAS = "squid-sim-samples"
GALLERY_MANIFEST = {
    "name": "Squid Simulated Samples",
    "description": (
        "Zarr datasets for the Squid virtual microscope simulator. "
        "Each entry is a real microscopy acquisition used as a ground-truth "
        "sample in simulation mode — no hardware required."
    ),
    "type": "squid-sim-gallery",
    "created_by": "squid-control",
    "tags": ["simulation", "zarr", "ome-zarr", "microscopy", "squid"],
}

WORKSPACE = "reef-imaging"
SERVER_URL = "https://hypha.aicell.io"


def _sanitize(name: str) -> str:
    """Lowercase alphanumeric + hyphens, start/end alphanumeric."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9\-]", "-", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    if not s:
        s = "dataset"
    if not s[0].isalnum():
        s = "s" + s
    if not s[-1].isalnum():
        s = s + "0"
    return s


def _dataset_alias(sample_key: str) -> str:
    """Derive a stable artifact alias for a sample, e.g. HPA_FULL_SCAN → sim-hpa-full-scan."""
    return "sim-" + _sanitize(sample_key)


def discover_samples(base_dir: Path) -> dict:
    """
    Scan base_dir for subdirectories that contain data.zarr.
    Cross-references with SIMULATION_SAMPLES to attach rich metadata.

    Returns:
        dict mapping sample_key → {zarr_path, metadata, ...}
        Folders with no match in SIMULATION_SAMPLES get minimal metadata.
    """
    from squid_control.simulation.samples import SIMULATION_SAMPLES

    # Build reverse map: zarr_dataset_path → sample_key
    path_to_key = {
        Path(v["zarr_dataset_path"]).parent: k
        for k, v in SIMULATION_SAMPLES.items()
    }

    found = {}
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        zarr_path = folder / "data.zarr"
        if not zarr_path.exists():
            logger.debug(f"Skipping {folder.name} — no data.zarr")
            continue

        sample_key = path_to_key.get(folder)
        if sample_key:
            meta = SIMULATION_SAMPLES[sample_key]
        else:
            logger.warning(
                f"Folder {folder.name} has no entry in SIMULATION_SAMPLES — "
                "will upload with minimal metadata."
            )
            sample_key = folder.name.upper().replace("-", "_")
            meta = {
                "config_name": "unknown",
                "description": f"Zarr dataset from {folder.name}.",
                "objective": "unknown",
                "channels": [],
            }

        found[sample_key] = {
            "zarr_path": zarr_path,
            "folder": folder,
            "meta": meta,
            "alias": _dataset_alias(sample_key),
        }
        logger.info(f"  Found: {sample_key} → {zarr_path}")

    return found


async def ensure_gallery(svc, workspace: str, dry_run: bool) -> None:
    """Create the gallery collection if it doesn't exist."""
    artifact_id = f"{workspace}/{GALLERY_ALIAS}"
    try:
        await svc.read(artifact_id=artifact_id)
        logger.info(f"Gallery already exists: {artifact_id}")
        return
    except Exception as e:
        if not any(k in str(e).lower() for k in ("not found", "does not exist", "keyerror")):
            raise

    logger.info(f"Creating gallery: {artifact_id}")
    if dry_run:
        logger.info("[dry-run] Would create gallery.")
        return

    await svc.create(
        alias=artifact_id,
        type="collection",
        manifest=GALLERY_MANIFEST,
        config={"permissions": {"*": "r", "@": "r+"}},
    )
    logger.info(f"Gallery created: {artifact_id}")


async def upload_sample(
    svc,
    sample_key: str,
    info: dict,
    workspace: str,
    token: str,
    dry_run: bool,
) -> dict:
    """Create (or overwrite) a child dataset under the gallery and upload its zarr."""
    from hypha_artifact import AsyncHyphaArtifact

    meta = info["meta"]
    zarr_path = info["zarr_path"]
    alias = info["alias"]
    dataset_id = f"{workspace}/{alias}"
    gallery_id = f"{workspace}/{GALLERY_ALIAS}"

    manifest = {
        "name": sample_key,
        "description": meta.get("description", ""),
        "sample_key": sample_key,
        "objective": meta.get("objective", ""),
        "scan_type": meta.get("scan_type", ""),
        "cell_line": meta.get("cell_line", ""),
        "staining": meta.get("staining", ""),
        "channels": meta.get("channels", []),
        "config_name": meta.get("config_name", ""),
        "zarr_path_local": str(zarr_path),
        "tags": ["simulation", "zarr", "ome-zarr"],
    }

    logger.info(f"[{sample_key}] alias={alias}")
    logger.info(f"[{sample_key}] zarr={zarr_path}")

    if dry_run:
        logger.info(f"[dry-run] Would upload {sample_key} → {dataset_id}")
        return {"success": True, "dry_run": True, "dataset_id": dataset_id}

    # Create (or stage for overwrite) via low-level service so we can set parent_id
    try:
        await svc.create(
            alias=dataset_id,
            type="generic",
            manifest=manifest,
            parent_id=gallery_id,
            overwrite=True,
        )
        logger.info(f"[{sample_key}] Dataset artifact created.")
    except Exception as exc:
        err = str(exc).lower()
        if "already" in err or "exist" in err:
            logger.info(f"[{sample_key}] Dataset exists — staging for overwrite.")
            await svc.edit(artifact_id=dataset_id, stage=True)
        else:
            raise

    # Upload zarr via AsyncHyphaArtifact (handles multipart, retries)
    logger.info(f"[{sample_key}] Uploading zarr ({zarr_path}) ...")
    async with AsyncHyphaArtifact(
        artifact_id=alias,
        workspace=workspace,
        token=token,
        server_url=SERVER_URL,
    ) as artifact:
        await artifact.edit(stage=True)
        await artifact.put(
            str(zarr_path),
            "data.zarr",
            recursive=True,
            multipart_config={
                "enable": True,
                "max_parallel_uploads": 8,
                "chunk_size": 20 * 1024 * 1024,  # 20 MB
            },
        )

    await svc.commit(
        artifact_id=dataset_id,
        comment=f"Uploaded OME-Zarr for simulated sample: {sample_key}",
    )
    logger.info(f"[{sample_key}] Done → {dataset_id}")
    return {"success": True, "dataset_id": dataset_id}


async def _async_main(args) -> None:
    from hypha_rpc import connect_to_server

    token = args.token or os.environ.get("REEF_WORKSPACE_TOKEN")
    if not token and not args.dry_run:
        logger.error(
            "No auth token. Set REEF_WORKSPACE_TOKEN or pass --token."
        )
        sys.exit(1)

    base_dir = Path(args.base_dir).resolve()
    if not base_dir.is_dir():
        logger.error(f"Base directory not found: {base_dir}")
        sys.exit(1)

    logger.info(f"Scanning {base_dir} for zarr sample folders...")
    samples = discover_samples(base_dir)

    if not samples:
        logger.error("No zarr sample folders found.")
        sys.exit(1)

    # Filter to requested samples if --samples was given
    if args.samples:
        requested = {s.upper() for s in args.samples}
        samples = {k: v for k, v in samples.items() if k in requested}
        if not samples:
            logger.error(f"None of {args.samples} found in {base_dir}.")
            sys.exit(1)

    logger.info(f"Samples to upload: {list(samples.keys())}")

    if args.dry_run:
        logger.info("[dry-run] No uploads will be performed.")
        await ensure_gallery(None, WORKSPACE, dry_run=True)
        for key, info in samples.items():
            await upload_sample(None, key, info, WORKSPACE, token="", dry_run=True)
        return

    logger.info(f"Connecting to {SERVER_URL} (workspace={WORKSPACE})...")
    server = await connect_to_server(
        {"server_url": SERVER_URL, "token": token, "workspace": WORKSPACE}
    )
    svc = await server.get_service("public/artifact-manager")

    await ensure_gallery(svc, WORKSPACE, dry_run=False)

    results = []
    for sample_key, info in samples.items():
        try:
            result = await upload_sample(
                svc, sample_key, info, WORKSPACE, token, dry_run=False
            )
            results.append(result)
        except Exception as exc:
            logger.error(f"[{sample_key}] Upload failed: {exc}", exc_info=True)
            results.append({"success": False, "sample_key": sample_key, "error": str(exc)})

    succeeded = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    logger.info("=" * 60)
    logger.info(f"Done. {len(succeeded)}/{len(results)} samples uploaded.")
    for r in succeeded:
        logger.info(f"  OK  {r['dataset_id']}")
    for r in failed:
        logger.info(f"  ERR {r.get('sample_key')} — {r.get('error')}")
    logger.info(f"Gallery: {SERVER_URL}/artifacts/{WORKSPACE}/{GALLERY_ALIAS}")
    logger.info("=" * 60)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Upload simulated microscope sample zarr datasets to Hypha.\n\n"
            f"Creates/updates gallery '{GALLERY_ALIAS}' under workspace '{WORKSPACE}',\n"
            "then uploads each discovered zarr folder as a child dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        default="/mnt/shared_documents",
        help="Directory containing zarr sample folders (default: /mnt/shared_documents).",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        metavar="SAMPLE_KEY",
        help=(
            "Upload only these sample keys, e.g. --samples HPA_FULL_SCAN U2OS_FUCCI. "
            "Uploads all discovered samples if omitted."
        ),
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
