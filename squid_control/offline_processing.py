"""
Offline processing module for microscopy experiment data.

Provides:
- Shared helper functions used by both the service and the standalone CLI.
- OfflineProcessor class — used by MicroscopeHyphaService to stitch and upload
  time-lapse data triggered via the RPC API.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import cv2
import dotenv
import pandas as pd

# Load .env from the repo root (two levels up from this script)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    dotenv.load_dotenv(_ENV_PATH)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pixel size & metadata helpers
# ---------------------------------------------------------------------------

def compute_pixel_size_um(params: dict) -> float:
    """
    Compute effective pixel size at the sample plane from acquisition parameters.

    Formula (matches squid_control/hardware/core.py):
        pixel_size_xy = sensor_pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))

    Args:
        params: dict loaded from "acquisition parameters.json"

    Returns:
        Effective pixel size in micrometres per pixel.
    """
    sensor_pixel_size_um: float = params["sensor_pixel_size_um"]
    tube_lens_mm: float = params["tube_lens_mm"]
    objective = params.get("objective", {})
    magnification: float = objective.get("magnification", 20)
    objective_tube_lens_mm: float = objective.get("tube_lens_f_mm", 180)

    pixel_size = sensor_pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
    logger.info(
        f"Pixel size: {pixel_size:.4f} µm/px "
        f"(sensor={sensor_pixel_size_um} µm, mag={magnification}x, "
        f"obj_tube={objective_tube_lens_mm}mm, body_tube={tube_lens_mm}mm)"
    )
    return pixel_size


def parse_configurations_xml(xml_path: Path) -> Dict[str, dict]:
    """
    Parse configurations.xml and return selected channels.

    Returns:
        Dict mapping channel Name → settings dict (exposure, intensity, etc.)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    channels = {}
    for mode in root.findall("mode"):
        if mode.get("Selected") == "1":
            name = mode.get("Name", "")
            channels[name] = {
                "exposure_time": float(mode.get("ExposureTime", 0)),
                "intensity": float(mode.get("IlluminationIntensity", 0)),
                "illumination_source": mode.get("IlluminationSource"),
                "analog_gain": float(mode.get("AnalogGain", 0)),
                "mode_id": mode.get("ID"),
            }
    logger.info(f"Selected channels from XML: {list(channels.keys())}")
    return channels


def get_channel_name_mapping() -> Dict[str, str]:
    """
    Build a mapping from filename canonical names to ZarrCanvas human names.

    Filenames use canonical_name (e.g. Fluorescence_405_nm_Ex).
    ZarrCanvas is indexed by human_name (e.g. "Fluorescence 405 nm Ex").

    Returns:
        Dict[canonical_name, human_name]
    """
    from squid_control.hardware.config import ChannelMapper

    mapping = {ch.canonical_name: ch.human_name for ch in ChannelMapper.CHANNELS.values()}
    logger.debug(f"Channel name mapping: {mapping}")
    return mapping


def sanitize_dataset_name(name: str) -> str:
    """
    Sanitize a dataset name for the Hypha artifact manager.
    Allowed: lowercase letters, numbers, hyphens.  Must start/end with alphanumeric.
    """
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


def make_dataset_name(data_folder: Path, override: Optional[str] = None) -> str:
    """Derive a sanitized dataset name from the folder name or an override."""
    if override:
        return sanitize_dataset_name(override)
    return sanitize_dataset_name(data_folder.name)


# ---------------------------------------------------------------------------
# Image loading & stitching
# ---------------------------------------------------------------------------

def _load_single_image_to_queue(
    img_file: Path,
    x_mm: float,
    y_mm: float,
    canvas,
    channel_mapping: Dict[str, str],
    available_channels: List[str],
) -> bool:
    """
    Load one BMP file and push it onto the canvas preprocessing queue.
    Designed to run inside a thread-pool worker.
    """
    try:
        # Extract canonical channel name from filename parts.
        # Filename format: {region}_{i}_{j}_{k}_{ChannelCanonical}.bmp
        parts = img_file.stem.split("_")
        channel_start = None
        for idx, part in enumerate(parts):
            if part in ("Fluorescence", "BF"):
                channel_start = idx
                break
        if channel_start is None:
            return False  # focus camera or unknown file

        canonical_name = "_".join(parts[channel_start:])
        human_name = channel_mapping.get(canonical_name, canonical_name)

        if human_name not in available_channels:
            return False

        zarr_channel_idx = canvas.get_zarr_channel_index(human_name)

        image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Failed to read image: {img_file.name}")
            return False

        queue_item = {
            "image": image.copy(),
            "x_mm": x_mm,
            "y_mm": y_mm,
            "channel_idx": zarr_channel_idx,
            "z_idx": 0,
            "timepoint": 0,
            "timestamp": time.time(),
            "quick_scan": False,
        }

        # Block-retry if queue is full to avoid unbounded memory growth
        while True:
            try:
                canvas.preprocessing_queue.put_nowait(queue_item)
                return True
            except Exception:
                time.sleep(0.2)

    except Exception as exc:
        logger.error(f"Error processing {img_file.name}: {exc}")
        return False


def load_and_stitch_images_sync(
    data_folder: Path,
    df_coords: pd.DataFrame,
    canvas,
    channel_mapping: Dict[str, str],
    num_workers: Optional[int] = None,
) -> int:
    """
    Load all BMP images from *data_folder* and push them onto the canvas
    preprocessing queue using a thread pool.

    Args:
        data_folder:     Folder containing .bmp files
        df_coords:       DataFrame with at minimum columns: region, x (mm), y (mm)
        canvas:          ZarrCanvas instance (stitching already started)
        channel_mapping: canonical_name → human_name
        num_workers:     Thread pool size (defaults to cpu_count - 1)

    Returns:
        Number of images successfully queued.
    """
    available_channels = list(canvas.channel_to_zarr_index.keys())
    logger.info(f"Canvas channels: {available_channels}")

    # Build region → (x_mm, y_mm) lookup
    coord_map: Dict[str, tuple] = {}
    for _, row in df_coords.iterrows():
        region = str(int(row["region"]))
        coord_map[region] = (float(row["x (mm)"]), float(row["y (mm)"]))

    # Collect image tasks: skip k != 0 (z-stacks) and non-channel files
    all_bmp = list(data_folder.glob("*.bmp"))
    logger.info(f"Total BMP files found: {len(all_bmp)}")

    tasks: List[tuple] = []
    skipped = 0
    for img_file in all_bmp:
        parts = img_file.stem.split("_")
        if len(parts) < 4:
            skipped += 1
            continue
        region_id = parts[0]
        try:
            file_k = int(parts[3])
        except ValueError:
            skipped += 1
            continue
        if file_k != 0:
            skipped += 1
            continue
        if region_id not in coord_map:
            skipped += 1
            continue
        x_mm, y_mm = coord_map[region_id]
        tasks.append((img_file, x_mm, y_mm))

    logger.info(f"Images to process: {len(tasks)} (skipped: {skipped})")

    if not tasks:
        logger.warning("No images to process — check data folder and coordinates CSV.")
        return 0

    num_workers = num_workers or max(1, (os.cpu_count() or 2) - 1)
    logger.info(f"Launching {num_workers} loader threads...")

    images_queued = 0
    counter_lock = Lock()
    last_log_time = time.time()

    def process_task(task):
        nonlocal images_queued, last_log_time
        img_file, x_mm, y_mm = task
        ok = _load_single_image_to_queue(
            img_file, x_mm, y_mm, canvas, channel_mapping, available_channels
        )
        with counter_lock:
            if ok:
                images_queued += 1
            now = time.time()
            if now - last_log_time >= 10.0:
                last_log_time = now
                logger.info(
                    f"Progress: {images_queued}/{len(tasks)} queued, "
                    f"preprocess_queue={canvas.preprocessing_queue.qsize()}"
                )

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pool.map(process_task, tasks)

    logger.info(f"Loading complete: {images_queued}/{len(tasks)} images queued")
    return images_queued


# ---------------------------------------------------------------------------
# Stitching completion wait
# ---------------------------------------------------------------------------

async def wait_for_stitching_completion(canvas, timeout_seconds: int = 7200) -> None:
    """
    Wait until the stitching pipeline is fully drained, then stop it cleanly.

    Polls both preprocessing_queue and zarr_write_queue. When both have been
    consistently empty for ~10 s, calls canvas.stop_stitching() which handles
    any remaining in-flight tasks and final zarr writes.
    """
    logger.info("Waiting for stitching pipeline to drain...")
    start = time.time()
    consecutive_empty = 0
    last_log_time = start

    while time.time() - start < timeout_seconds:
        pre_q = canvas.preprocessing_queue.qsize()
        write_q = canvas.zarr_write_queue.qsize()
        now = time.time()

        if pre_q == 0 and write_q == 0:
            consecutive_empty += 1
            if now - last_log_time >= 10.0:
                logger.info(
                    f"Queues empty (consecutive={consecutive_empty}/5), "
                    "waiting for final writes..."
                )
                last_log_time = now
            if consecutive_empty >= 5:
                break
        else:
            consecutive_empty = 0
            if now - last_log_time >= 15.0:
                logger.info(
                    f"Stitching in progress — preprocess_queue={pre_q}, "
                    f"zarr_write_queue={write_q}"
                )
                last_log_time = now

        await asyncio.sleep(2.0)

    # Proper shutdown: flushes any remaining items and awaits background tasks
    logger.info("Stopping stitching pipeline (final flush)...")
    await canvas.stop_stitching()
    logger.info(f"Stitching complete in {time.time() - start:.0f}s")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

async def upload_zarr_to_artifact_manager(
    zarr_path: Path,
    dataset_name: str,
    acquisition_settings: Optional[dict] = None,
    description: str = "",
    workspace: str = "reef-imaging",
    server_url: str = "https://hypha.aicell.io",
    token: Optional[str] = None,
) -> dict:
    """
    Upload a zarr directory to the Hypha Artifact Manager using hypha-artifact
    with parallel multipart uploads.

    Args:
        zarr_path:            Local path to the .zarr directory
        dataset_name:         Artifact alias within the workspace
        acquisition_settings: Optional metadata stored as manifest.json
        description:          Human-readable description
        workspace:            Hypha workspace (default: "reef-imaging")
        server_url:           Hypha server URL
        token:                Auth token (falls back to REEF_WORKSPACE_TOKEN env var)

    Returns:
        dict with upload result information
    """
    from hypha_artifact import AsyncHyphaArtifact

    token = token or os.environ.get("REEF_WORKSPACE_TOKEN")
    if not token:
        raise RuntimeError(
            "REEF_WORKSPACE_TOKEN environment variable not set. "
            "Export it before running: export REEF_WORKSPACE_TOKEN=..."
        )

    artifact_id = f"{workspace}/{dataset_name}"
    logger.info(f"Uploading to artifact: {artifact_id}")

    async with AsyncHyphaArtifact(
        artifact_id=dataset_name,
        workspace=workspace,
        token=token,
        server_url=server_url,
    ) as artifact:
        # Create or overwrite the artifact
        try:
            await artifact.create(
                manifest={
                    "name": dataset_name,
                    "description": description,
                    "type": "ome-zarr-dataset",
                },
                overwrite=True,
            )
            logger.info(f"Artifact created: {artifact_id}")
        except Exception as exc:
            err = str(exc).lower()
            if "already" in err or "exist" in err:
                logger.info(f"Artifact exists, staging for overwrite: {artifact_id}")
                await artifact.edit(stage=True)
            else:
                raise

        await artifact.edit(stage=True)

        # Upload zarr directory with parallel multipart
        logger.info(f"Uploading zarr: {zarr_path}")
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
        logger.info("Zarr directory uploaded.")

        # Upload manifest with acquisition settings
        if acquisition_settings:
            manifest_content = json.dumps(
                {
                    "name": dataset_name,
                    "description": description,
                    "acquisition_settings": acquisition_settings,
                },
                indent=2,
            )
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                tmp.write(manifest_content)
                tmp_path = tmp.name
            try:
                await artifact.put(tmp_path, "manifest.json")
                logger.info("manifest.json uploaded.")
            finally:
                os.unlink(tmp_path)

        await artifact.commit(
            comment=description or f"Uploaded OME-Zarr dataset: {dataset_name}"
        )
        logger.info(f"Upload committed: {artifact_id}")

    return {
        "success": True,
        "artifact_id": artifact_id,
        "dataset_name": dataset_name,
        "server_url": server_url,
        "workspace": workspace,
    }


# ---------------------------------------------------------------------------
# Standalone pipeline
# ---------------------------------------------------------------------------

async def process_data_folder(
    data_folder: Path,
    output_dir: Path,
    dataset_name: str,
    pixel_size_um: Optional[float] = None,
    upload: bool = True,
    cleanup: bool = True,
    token: Optional[str] = None,
    workspace: str = "reef-imaging",
    server_url: str = "https://hypha.aicell.io",
    num_loader_workers: Optional[int] = None,
) -> dict:
    """
    Full pipeline: parse metadata → create OME-Zarr → upload.

    Steps:
    1. Read acquisition parameters and compute pixel size (or use provided value)
    2. Parse configurations.xml for active channels
    3. Parse coordinates.csv for stage positions
    4. Create OME-Zarr canvas and run stitching pipeline
    5. Upload to Hypha artifact manager

    Supports resume: if a .done marker exists in the output directory the zarr
    creation step is skipped and the script goes straight to upload.

    Args:
        data_folder:        Experiment folder (contains acquisition params, XML, 0/ subfolder)
        output_dir:         Parent directory for temporary zarr output
        dataset_name:       Artifact alias (already sanitized)
        pixel_size_um:      Override pixel size; computed from params JSON if None
        upload:             Whether to upload after stitching
        cleanup:            Whether to delete temp zarr after successful upload
        token:              Hypha auth token
        workspace:          Hypha workspace
        server_url:         Hypha server URL
        num_loader_workers: Image-loader thread count

    Returns:
        dict with keys: success, dataset_name, zarr_path, images_queued, upload_result
    """
    from squid_control.stitching.zarr_canvas import ExperimentManager

    data_folder = Path(data_folder)
    images_folder = data_folder / "0"

    if not images_folder.is_dir():
        raise FileNotFoundError(f"Images subfolder not found: {images_folder}")

    # 1. Acquisition parameters
    params_file = data_folder / "acquisition parameters.json"
    if not params_file.exists():
        raise FileNotFoundError(f"acquisition parameters.json not found in {data_folder}")
    with open(params_file) as f:
        acq_params = json.load(f)
    logger.info(f"Acquisition parameters: {acq_params}")

    if pixel_size_um is None:
        pixel_size_um = compute_pixel_size_um(acq_params)

    # 2. Channel configuration
    xml_file = data_folder / "configurations.xml"
    if not xml_file.exists():
        raise FileNotFoundError(f"configurations.xml not found in {data_folder}")
    xml_channels = parse_configurations_xml(xml_file)
    channel_mapping = get_channel_name_mapping()

    # 3. Coordinate positions
    csv_file = images_folder / "coordinates.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"coordinates.csv not found in {images_folder}")
    df_coords = pd.read_csv(csv_file)
    k_col = (
        "k" if "k" in df_coords.columns
        else "z_level" if "z_level" in df_coords.columns
        else None
    )
    if k_col:
        df_coords = df_coords[df_coords[k_col] == 0].copy()
    logger.info(
        f"Coordinates: {len(df_coords)} positions, "
        f"x=[{df_coords['x (mm)'].min():.2f}, {df_coords['x (mm)'].max():.2f}]mm "
        f"y=[{df_coords['y (mm)'].min():.2f}, {df_coords['y (mm)'].max():.2f}]mm"
    )

    # 4. Output path & resume support
    zarr_output_dir = Path(output_dir) / f"zarr_{dataset_name}"
    done_marker = zarr_output_dir / ".done"
    zarr_output_dir.mkdir(parents=True, exist_ok=True)
    images_queued = 0

    if done_marker.exists():
        logger.info("Found .done marker — zarr already built, skipping stitching.")
        zarr_path = zarr_output_dir / "data.zarr"
    else:
        # 5. Create ZarrCanvas via ExperimentManager
        logger.info(f"Creating ZarrCanvas at: {zarr_output_dir}")
        exp_manager = ExperimentManager(
            base_path=str(zarr_output_dir),
            pixel_size_xy_um=pixel_size_um,
            stage_limits={
                "x_positive": 120.0,
                "x_negative": 0.0,
                "y_positive": 86.0,
                "y_negative": 0.0,
            },
        )

        # "." puts the canvas directly at zarr_output_dir/data.zarr
        exp_manager.create_experiment(".")
        canvas = exp_manager.get_canvas(".", initialize_new=True)
        zarr_path = canvas.zarr_path

        # 6. Start async stitching pipeline
        await canvas.start_stitching()
        logger.info("Stitching pipeline started.")

        # 7. Load images in background thread (non-blocking for event loop)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            images_queued = await loop.run_in_executor(
                executor,
                load_and_stitch_images_sync,
                images_folder,
                df_coords,
                canvas,
                channel_mapping,
                num_loader_workers,
            )

        logger.info(f"All {images_queued} images pushed to queue.")

        # 8. Wait for stitching to complete
        await wait_for_stitching_completion(canvas)

        canvas.activate_channels_with_data()
        logger.info("Channel activation metadata updated.")

        canvas_info = canvas.get_export_info()
        total_size_mb = canvas_info.get("total_size_mb", 0)
        logger.info(f"Zarr created: {zarr_path} ({total_size_mb:.1f} MB)")

        done_marker.write_text(
            f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"images_queued={images_queued}\n"
            f"zarr_size_mb={total_size_mb:.1f}\n"
        )

    # 9. Upload
    upload_result = None
    if upload:
        if not zarr_path.exists():
            raise FileNotFoundError(
                f"Expected zarr at {zarr_path} but it does not exist."
            )
        logger.info(f"Starting upload of '{dataset_name}' to {workspace}...")
        upload_result = await upload_zarr_to_artifact_manager(
            zarr_path=zarr_path,
            dataset_name=dataset_name,
            acquisition_settings={
                "source_folder": str(data_folder),
                "acquisition_params": acq_params,
                "channels": list(xml_channels.keys()),
                "num_positions": len(df_coords),
                "pixel_size_um": pixel_size_um,
            },
            description=(
                f"OME-Zarr dataset imported from {data_folder.name}. "
                f"{len(df_coords)} positions, {len(xml_channels)} channels."
            ),
            workspace=workspace,
            server_url=server_url,
            token=token,
        )

        if cleanup:
            logger.info(f"Cleaning up temporary zarr: {zarr_output_dir}")
            shutil.rmtree(zarr_output_dir, ignore_errors=True)

    return {
        "success": True,
        "dataset_name": dataset_name,
        "zarr_path": str(zarr_path),
        "images_queued": images_queued,
        "upload_result": upload_result,
    }


# ---------------------------------------------------------------------------
# OfflineProcessor — used by MicroscopeHyphaService
# ---------------------------------------------------------------------------

class OfflineProcessor:
    """
    Handles offline stitching and uploading of time-lapse microscopy data.

    Used by MicroscopeHyphaService to process experiment folders stored at
    CONFIG.DEFAULT_SAVING_PATH.  The service calls::

        processor = OfflineProcessor(squid_controller, artifact_manager, service_id)
        result = await processor.stitch_and_upload_timelapse(experiment_id, ...)
    """

    def __init__(self, squid_controller, zarr_artifact_manager=None,
                 service_id: Optional[str] = None):
        self.squid_controller = squid_controller
        self.zarr_artifact_manager = zarr_artifact_manager
        self.service_id = service_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stitch_and_upload_timelapse(
        self,
        experiment_id: str,
        upload_immediately: bool = True,
        cleanup_temp_files: bool = True,
        max_concurrent_runs: int = 1,
        use_parallel_wells: bool = True,
    ) -> dict:
        """
        Stitch and upload all experiment folders that match *experiment_id*.

        Finds folders under CONFIG.DEFAULT_SAVING_PATH whose names start with
        experiment_id followed by '_' or '-', then processes each one.

        Args:
            experiment_id:        Prefix used to locate experiment folders.
            upload_immediately:   Upload each dataset right after stitching.
            cleanup_temp_files:   Delete temporary zarr after successful upload.
            max_concurrent_runs:  Ignored (kept for API compatibility).
            use_parallel_wells:   Ignored (kept for API compatibility).

        Returns:
            dict with keys: success, processed_runs, failed_runs, total_datasets,
                            processing_time_seconds
        """
        results: dict = {
            "success": True,
            "experiment_id": experiment_id,
            "processed_runs": [],
            "failed_runs": [],
            "total_datasets": 0,
            "start_time": time.time(),
        }

        try:
            experiment_folders = self._find_experiment_folders(experiment_id)
            if not experiment_folders:
                results["success"] = False
                results["message"] = f"No experiment folders found for ID: {experiment_id}"
                results["processing_time_seconds"] = time.time() - results["start_time"]
                return results

            for folder in experiment_folders:
                logger.info(f"Processing folder: {folder.name}")
                run_result = await self._process_folder(
                    folder,
                    experiment_id,
                    upload_immediately,
                    cleanup_temp_files,
                )
                if run_result["success"]:
                    results["processed_runs"].append(run_result)
                    results["total_datasets"] += 1
                else:
                    results["failed_runs"].append(run_result)
                    logger.error(f"Failed: {folder.name} — {run_result.get('error')}")

        except Exception as exc:
            results["success"] = False
            results["error"] = str(exc)
            logger.error(f"stitch_and_upload_timelapse failed: {exc}", exc_info=True)

        results["processing_time_seconds"] = time.time() - results["start_time"]
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_experiment_folders(self, experiment_id: str) -> List[Path]:
        """
        Find experiment folders matching experiment_id under CONFIG.DEFAULT_SAVING_PATH.
        """
        from squid_control.hardware.config import CONFIG

        base_path = Path(CONFIG.DEFAULT_SAVING_PATH)
        if not base_path.exists():
            raise FileNotFoundError(f"Saving path does not exist: {base_path}")

        folders = sorted(
            set(base_path.glob(f"{experiment_id}_*"))
            | set(base_path.glob(f"{experiment_id}-*"))
        )
        # Keep only directories with a "0" subfolder
        valid = [
            f for f in folders
            if f.is_dir() and (f / "0").is_dir()
        ]
        logger.info(f"Found {len(valid)} valid folder(s) for '{experiment_id}'")
        return valid

    def _get_pixel_size(self, experiment_folder: Path) -> float:
        """
        Get pixel size — uses squid_controller if available, otherwise reads
        acquisition parameters.json from the experiment folder.
        """
        if self.squid_controller is not None:
            return float(self.squid_controller.pixel_size_xy)

        params_file = experiment_folder / "acquisition parameters.json"
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            return compute_pixel_size_um(params)

        raise FileNotFoundError(
            f"No squid_controller and no acquisition parameters.json in {experiment_folder}"
        )

    def _get_temp_dir(self, experiment_folder: Path) -> Path:
        """Return a consistent temp directory for stitching (supports resume)."""
        from squid_control.hardware.config import CONFIG

        base = Path(CONFIG.DEFAULT_SAVING_PATH)
        if base.exists():
            return base / f"offline_stitch_{experiment_folder.name}"
        return Path(tempfile.mkdtemp(prefix=f"offline_stitch_{experiment_folder.name}_"))

    async def _process_folder(
        self,
        experiment_folder: Path,
        experiment_id: str,
        upload: bool,
        cleanup: bool,
    ) -> dict:
        """Process a single experiment folder."""
        try:
            pixel_size_um = self._get_pixel_size(experiment_folder)
            temp_dir = self._get_temp_dir(experiment_folder)
            dataset_name = make_dataset_name(experiment_folder, experiment_id)

            result = await process_data_folder(
                data_folder=experiment_folder,
                output_dir=temp_dir.parent,
                dataset_name=dataset_name,
                pixel_size_um=pixel_size_um,
                upload=upload,
                cleanup=cleanup,
                workspace="reef-imaging",
                server_url="https://hypha.aicell.io",
            )
            result["experiment_folder"] = experiment_folder.name
            return result

        except Exception as exc:
            logger.error(
                f"Error processing {experiment_folder.name}: {exc}", exc_info=True
            )
            return {
                "success": False,
                "experiment_folder": experiment_folder.name,
                "error": str(exc),
            }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Import raw microscopy images into OME-Zarr format and upload "
            "to the Hypha Artifact Manager (reef-imaging workspace).\n\n"
            "Usage example:\n"
            "  conda run -n squid python scripts/offline_processing.py \\\n"
            "    --data-folder /data/hpa-sample-full_2026-03-10_17-48-13.841946"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-folder",
        required=True,
        help=(
            "Path to the experiment folder. Must contain: "
            "'acquisition parameters.json', 'configurations.xml', "
            "and a '0/' subdirectory with BMP files and coordinates.csv."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Artifact alias in the workspace. Derived from folder name if not given.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for temporary zarr output (default: parent of --data-folder).",
    )
    parser.add_argument(
        "--workspace",
        default="reef-imaging",
        help="Hypha workspace to upload into (default: reef-imaging).",
    )
    parser.add_argument(
        "--server-url",
        default="https://hypha.aicell.io",
        help="Hypha server URL (default: https://hypha.aicell.io).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Build zarr locally but skip upload.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary zarr files after upload.",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=None,
        help="Number of image-loader threads (default: cpu_count - 1).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hypha auth token. Falls back to REEF_WORKSPACE_TOKEN env var.",
    )
    return parser.parse_args()


async def _async_main() -> None:
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()

    data_folder = Path(args.data_folder).resolve()
    if not data_folder.is_dir():
        logger.error(f"Data folder does not exist: {data_folder}")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_folder.parent
    dataset_name = make_dataset_name(data_folder, args.dataset_name)

    logger.info("=" * 60)
    logger.info("Squid OME-Zarr Import & Upload")
    logger.info(f"  data_folder : {data_folder}")
    logger.info(f"  output_dir  : {output_dir}")
    logger.info(f"  dataset_name: {dataset_name}")
    logger.info(f"  workspace   : {args.workspace}")
    logger.info(f"  upload      : {not args.no_upload}")
    logger.info("=" * 60)

    t0 = time.time()
    result = await process_data_folder(
        data_folder=data_folder,
        output_dir=output_dir,
        dataset_name=dataset_name,
        upload=not args.no_upload,
        cleanup=not args.no_cleanup,
        token=args.token,
        workspace=args.workspace,
        server_url=args.server_url,
        num_loader_workers=args.loader_workers,
    )
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info(f"Done in {elapsed:.0f}s")
    logger.info(f"  dataset_name  : {result['dataset_name']}")
    logger.info(f"  images_queued : {result['images_queued']}")
    if result.get("upload_result"):
        ur = result["upload_result"]
        logger.info(f"  artifact_id   : {ur.get('artifact_id')}")
        logger.info(f"  server_url    : {ur.get('server_url')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(_async_main())
