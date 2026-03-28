import asyncio
import json
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote


JSON_METADATA_FILES = {".zgroup", ".zarray", ".zattrs"}
EXPOSE_HEADERS = b"Content-Length, Content-Range, Accept-Ranges"
ALLOW_HEADERS = b"Range"


def create_simulation_zarr_asgi_app(
    get_zarr_dataset_path: Callable[[], Optional[str]],
    logger,
):
    """Create a lightweight ASGI app that serves the active simulation Zarr store."""

    async def serve(args, context=None):
        scope = args["scope"]
        receive = args["receive"]
        send = args["send"]

        if scope["type"] != "http":
            return

        method = scope.get("method", "GET").upper()
        if method == "OPTIONS":
            await _send_response(send, 200, _build_cors_headers())
            return

        if method not in {"GET", "HEAD"}:
            await _send_error(
                send,
                405,
                f"Method not allowed: {method}",
            )
            return

        dataset_path = get_zarr_dataset_path()
        if not dataset_path:
            await _send_error(
                send, 503, "Simulation Zarr dataset path is not available."
            )
            return

        zarr_base = Path(dataset_path).resolve()
        request_path = _get_request_path(scope)

        try:
            resolved_path = validate_zarr_path(request_path, zarr_base)
        except PermissionError:
            await _send_error(
                send, 403, "Access denied: Path outside zarr dataset directory."
            )
            return

        if not resolved_path.exists():
            if request_path == ".zgroup":
                synthetic_response = await _maybe_build_synthetic_zgroup(zarr_base)
                if synthetic_response is not None:
                    await _send_bytes(
                        send,
                        content=synthetic_response,
                        content_type="application/json",
                        method=method,
                    )
                    return

            await _send_error(
                send,
                404,
                f"File not found: {request_path}",
            )
            return

        if not resolved_path.is_file():
            await _send_error(send, 404, f"Unsupported Zarr path: {request_path}")
            return

        range_header = _get_header(scope, b"range")
        try:
            await _send_file(
                send,
                resolved_path,
                method=method,
                range_header=range_header,
            )
        except ValueError:
            await _send_error(
                send,
                416,
                "Invalid Range header.",
                extra_headers=[
                    (
                        b"content-range",
                        f"bytes */{resolved_path.stat().st_size}".encode("ascii"),
                    )
                ],
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error serving simulation zarr file %s: %s", request_path, exc)
            await _send_error(send, 500, str(exc))

    return serve


def validate_zarr_path(file_path: str, zarr_base: Path) -> Path:
    """Resolve a file path inside a Zarr dataset and reject path traversal."""
    normalized_path = file_path.lstrip("/")
    if not normalized_path:
        normalized_path = ".zgroup"

    resolved_path = (zarr_base / normalized_path).resolve()
    try:
        resolved_path.relative_to(zarr_base)
    except ValueError as exc:
        raise PermissionError from exc
    return resolved_path


def _get_request_path(scope) -> str:
    raw_path = scope.get("path", "") or ""
    path = unquote(raw_path).lstrip("/")
    return path or ".zgroup"


def _get_header(scope, header_name: bytes) -> Optional[str]:
    for name, value in scope.get("headers", []):
        if name.lower() == header_name:
            return value.decode("latin-1")
    return None


def _build_cors_headers():
    return [
        (b"access-control-allow-origin", b"*"),
        (b"access-control-allow-methods", b"GET, HEAD, OPTIONS"),
        (b"access-control-allow-headers", ALLOW_HEADERS),
        (b"access-control-expose-headers", EXPOSE_HEADERS),
        (b"cache-control", b"no-store"),
        (b"pragma", b"no-cache"),
    ]


def _build_response_headers(content_type: str, content_length: int, extra_headers=None):
    headers = [
        (b"content-type", content_type.encode("ascii")),
        (b"content-length", str(content_length).encode("ascii")),
        (b"accept-ranges", b"bytes"),
        *_build_cors_headers(),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    return headers


async def _send_response(send, status: int, headers, body: bytes = b""):
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        }
    )


async def _send_error(send, status: int, message: str, extra_headers=None):
    body = json.dumps({"detail": message}).encode("utf-8")
    headers = _build_response_headers(
        "application/json",
        len(body),
        extra_headers=extra_headers,
    )
    await _send_response(send, status, headers, body)


async def _send_bytes(send, content: bytes, content_type: str, method: str = "GET"):
    headers = _build_response_headers(content_type, len(content))
    body = b"" if method == "HEAD" else content
    await _send_response(send, 200, headers, body)


async def _maybe_build_synthetic_zgroup(zarr_base: Path) -> Optional[bytes]:
    zattrs_path = zarr_base / ".zattrs"
    if zarr_base.exists() and zattrs_path.exists():
        return json.dumps({"zarr_format": 2}).encode("utf-8")
    return None


async def _send_file(send, file_path: Path, method: str, range_header: Optional[str]):
    file_size = file_path.stat().st_size
    content_type = (
        "application/json"
        if file_path.name in JSON_METADATA_FILES
        else "application/octet-stream"
    )

    if range_header:
        start, end = _parse_range_header(range_header, file_size)
        length = end - start + 1
        headers = _build_response_headers(
            content_type,
            length,
            extra_headers=[
                (
                    b"content-range",
                    f"bytes {start}-{end}/{file_size}".encode("ascii"),
                )
            ],
        )
        body = (
            b""
            if method == "HEAD"
            else await asyncio.to_thread(_read_file_slice, file_path, start, length)
        )
        await _send_response(send, 206, headers, body)
        return

    headers = _build_response_headers(content_type, file_size)
    body = b"" if method == "HEAD" else await asyncio.to_thread(_read_file, file_path)
    await _send_response(send, 200, headers, body)


def _read_file(file_path: Path) -> bytes:
    return file_path.read_bytes()


def _read_file_slice(file_path: Path, start: int, length: int) -> bytes:
    with file_path.open("rb") as handle:
        handle.seek(start)
        return handle.read(length)


def _parse_range_header(range_header: str, file_size: int):
    if not range_header.startswith("bytes="):
        raise ValueError("Only byte ranges are supported.")

    first_range = range_header.split("=", 1)[1].split(",", 1)[0].strip()
    if "-" not in first_range:
        raise ValueError("Invalid byte range.")

    start_text, end_text = first_range.split("-", 1)

    if start_text == "":
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("Invalid suffix range.")
        start = max(file_size - suffix_length, 0)
        end = file_size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1

    if start < 0 or end < start or start >= file_size:
        raise ValueError("Requested range is not satisfiable.")

    end = min(end, file_size - 1)
    return start, end
