"""IIIF Presentation and Image API access for the Kadaster aerial photographs.

Nothing here downloads a full-resolution scan. Each frame is read first as a
reduced overview (``full/<w>,``) and afterwards only as the regions the
refinement asks for (``x,y,w,h/<w>,``), every response cached on disk by URL.
The client keeps a per-image ledger of the pixels it fetched so a run can report
how small a fraction of each scan it actually needed.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IIIFImage:
    """One painted image on a canvas: its Image API service and where it came from."""

    service: str
    service_type: str
    width: int
    height: int
    canvas: str
    manifest: str
    metadata: tuple[tuple[str, str], ...] = ()

    @property
    def image_id(self) -> str:
        return self.service.rstrip("/").rsplit("/", 1)[-1]

    def meta(self, label: str, default: str | None = None) -> str | None:
        for key, value in self.metadata:
            if key == label:
                return value
        return default


def _label(value) -> str:
    """Flatten a v2 string/list or v3 language map label to plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_label(item) for item in value)
    if isinstance(value, dict):
        if "@value" in value:
            return str(value["@value"])
        return " ".join(_label(item) for item in value.values())
    return "" if value is None else str(value)


def _metadata(entries) -> tuple[tuple[str, str], ...]:
    return tuple((_label(entry.get("label")), _label(entry.get("value"))) for entry in entries or ())


def _service_type(service: dict) -> str:
    kind = service.get("type") or service.get("@type") or ""
    if kind in ("ImageService1", "ImageService2", "ImageService3"):
        return kind
    context = json.dumps(service.get("@context", "")) + json.dumps(service.get("profile", ""))
    return "ImageService3" if "image/3" in context else "ImageService2"


def _first_service(resource: dict) -> dict:
    service = resource.get("service")
    if isinstance(service, list):
        service = service[0] if service else None
    if not isinstance(service, dict):
        raise ValueError(f"Painted resource {resource.get('id') or resource.get('@id')} has no image service")
    return service


def parse_manifest(document: dict, manifest_url: str | None = None) -> list[IIIFImage]:
    """Return the painted images of a Presentation 2 or 3 manifest in canvas order."""
    manifest = manifest_url or document.get("id") or document.get("@id") or ""
    images = []
    if "sequences" in document:
        canvases = [c for sequence in document["sequences"] for c in sequence.get("canvases", [])]
        for canvas in canvases:
            for annotation in canvas.get("images", []):
                resource = annotation["resource"]
                service = _first_service(resource)
                images.append(
                    IIIFImage(
                        service=service.get("@id") or service["id"],
                        service_type=_service_type(service),
                        width=int(service.get("width") or canvas["width"]),
                        height=int(service.get("height") or canvas["height"]),
                        canvas=canvas.get("@id", ""),
                        manifest=manifest,
                        metadata=_metadata(canvas.get("metadata")),
                    )
                )
        return images
    for canvas in document.get("items", []):
        for page in canvas.get("items", []):
            for annotation in page.get("items", []):
                body = annotation["body"]
                body = body[0] if isinstance(body, list) else body
                service = _first_service(body)
                images.append(
                    IIIFImage(
                        service=service.get("id") or service["@id"],
                        service_type=_service_type(service),
                        width=int(service.get("width") or canvas["width"]),
                        height=int(service.get("height") or canvas["height"]),
                        canvas=canvas.get("id", ""),
                        manifest=manifest,
                        metadata=_metadata(canvas.get("metadata")),
                    )
                )
    return images


def region_url(image: IIIFImage, x: int, y: int, w: int, h: int, width: int | None) -> str:
    """Image API request for a pixel region scaled to ``width`` (None = native size)."""
    full = x == 0 and y == 0 and w == image.width and h == image.height
    region = "full" if full else f"{x},{y},{w},{h}"
    if width is None or width >= w:
        size = "max" if image.service_type == "ImageService3" else "full"
    else:
        size = f"{width},"
    return f"{image.service.rstrip('/')}/{region}/{size}/0/default.jpg"


@dataclass
class FetchLedger:
    requests: int = 0
    cache_hits: int = 0
    bytes_downloaded: int = 0
    # Per image service: delivered pixels and the full-resolution area those
    # requests covered, against the scan's own pixel count.
    delivered: dict[str, int] = field(default_factory=dict)
    source_area: dict[str, int] = field(default_factory=dict)
    full_area: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        images = {
            key: {
                "delivered_px": self.delivered.get(key, 0),
                "full_resolution_px": self.full_area[key],
                "delivered_fraction": round(self.delivered.get(key, 0) / self.full_area[key], 4),
            }
            for key in sorted(self.full_area)
        }
        delivered = sum(self.delivered.values())
        full = sum(self.full_area.values()) or 1
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "bytes_downloaded": self.bytes_downloaded,
            "delivered_px": delivered,
            "full_resolution_px": full,
            "delivered_fraction": round(delivered / full, 4),
            "images": images,
        }


class IIIFClient:
    def __init__(self, cache_dir: Path, *, timeout: float = 120.0, retries: int = 4, tile_px: int = 1024):
        import requests

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.timeout = timeout
        self.retries = retries
        self.tile_px = tile_px
        self.ledger = FetchLedger()

    def _get(self, url: str, suffix: str) -> bytes:
        data, _ = self._fetch(url, suffix)
        return data

    def _fetch(self, url: str, suffix: str) -> tuple[bytes, bool]:
        """Response bytes and whether they came over the network."""
        path = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + suffix)
        if path.is_file():
            self.ledger.cache_hits += 1
            return path.read_bytes(), False
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code < 500:
                    response.raise_for_status()
                    break
            except Exception:  # noqa: BLE001 - retried below, re-raised on the last attempt
                if attempt == self.retries - 1:
                    raise
            time.sleep(2**attempt)
        else:
            raise RuntimeError(f"IIIF request kept failing: {url}")
        self.ledger.requests += 1
        self.ledger.bytes_downloaded += len(response.content)
        path.write_bytes(response.content)
        return response.content, True

    def manifest(self, url: str, *, refresh: bool = False) -> list[IIIFImage]:
        if refresh:
            (self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")).unlink(missing_ok=True)
        return parse_manifest(json.loads(self._get(url, ".json")), url)

    def _request(self, image: IIIFImage, x: int, y: int, w: int, h: int, width: int | None):
        import cv2
        import numpy as np

        data, fetched = self._fetch(region_url(image, x, y, w, h, width), ".jpg")
        array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if array is None:
            raise RuntimeError(f"Could not decode IIIF response for {image.image_id}")
        # Only network deliveries count; a cached tile costs the server nothing.
        key = image.image_id
        self.ledger.full_area[key] = image.width * image.height
        if fetched:
            self.ledger.delivered[key] = self.ledger.delivered.get(key, 0) + array.size
            self.ledger.source_area[key] = self.ledger.source_area.get(key, 0) + w * h
        return array

    def region(self, image: IIIFImage, x: int, y: int, w: int, h: int, width: int | None = None):
        """Greyscale region scaled to ``width``, plus the (sx, sy) scale actually delivered.

        Read from fixed tiles at power-of-two scale levels, the way a IIIF viewer
        reads a deep-zoom image. Overlapping requests then share tiles, and a
        frame never costs more than the full pixel count of its finest level used.
        """
        import cv2
        import numpy as np

        x, y = max(0, int(x)), max(0, int(y))
        w, h = min(int(w), image.width - x), min(int(h), image.height - y)
        if w <= 0 or h <= 0:
            raise ValueError(f"Empty IIIF region requested on {image.image_id}")
        factor = 1.0 if width is None else min(1.0, max(1, int(width)) / w)
        level = 1
        while level < 64 and 1.0 / (2 * level) >= factor:
            level *= 2
        span = self.tile_px * level
        tx0, ty0 = (x // span) * span, (y // span) * span
        tx1, ty1 = x + w, y + h
        canvas = np.zeros((math.ceil((ty1 - ty0) / level) + 1, math.ceil((tx1 - tx0) / level) + 1), np.uint8)
        for ty in range(ty0, ty1, span):
            for tx in range(tx0, tx1, span):
                tw, th = min(span, image.width - tx), min(span, image.height - ty)
                tile = self._request(image, tx, ty, tw, th, math.ceil(tw / level))
                r, c = (ty - ty0) // level, (tx - tx0) // level
                tile = tile[: canvas.shape[0] - r, : canvas.shape[1] - c]
                canvas[r : r + tile.shape[0], c : c + tile.shape[1]] = tile
        crop = canvas[
            round((y - ty0) / level) : round((y - ty0 + h) / level),
            round((x - tx0) / level) : round((x - tx0 + w) / level),
        ]
        out_w = max(1, round(w * factor))
        out_h = max(1, round(h * factor))
        if (out_w, out_h) != (crop.shape[1], crop.shape[0]):
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return crop, (crop.shape[1] / w, crop.shape[0] / h)

    def overview(self, image: IIIFImage, width: int):
        """One reduced ``full/<w>,`` request: the first thing read of every frame."""
        width = min(width, image.width)
        array = self._request(image, 0, 0, image.width, image.height, width)
        return array, (array.shape[1] / image.width, array.shape[0] / image.height)


def overview_width_for(image: IIIFImage, longest_side: int) -> int:
    """Width that gives ``longest_side`` pixels on the scan's longer edge."""
    if image.width >= image.height:
        return longest_side
    return max(1, math.floor(longest_side * image.width / image.height))
