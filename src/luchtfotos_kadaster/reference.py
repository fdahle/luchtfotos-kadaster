"""RD New raster grids and the PDOK orthophoto reference.

Every comparison in this subproject happens on a north-up EPSG:28992 grid: the
reference is requested for exactly that grid from PDOK's WMS, and a photograph
that is already georeferenced is warped onto the same grid from IIIF regions.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RasterGrid:
    """North-up grid; integer pixel coordinates sit on pixel centres (OpenCV)."""

    xmin: float
    ymax: float
    resolution: float
    width: int
    height: int

    @classmethod
    def around(cls, x: float, y: float, half_width: float, resolution: float) -> "RasterGrid":
        size = max(1, int(round(2 * half_width / resolution)))
        return cls(x - size * resolution / 2, y + size * resolution / 2, resolution, size, size)

    @classmethod
    def covering(cls, bounds, resolution: float, margin: float = 0.0) -> "RasterGrid":
        xmin, ymin, xmax, ymax = bounds
        xmin, ymin, xmax, ymax = xmin - margin, ymin - margin, xmax + margin, ymax + margin
        width = max(1, math.ceil((xmax - xmin) / resolution))
        height = max(1, math.ceil((ymax - ymin) / resolution))
        return cls(xmin, ymax, resolution, width, height)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.xmin,
            self.ymax - self.height * self.resolution,
            self.xmin + self.width * self.resolution,
            self.ymax,
        )

    def pixel_to_world(self):
        """3x3 matrix mapping (col, row, 1) to (x, y, 1)."""
        import numpy as np

        r = self.resolution
        return np.array(
            [[r, 0.0, self.xmin + 0.5 * r], [0.0, -r, self.ymax - 0.5 * r], [0.0, 0.0, 1.0]]
        )

    def world_file(self) -> str:
        r = self.resolution
        return f"{r}\n0\n0\n{-r}\n{self.xmin + 0.5 * r}\n{self.ymax - 0.5 * r}\n"


class PdokReference:
    """Greyscale PDOK luchtfoto WMS renders, tiled and cached per request URL."""

    def __init__(self, url: str, layer: str, cache_dir: Path, *, max_request_px: int = 2048):
        import requests

        self.url = url
        self.layer = layer
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_request_px = max_request_px
        self.session = requests.Session()
        self.requests = 0

    def _tile(self, bounds, width: int, height: int):
        import cv2
        import numpy as np

        params = {
            "service": "WMS",
            "request": "GetMap",
            "version": "1.3.0",
            "layers": self.layer,
            "styles": "",
            "crs": "EPSG:28992",
            "bbox": ",".join(f"{value:.3f}" for value in bounds),
            "width": str(width),
            "height": str(height),
            "format": "image/jpeg",
        }
        key = hashlib.sha1(repr(sorted(params.items())).encode() + self.url.encode()).hexdigest()
        path = self.cache_dir / f"{key}.jpg"
        if not path.is_file():
            for attempt in range(4):
                response = self.session.get(self.url, params=params, timeout=180)
                if response.ok and response.headers.get("content-type", "").startswith("image/"):
                    break
                if attempt == 3:
                    raise RuntimeError(f"PDOK WMS request failed ({response.status_code}): {response.url}")
                time.sleep(2**attempt)
            self.requests += 1
            path.write_bytes(response.content)
        array = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if array is None or array.shape != (height, width):
            raise RuntimeError(f"PDOK WMS returned an unreadable tile: {path}")
        return array

    def render(self, grid: RasterGrid):
        import numpy as np

        out = np.zeros((grid.height, grid.width), np.uint8)
        step = self.max_request_px
        r = grid.resolution
        for row in range(0, grid.height, step):
            for col in range(0, grid.width, step):
                w, h = min(step, grid.width - col), min(step, grid.height - row)
                xmin = grid.xmin + col * r
                ymax = grid.ymax - row * r
                out[row : row + h, col : col + w] = self._tile((xmin, ymax - h * r, xmin + w * r, ymax), w, h)
        return out
