"""RoMa v2 dense matching with a content-addressed cache.

RoMa v2 (``pip install romav2``) is a dense matcher: it holds up across
decades of change between a historical photograph and a modern orthophoto,
and across arbitrary in-plane rotation, which sparse matchers on 90 degree
steps do not. Results are cached under a hash of both images' bytes and
the matcher settings, so a rerun on unchanged renders costs nothing and a
changed render can never be served a stale result.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Bump when matching semantics change, so stale cache entries stop matching.
CACHE_SCHEMA = 1


@dataclass(frozen=True)
class MatchRequest:
    image0: Path
    image1: Path
    # image1 is rotated by this angle (counter-clockwise, degrees) before
    # matching; returned points are in the unrotated image1.
    rotation_deg: float = 0.0


@dataclass(frozen=True)
class Matches:
    """Pixel correspondences, (x, y) with a top-left origin, and their [0, 1] confidence."""

    points0: np.ndarray
    points1: np.ndarray
    confidence: np.ndarray


class Matcher:
    def __init__(self, cache_dir: Path, *, setting: str = "precise", max_matches: int = 5000, seed: int = 0):
        self.cache_dir = Path(cache_dir)
        self.setting = setting
        self.max_matches = max_matches
        self.seed = seed
        self._model = None
        self.stats = {"requests": 0, "cache_hits": 0, "cache_misses": 0}

    def describe_device(self) -> str:
        import torch

        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] // 2**20
            return f"cuda:{torch.cuda.get_device_name(0)}, {free} MiB free"
        return "cpu (RoMa v2 is slow without a GPU)"

    def match_many(self, requests: list[MatchRequest]) -> list[Matches]:
        results = []
        for request in requests:
            self.stats["requests"] += 1
            path = self.cache_dir / f"{self._key(request)}.npz"
            if path.is_file():
                self.stats["cache_hits"] += 1
                with np.load(path) as data:
                    results.append(Matches(data["points0"], data["points1"], data["confidence"]))
                continue
            self.stats["cache_misses"] += 1
            matches = self._match(request)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path, points0=matches.points0, points1=matches.points1, confidence=matches.confidence
            )
            results.append(matches)
        return results

    def _key(self, request: MatchRequest) -> str:
        payload = {
            "schema": CACHE_SCHEMA,
            "setting": self.setting,
            "max_matches": self.max_matches,
            "seed": self.seed,
            "image0": _sha256(request.image0),
            "image1": _sha256(request.image1),
            "rotation_deg": request.rotation_deg % 360,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _match(self, request: MatchRequest) -> Matches:
        import cv2
        import tempfile

        model = self._load()
        _seed(self.seed)
        image1 = cv2.imread(str(request.image1), cv2.IMREAD_UNCHANGED)
        if image1 is None:
            raise RuntimeError(f"Could not read matcher input {request.image1}")
        height1, width1 = image1.shape[:2]
        image0 = cv2.imread(str(request.image0), cv2.IMREAD_UNCHANGED)
        if image0 is None:
            raise RuntimeError(f"Could not read matcher input {request.image0}")
        height0, width0 = image0.shape[:2]

        forward = None
        with tempfile.TemporaryDirectory(prefix="luchtfotos-match-") as directory:
            source1 = request.image1
            if request.rotation_deg % 360:
                rotated, forward = _rotate_bound(image1, request.rotation_deg)
                source1 = Path(directory) / "image1.png"
                cv2.imwrite(str(source1), rotated)
                height1, width1 = rotated.shape[:2]
            predictions = model.match(str(request.image0), str(source1))
            matches, overlaps, _, _ = model.sample(predictions, self.max_matches)
            points0, points1 = model.to_pixel_coordinates(matches, height0, width0, height1, width1)
        points0, points1 = _numpy(points0), _numpy(points1)
        confidence = _numpy(overlaps)
        if forward is not None:
            points1 = cv2.transform(points1[None].astype(np.float32), cv2.invertAffineTransform(forward))[0]
            height, width = image1.shape[:2]
            inside = (points1[:, 0] >= 0) & (points1[:, 0] < width) & (points1[:, 1] >= 0) & (points1[:, 1] < height)
            points0, points1, confidence = points0[inside], points1[inside], confidence[inside]
        return Matches(points0, points1, confidence)

    def _load(self):
        if self._model is None:
            try:
                from romav2 import RoMaV2
            except ImportError as exc:
                raise RuntimeError("Matching needs RoMa v2: pip install romav2") from exc
            # Eager rather than compiled: compiling the precise preset can
            # exhaust a 16 GB GPU even though eager inference fits.
            self._model = RoMaV2(RoMaV2.Cfg(compile=False, setting=self.setting))
        return self._model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _seed(seed: int) -> None:
    """Seed every random source RoMa v2's sampling draws from, once per pair."""
    import cv2
    import torch

    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _rotate_bound(image, angle: float):
    """Rotate without cropping; returns the image and the 2x3 forward matrix."""
    import cv2

    height, width = image.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_width = int(round(height * sine + width * cosine))
    out_height = int(round(height * cosine + width * sine))
    matrix[0, 2] += out_width / 2.0 - centre[0]
    matrix[1, 2] += out_height / 2.0 - centre[1]
    return cv2.warpAffine(image, matrix, (out_width, out_height), borderValue=0), matrix
