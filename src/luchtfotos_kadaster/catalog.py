"""Settings, sheet-index files and the frame catalogue for the Kadaster sites."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from luchtfotos_kadaster.iiif import IIIFImage


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SiteSettings(_Model):
    # Sheet-index text file shipped with the photographs; its "RD:" line is the
    # sheet centre that bounds the coarse search.
    sheet: Path
    years: list[int] | None = None


class ReferenceSettings(_Model):
    wms_url: str = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
    layer: str = "Actueel_ortho25"
    # Older epochs are matched against the georeferenced photographs of the
    # nearest newer epoch, with PDOK filling whatever they do not cover. Off
    # means every epoch is matched against PDOK directly.
    chain_epochs: bool = True


class MatchingSettings(_Model):
    # RoMa v2 preset: "precise" adds a 1280 px refinement and matches both ways.
    romav2_setting: str = "precise"
    max_matches: int = 5000
    # Seeded before every pair, so a rerun reproduces the cached results.
    seed: int = 0


class CoarseSettings(_Model):
    overview_px: int = 1600
    search_half_width_m: float = 4000.0
    resolution_m: float = 4.0
    # Tried in order; later angles only for frames the earlier ones did not place.
    rotations_deg: list[float] = Field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    # A frame the centred window misses is retried on windows shifted by this
    # much in x and y (3 x 3 pattern) before the other rotations are tried.
    window_shift_m: float = 3000.0
    # Frames still unplaced are matched photo-to-photo against this many
    # anchors: frames of the same epoch placed coarsely, then newer epochs.
    max_photo_anchors: int = 8
    min_inliers: int = 300
    min_inlier_fraction: float = 0.1
    # Plausible native scan GSD; rejects homographies that map the frame onto a
    # wildly wrong scale.
    gsd_range_m: tuple[float, float] = (0.1, 2.0)


class RefineSettings(_Model):
    grid: int = 4
    resolution_m: float = 1.0
    segment_margin: float = 0.15
    inlier_threshold_m: float = 4.0
    max_rmse_m: float = 6.0
    min_gcps: int = 20


class OutputSettings(_Model):
    max_gcps_per_image: int = 40
    max_rcps_per_pair: int = 30
    transformation: str = "projective"
    annotation_base: str = "https://fdahle.github.io/luchtfotos-kadaster/maps"
    # Appended to every annotation id, e.g. ".json" so a static host serves
    # the per-map file under its id with a JSON content type.
    annotation_suffix: str = ""
    min_pair_overlap: float = 0.05


class KadasterSettings(_Model):
    manifest_base: str
    image_index: Path | None = None
    output_root: Path
    # The published static site (GitHub Pages serves it); `publish` writes here.
    site_dir: Path = Path("../docs")
    sites: dict[str, SiteSettings]
    reference: ReferenceSettings = ReferenceSettings()
    matching: MatchingSettings = MatchingSettings()
    coarse: CoarseSettings = CoarseSettings()
    refine: RefineSettings = RefineSettings()
    output: OutputSettings = OutputSettings()


def load_settings(path: Path) -> KadasterSettings:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    settings = KadasterSettings.model_validate(raw)
    base = path.parent
    if not settings.site_dir.is_absolute():
        settings.site_dir = (base / settings.site_dir).resolve()
    if settings.image_index is not None and not settings.image_index.is_absolute():
        settings.image_index = base / settings.image_index
    for site in settings.sites.values():
        if not site.sheet.is_absolute():
            site.sheet = base / site.sheet
    return settings


@dataclass(frozen=True)
class SheetIndex:
    sheets: str
    centre_rd: tuple[float, float]
    epochs: dict[int, tuple[str, ...]]
    rotated_years: frozenset[int]


_EPOCH = re.compile(r"^\s*(\d{4})\s+(\d+)x\s+(.*)$")


def parse_sheet_index(text: str) -> SheetIndex:
    sheets, centre, epochs, rotated = "", None, {}, set()
    for line in text.splitlines():
        if line.startswith("Kaartbladnummers:"):
            sheets = line.split(":", 1)[1].strip()
        elif line.startswith("RD:"):
            x, y = line.split(":", 1)[1].split()[:2]
            centre = (float(x), float(y))
        elif match := _EPOCH.match(line):
            year, rest = int(match.group(1)), match.group(3)
            if "GEROTEERD" in rest.upper():
                rotated.add(year)
            names = tuple(token for token in rest.split() if not token.startswith("("))
            epochs[year] = names
    if centre is None:
        raise ValueError("Sheet index has no 'RD:' line")
    return SheetIndex(sheets, centre, epochs, frozenset(rotated))


@dataclass(frozen=True)
class Frame:
    site: str
    year: int
    name: str
    image: IIIFImage

    @property
    def key(self) -> str:
        # DLCS filenames already lead with the year (2002_37111).
        if self.name.startswith(f"{self.year}_"):
            return f"{self.site}_{self.name}"
        return f"{self.site}_{self.year}_{self.name}"


def read_image_index(path: Path | None) -> dict[str, str]:
    """DLCS export: image-service id -> original file stem (e.g. 2002_37111)."""
    if path is None or not Path(path).is_file():
        return {}
    with open(path, newline="") as handle:
        return {
            row["ID"]: Path(row["Origin"]).stem
            for row in csv.DictReader(handle)
            if row.get("ID") and row.get("Origin")
        }


def site_frames(site: str, images: list[IIIFImage], index: dict[str, str]) -> list[Frame]:
    """Group a site manifest's images into frames, using the DLCS metadata for the year.

    The generated manifest labels canvases "Canvas n", so the frame name comes
    from the image index (the original filename) and falls back to the service id.
    """
    frames = []
    for image in images:
        year = image.meta("String 3")
        name = index.get(image.image_id, image.image_id)
        if year is None or not year.strip().isdigit():
            match = re.match(r"(\d{4})_", name)
            if match is None:
                raise ValueError(f"Cannot tell the year of {image.service}")
            year = match.group(1)
        frames.append(Frame(site, int(year), name, image))
    return frames
