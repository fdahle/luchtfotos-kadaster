"""Place Kadaster frames on RD New from IIIF overviews and regions.

A frame's georeference is one projective transform from IIIF resource pixels
to RD metres. The terrain is flat to a few metres, so for a vertical frame
the transform is close to exact. It is found in two passes. The coarse pass
matches a reduced overview against a reference rendered around the sheet
centre. The refinement pass splits the placed frame into a grid of IIIF
regions read near the refinement resolution, matches each against a reference
rendered for the same RD box, and refits on every inlier. The reference is
PDOK, or the already-placed frames of newer epochs over PDOK when epochs are
chained. A reference pixel taken from a newer frame is a link between the two
frames' resources, so it becomes a Resource Control Point pair as well as a GCP.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from luchtfotos_kadaster.catalog import Frame, KadasterSettings
from luchtfotos_kadaster.iiif import IIIFClient, overview_width_for
from luchtfotos_kadaster.reference import PdokReference, RasterGrid
from luchtfotos_kadaster.matching import Matcher, MatchRequest

PDOK = -1


@dataclass
class Placement:
    frame: Frame
    status: str
    reason: str = ""
    # resource px -> RD metres
    homography: list[list[float]] | None = None
    mask: list[list[float]] = field(default_factory=list)
    rotation_deg: float = 0.0
    # [resource_x, resource_y, rd_x, rd_y, source] where source is "pdok" or a frame key
    gcps: list[list] = field(default_factory=list)
    # [resource_x, resource_y, other_key, other_x, other_y] cross-epoch links
    links: list[list] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_json(self) -> dict:
        return {
            "key": self.frame.key,
            "image_service": self.frame.image.service,
            "status": self.status,
            "reason": self.reason,
            "homography": self.homography,
            "mask": self.mask,
            "rotation_deg": self.rotation_deg,
            "gcps": self.gcps,
            "links": self.links,
            "stats": self.stats,
        }

    @classmethod
    def from_json(cls, frame: Frame, data: dict) -> "Placement":
        return cls(
            frame,
            data["status"],
            data.get("reason", ""),
            data.get("homography"),
            data.get("mask", []),
            data.get("rotation_deg", 0.0),
            data.get("gcps", []),
            data.get("links", []),
            data.get("stats", {}),
        )


# -- small geometry helpers -------------------------------------------------


def scale_matrix(sx: float, sy: float, x0: float = 0.0, y0: float = 0.0):
    """Full-resolution pixel -> reduced pixel of a region starting at (x0, y0), centre convention."""
    import numpy as np

    return np.array(
        [[sx, 0.0, (0.5 - x0) * sx - 0.5], [0.0, sy, (0.5 - y0) * sy - 0.5], [0.0, 0.0, 1.0]]
    )


def transform(matrix, points):
    import numpy as np

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(points):
        return points
    stacked = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    return stacked[:, :2] / stacked[:, 2:3]


def local_gsd(matrix, point) -> float:
    """Ground metres per resource pixel of ``matrix`` at ``point``."""
    import numpy as np

    h = np.asarray(matrix)
    x, y = point
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    mapped = transform(h, [point])[0]
    jacobian = (h[:2, :2] - np.outer(mapped, h[2, :2])) / w
    return float(math.sqrt(abs(np.linalg.det(jacobian))))


def fit_homography(source, target, threshold: float):
    import cv2
    import numpy as np

    source = np.asarray(source, np.float64).reshape(-1, 2)
    target = np.asarray(target, np.float64).reshape(-1, 2)
    if len(source) < 8:
        return None, np.zeros(len(source), bool)
    # Fit in a centred frame: RD coordinates in the 1e5 range are badly
    # conditioned for MAGSAC's normalised solver otherwise. USAC also rejects
    # orientation-reversing models, and image rows down against northing up is
    # one, so a reflected target is fitted with y negated and flipped back.
    s_off, t_off = source.mean(0), target.mean(0)
    flip = np.diag([1.0, -1.0, 1.0]) if _reflected(source, target) else np.eye(3)
    matrix, mask = cv2.findHomography(
        (source - s_off).astype(np.float32),
        ((target - t_off) * np.diag(flip)[:2]).astype(np.float32),
        cv2.USAC_MAGSAC,
        threshold,
        maxIters=20000,
        confidence=0.9999,
    )
    if matrix is None:
        return None, np.zeros(len(source), bool)
    to_centre = np.array([[1, 0, -s_off[0]], [0, 1, -s_off[1]], [0, 0, 1.0]])
    from_centre = np.array([[1, 0, t_off[0]], [0, 1, t_off[1]], [0, 0, 1.0]])
    matrix = from_centre @ flip @ matrix @ to_centre
    return matrix / matrix[2, 2], mask.ravel().astype(bool)


def _reflected(source, target) -> bool:
    """Whether the best affine map between the point sets reverses orientation."""
    import numpy as np

    design = np.c_[source - source.mean(0), np.ones(len(source))]
    affine, *_ = np.linalg.lstsq(design, target - target.mean(0), rcond=None)
    return bool(np.linalg.det(affine[:2].T) < 0)


def balanced_subset(points, count: int, grid: int = 8):
    """Indices of at most ``count`` points spread over a ``grid`` x ``grid`` partition."""
    import numpy as np

    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) <= count:
        return np.arange(len(points))
    lo, hi = points.min(0), points.max(0)
    cells = np.floor((points - lo) / np.maximum(hi - lo, 1e-9) * (grid - 1e-9)).astype(int)
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(map(tuple, cells)):
        buckets.setdefault(cell, []).append(index)
    rng = np.random.default_rng(0)
    for members in buckets.values():
        rng.shuffle(members)
    chosen, depth = [], 0
    while len(chosen) < count:
        layer = [members[depth] for members in buckets.values() if depth < len(members)]
        if not layer:
            break
        chosen.extend(layer[: count - len(chosen)])
        depth += 1
    return np.asarray(sorted(chosen))


def footprint(placement: Placement):
    from shapely.geometry import Polygon

    return Polygon(transform(placement.homography, placement.mask))


# -- rendering --------------------------------------------------------------


def render_placement(client: IIIFClient, placement: Placement, grid: RasterGrid):
    """Warp a placed frame onto ``grid`` from one IIIF region; (image, valid) or None."""
    import cv2
    import numpy as np

    h = np.asarray(placement.homography)
    inverse = np.linalg.inv(h)
    xmin, ymin, xmax, ymax = grid.bounds
    corners = transform(inverse, [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)])
    mask = np.asarray(placement.mask)
    lo = np.maximum(corners.min(0), mask.min(0)) - 2
    hi = np.minimum(corners.max(0), mask.max(0)) + 2
    if np.any(hi - lo < 4):
        return None
    image = placement.frame.image
    x0, y0 = max(0, int(lo[0])), max(0, int(lo[1]))
    x1, y1 = min(image.width, int(math.ceil(hi[0]))), min(image.height, int(math.ceil(hi[1])))
    centre = ((x0 + x1) / 2, (y0 + y1) / 2)
    factor = min(1.0, local_gsd(h, centre) / grid.resolution)
    region, (sx, sy) = client.region(image, x0, y0, x1 - x0, y1 - y0, math.ceil((x1 - x0) * factor))
    grid_to_region = scale_matrix(sx, sy, x0, y0) @ inverse @ grid.pixel_to_world()
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    size = (grid.width, grid.height)
    warped = cv2.warpPerspective(region, grid_to_region, size, flags=flags)
    region_mask = np.zeros(region.shape, np.uint8)
    polygon = transform(scale_matrix(sx, sy, x0, y0), mask)
    cv2.fillPoly(region_mask, [np.round(polygon).astype(np.int32)], 255)
    valid = cv2.warpPerspective(region_mask, grid_to_region, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0
    return warped, valid


def render_reference(client: IIIFClient, pdok: PdokReference, grid: RasterGrid, placed: list[Placement]):
    """PDOK under the given placements (first one wins); returns image and owner index (-1 = PDOK)."""
    import numpy as np
    from shapely.geometry import box

    image = pdok.render(grid)
    owner = np.full(image.shape, PDOK, np.int32)
    area = box(*grid.bounds)
    for index, placement in enumerate(placed):
        if not footprint(placement).intersects(area):
            continue
        rendered = render_placement(client, placement, grid)
        if rendered is None:
            continue
        warped, valid = rendered
        take = valid & (owner == PDOK)
        image[take] = warped[take]
        owner[take] = index
    return image, owner


# -- matching ---------------------------------------------------------------


def _requests(settings: KadasterSettings, pairs, rotation: float = 0.0):
    return [MatchRequest(image0=reference, image1=photo, rotation_deg=float(rotation)) for reference, photo in pairs]


def _write(path: Path, array) -> Path:
    import cv2

    # Always rewritten: the matcher cache keys on content, so an unchanged
    # render still hits it and a changed one cannot be served stale.
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), array)
    return path


# -- coarse ----------------------------------------------------------------


def coarse_placements(
    frames: list[Frame],
    centre: tuple[float, float],
    placed_newer: list[Placement],
    *,
    settings: KadasterSettings,
    iiif: IIIFClient,
    pdok: PdokReference,
    matcher: Matcher,
    work: Path,
) -> list[Placement]:
    import numpy as np

    c = settings.coarse
    overviews = {}
    for frame in frames:
        array, scale = iiif.overview(frame.image, overview_width_for(frame.image, c.overview_px))
        overviews[frame.key] = (_write(work / "coarse" / f"{frame.key}.png", array), scale, textured(array))

    # Unrotated at the sheet centre first, then shifted windows (a frame can
    # reach past the edge of the centred one), then the other rotations.
    step = c.window_shift_m
    shifts = [(dx, dy) for dy in (0.0, -step, step) for dx in (0.0, -step, step)]
    attempts = [(shift, c.rotations_deg[0]) for shift in shifts]
    attempts += [((0.0, 0.0), rotation) for rotation in c.rotations_deg[1:]]
    references: dict[tuple[float, float], tuple[Path, RasterGrid]] = {}
    results: dict[str, Placement] = {}
    pending = list(frames)
    for (dx, dy), rotation in attempts:
        if not pending:
            break
        if (dx, dy) not in references:
            grid = RasterGrid.around(centre[0] + dx, centre[1] + dy, c.search_half_width_m, c.resolution_m)
            reference, _ = render_reference(iiif, pdok, grid, placed_newer)
            name = f"reference_{len(placed_newer)}_{dx:+.0f}_{dy:+.0f}.png"
            references[(dx, dy)] = (_write(work / "coarse" / name, reference), grid)
        reference_path, grid = references[(dx, dy)]
        matches = matcher.match_many(
            _requests(settings, [(reference_path, overviews[f.key][0]) for f in pending], rotation)
        )
        still = []
        for frame, match in zip(pending, matches):
            path, (sx, sy), texture = overviews[frame.key]
            to_overview = scale_matrix(sx, sy)
            resource = transform(np.linalg.inv(to_overview), match.points1)
            world = transform(grid.pixel_to_world(), match.points0)
            matrix, inliers = fit_homography(resource, world, 2.5 * c.resolution_m)
            count = int(inliers.sum())
            fraction = count / max(1, len(resource))
            placement = Placement(frame, "failed", rotation_deg=rotation)
            placement.stats = {
                "coarse_matches": len(resource),
                "coarse_inliers": count,
                "coarse_window_shift_m": [dx, dy],
            }
            if matrix is None or count < c.min_inliers or fraction < c.min_inlier_fraction:
                placement.reason = (
                    f"coarse: best attempt {count} inliers ({fraction:.0%}) "
                    f"at {rotation:g} deg, window shift ({dx:+.0f}, {dy:+.0f}) m"
                )
            else:
                gsd = local_gsd(matrix, (frame.image.width / 2, frame.image.height / 2))
                placement.stats["coarse_gsd_m"] = round(gsd, 4)
                if not c.gsd_range_m[0] <= gsd <= c.gsd_range_m[1]:
                    placement.reason = f"coarse: implausible scale {gsd:.3f} m/px"
                else:
                    placement.status, placement.reason = "coarse", ""
                    placement.homography = matrix.tolist()
                    placement.mask = photo_mask(texture, to_overview, resource[inliers])
            previous = results.get(frame.key)
            if placement.status == "failed":
                still.append(frame)
                # Report the strongest failed attempt, not the first one.
                if previous is None or count > previous.stats["coarse_inliers"]:
                    results[frame.key] = placement
            else:
                results[frame.key] = placement
        pending = still
    if pending:
        anchors = [results[f.key] for f in frames if results[f.key].status == "coarse"] + placed_newer
        _photo_to_photo(pending, anchors[: c.max_photo_anchors], results, overviews, settings=settings, iiif=iiif, matcher=matcher, work=work)
    return [results[frame.key] for frame in frames]


def _photo_to_photo(pending, anchors, results, overviews, *, settings, iiif, matcher, work) -> None:
    """Place frames the rendered reference could not by matching other frames' overviews.

    A composite of several placed frames over PDOK, resampled to the coarse
    grid, can be much harder to match than one of the photographs it is built
    from. Each still-unplaced overview is matched against the overview of each
    anchor (a frame of this epoch already placed coarsely, or a placed frame
    of a newer epoch). The strongest pair places it through the anchor's
    transform.
    """
    import numpy as np

    c = settings.coarse
    if not anchors:
        return
    anchor_overviews = {}
    for anchor in anchors:
        image = anchor.frame.image
        array, scale = iiif.overview(image, overview_width_for(image, c.overview_px))
        anchor_overviews[anchor.frame.key] = (_write(work / "coarse" / "anchors" / f"{anchor.frame.key}.png", array), scale)
    pairs = [(frame, anchor) for frame in pending for anchor in anchors]
    matches = matcher.match_many(
        _requests(settings, [(anchor_overviews[a.frame.key][0], overviews[f.key][0]) for f, a in pairs])
    )
    best: dict[str, tuple] = {}
    for (frame, anchor), match in zip(pairs, matches):
        _, (sx, sy), _ = overviews[frame.key]
        _, (ax, ay) = anchor_overviews[anchor.frame.key]
        resource = transform(np.linalg.inv(scale_matrix(sx, sy)), match.points1)
        anchor_resource = transform(np.linalg.inv(scale_matrix(ax, ay)), match.points0)
        # Threshold in anchor resource pixels, about 2.5 coarse cells on the ground.
        threshold = 2.5 * c.resolution_m / max(local_gsd(anchor.homography, anchor_resource.mean(0)), 1e-6)
        matrix, inliers = fit_homography(resource, anchor_resource, threshold)
        count = int(inliers.sum())
        if matrix is not None and count > best.get(frame.key, (0,))[0]:
            best[frame.key] = (count, count / max(1, len(resource)), anchor, matrix, resource[inliers])
    for frame in pending:
        if frame.key not in best:
            continue
        count, fraction, anchor, matrix, inlier_resource = best[frame.key]
        if count < c.min_inliers or fraction < c.min_inlier_fraction:
            continue
        homography = np.asarray(anchor.homography) @ matrix
        gsd = local_gsd(homography, (frame.image.width / 2, frame.image.height / 2))
        if not c.gsd_range_m[0] <= gsd <= c.gsd_range_m[1]:
            continue
        _, (sx, sy), texture = overviews[frame.key]
        placement = Placement(frame, "coarse")
        placement.homography = (homography / homography[2, 2]).tolist()
        placement.mask = photo_mask(texture, scale_matrix(sx, sy), inlier_resource)
        placement.stats = {
            "coarse_matches": results[frame.key].stats.get("coarse_matches"),
            "coarse_inliers": count,
            "coarse_anchor": anchor.frame.key,
            "coarse_gsd_m": round(gsd, 4),
        }
        results[frame.key] = placement


def textured(image, window: int = 21, min_std: float = 8.0):
    """Pixels on photo content: not saturated margin, not flat film base."""
    import cv2
    import numpy as np

    image = image.astype(np.float32)
    mean = cv2.blur(image, (window, window))
    std = np.sqrt(np.maximum(cv2.blur(image * image, (window, window)) - mean * mean, 0))
    keep = (std > min_std) & (mean > 20) & (mean < 235)
    kernel = np.ones((window, window), np.uint8)
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    # Annotation text on the data strip is textured too, but a black band
    # separates it from the photograph: keep the largest component only.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(keep, connectivity=4)
    if count <= 1:
        return keep > 0
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def photo_mask(texture, to_overview, inlier_resource) -> list[list[float]]:
    """Resource-pixel polygon of the photograph.

    The hull of the textured component covers the whole exposed frame without
    the margins and data strip. The hull of the matches would stop where the
    reference has nothing matchable. Dense matches also interpolate through
    the black margins and land on the homography there, so they cannot bound
    the photograph either. They are the fallback only when the texture
    component is implausibly small.
    """
    import numpy as np

    rows, cols = np.nonzero(texture)
    if len(rows) < 0.3 * texture.size:
        return _hull(inlier_resource)
    edge = np.c_[cols, rows].astype(np.float64)
    return _hull(transform(np.linalg.inv(to_overview), edge))


def inside(polygon, points):
    import numpy as np
    import shapely
    from shapely.geometry import Polygon

    points = np.asarray(points, np.float64).reshape(-1, 2)
    return shapely.contains_xy(Polygon(polygon), points[:, 0], points[:, 1])


def _hull(points) -> list[list[float]]:
    import cv2
    import numpy as np

    hull = cv2.convexHull(np.asarray(points, np.float32))
    hull = cv2.approxPolyDP(hull, 0.003 * cv2.arcLength(hull, True), True)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in hull.reshape(-1, 2)]


# -- refinement ------------------------------------------------------------


def refine_placements(
    placements: list[Placement],
    placed_newer: list[Placement],
    *,
    settings: KadasterSettings,
    iiif: IIIFClient,
    pdok: PdokReference,
    matcher: Matcher,
    work: Path,
) -> None:
    import numpy as np

    r = settings.refine
    jobs = []  # (placement, cell id, reference path, owner, grid, photo path, region origin+scale)
    for placement in placements:
        if placement.status != "coarse":
            continue
        h = np.asarray(placement.homography)
        mask = np.asarray(placement.mask)
        lo, hi = mask.min(0), mask.max(0)
        step = (hi - lo) / r.grid
        for row in range(r.grid):
            for col in range(r.grid):
                cell_lo = lo + step * (col, row) - step * r.segment_margin
                cell_hi = lo + step * (col + 1, row + 1) + step * r.segment_margin
                image = placement.frame.image
                x0, y0 = max(0, int(cell_lo[0])), max(0, int(cell_lo[1]))
                x1 = min(image.width, int(math.ceil(cell_hi[0])))
                y1 = min(image.height, int(math.ceil(cell_hi[1])))
                corners = transform(h, [(x0, y0), (x1, y0), (x0, y1), (x1, y1)])
                grid = RasterGrid.covering(
                    (*corners.min(0), *corners.max(0)), r.resolution_m, margin=4 * r.inlier_threshold_m
                )
                reference, owner = render_reference(iiif, pdok, grid, placed_newer)
                factor = min(1.0, local_gsd(h, ((x0 + x1) / 2, (y0 + y1) / 2)) / r.resolution_m)
                photo, scale = iiif.region(image, x0, y0, x1 - x0, y1 - y0, math.ceil((x1 - x0) * factor))
                stem = work / "refine" / placement.frame.key / f"r{row}c{col}"
                jobs.append(
                    (
                        placement,
                        _write(stem.with_name(stem.name + "_ref.png"), reference),
                        owner,
                        grid,
                        _write(stem.with_name(stem.name + "_img.png"), photo),
                        (x0, y0, scale),
                    )
                )
    if not jobs:
        return
    matches = matcher.match_many(_requests(settings, [(job[1], job[4]) for job in jobs]))

    collected: dict[str, list] = {}
    for (placement, _, owner, grid, _, (x0, y0, (sx, sy))), match in zip(jobs, matches):
        resource = transform(np.linalg.inv(scale_matrix(sx, sy, x0, y0)), match.points1)
        points0 = np.asarray(match.points0).reshape(-1, 2)
        within = inside(placement.mask, resource)
        resource, points0 = resource[within], points0[within]
        world = transform(grid.pixel_to_world(), points0)
        _, inliers = fit_homography(resource, world, r.inlier_threshold_m)
        cols = np.clip(np.round(points0[:, 0]).astype(int), 0, grid.width - 1)
        rows = np.clip(np.round(points0[:, 1]).astype(int), 0, grid.height - 1)
        sources = owner[rows, cols]
        bucket = collected.setdefault(placement.frame.key, [placement, [], [], []])
        bucket[1].append(resource[inliers])
        bucket[2].append(world[inliers])
        bucket[3].append(sources[inliers])

    out = settings.output
    for placement, resources, worlds, sources in collected.values():
        resource, world, source = np.vstack(resources), np.vstack(worlds), np.concatenate(sources)
        matrix, inliers = fit_homography(resource, world, r.inlier_threshold_m)
        if matrix is None:
            placement.status, placement.reason = "failed", "refine: no consistent transform"
            continue
        residual = np.linalg.norm(transform(matrix, resource[inliers]) - world[inliers], axis=1)
        rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else math.inf
        placement.stats.update(
            refine_matches=int(len(resource)),
            refine_inliers=int(inliers.sum()),
            rmse_m=round(rmse, 2),
            median_residual_m=round(float(np.median(residual)), 2) if len(residual) else None,
            gsd_m=round(local_gsd(matrix, (placement.frame.image.width / 2, placement.frame.image.height / 2)), 4),
            control_from_newer_epochs=round(float((source[inliers] != PDOK).mean()), 3) if inliers.any() else 0.0,
        )
        if inliers.sum() < r.min_gcps or rmse > r.max_rmse_m:
            placement.status = "failed"
            placement.reason = f"refine: {int(inliers.sum())} inliers, RMSE {rmse:.1f} m"
            continue
        placement.status = "ok"
        placement.homography = (matrix / matrix[2, 2]).tolist()
        keep = np.flatnonzero(inliers)[balanced_subset(resource[inliers], out.max_gcps_per_image)]
        placement.gcps, placement.links = [], []
        for index in keep:
            rx, ry = resource[index]
            wx, wy = world[index]
            owner_index = int(source[index])
            label = "pdok" if owner_index == PDOK else placed_newer[owner_index].frame.key
            placement.gcps.append([round(float(rx), 2), round(float(ry), 2), round(float(wx), 3), round(float(wy), 3), label])
            if owner_index != PDOK:
                other = placed_newer[owner_index]
                ox, oy = transform(np.linalg.inv(other.homography), [(wx, wy)])[0]
                placement.links.append([round(float(rx), 2), round(float(ry), 2), other.frame.key, round(float(ox), 2), round(float(oy), 2)])


# -- tie points within an epoch -----------------------------------------------


def epoch_ties(
    placements: list[Placement],
    *,
    settings: KadasterSettings,
    iiif: IIIFClient,
    matcher: Matcher,
    work: Path,
    max_side_px: int = 3000,
) -> list[dict]:
    """Resource-to-resource tie points for every overlapping pair of placed frames."""
    import numpy as np

    placed = [p for p in placements if p.ok]
    jobs = []
    for i, first in enumerate(placed):
        for second in placed[i + 1 :]:
            a, b = footprint(first), footprint(second)
            overlap = a.intersection(b)
            if overlap.is_empty or overlap.area / min(a.area, b.area) < settings.output.min_pair_overlap:
                continue
            xmin, ymin, xmax, ymax = overlap.bounds
            resolution = max(settings.refine.resolution_m, max(xmax - xmin, ymax - ymin) / max_side_px)
            grid = RasterGrid.covering(overlap.bounds, resolution)
            one, two = render_placement(iiif, first, grid), render_placement(iiif, second, grid)
            if one is None or two is None:
                continue
            stem = work / "ties" / f"{first.frame.key}__{second.frame.key}"
            jobs.append(
                (
                    first,
                    second,
                    grid,
                    one[1],
                    two[1],
                    _write(stem.with_name(stem.name + "_a.png"), one[0]),
                    _write(stem.with_name(stem.name + "_b.png"), two[0]),
                    overlap.area / min(a.area, b.area),
                )
            )
    if not jobs:
        return []
    matches = matcher.match_many(_requests(settings, [(job[5], job[6]) for job in jobs]))
    ties = []
    for (first, second, grid, valid_a, valid_b, _, _, overlap), match in zip(jobs, matches):
        pa, pb = np.asarray(match.points0).reshape(-1, 2), np.asarray(match.points1).reshape(-1, 2)
        inside = np.ones(len(pa), bool)
        for points, valid in ((pa, valid_a), (pb, valid_b)):
            cols = np.clip(np.round(points[:, 0]).astype(int), 0, grid.width - 1)
            rows = np.clip(np.round(points[:, 1]).astype(int), 0, grid.height - 1)
            inside &= valid[rows, cols]
        pa, pb = pa[inside], pb[inside]
        # Both renders already sit on RD, so true ties nearly coincide; the
        # homography only absorbs the residual georeference error and relief.
        _, inliers = fit_homography(pa, pb, settings.refine.inlier_threshold_m / grid.resolution)
        pa, pb = pa[inliers], pb[inliers]
        keep = balanced_subset(pa, settings.output.max_rcps_per_pair)
        world_a = transform(grid.pixel_to_world(), pa[keep])
        world_b = transform(grid.pixel_to_world(), pb[keep])
        resource_a = transform(np.linalg.inv(first.homography), world_a)
        resource_b = transform(np.linalg.inv(second.homography), world_b)
        ties.append(
            {
                "a": first.frame.key,
                "b": second.frame.key,
                "overlap": round(float(overlap), 3),
                "inliers": int(inliers.sum()),
                "points": [
                    [round(float(ax), 2), round(float(ay), 2), round(float(bx), 2), round(float(by), 2)]
                    for (ax, ay), (bx, by) in zip(resource_a, resource_b)
                ],
            }
        )
    return ties


def save_placement(path: Path, placement: Placement) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(placement.to_json(), indent=1))
