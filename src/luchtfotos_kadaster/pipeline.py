"""Run one Kadaster site: newest epoch against PDOK, then backwards in time.

Output layout under ``output_root/<site>/``::

    <year>/frames/<key>.json   placement, GCPs and cross-epoch links (resume state)
    <year>/annotations.json    Georeference AnnotationPage for the epoch
    <year>/rcps.json           RCPs: ties inside the epoch + links to newer epochs
    <year>/mosaic.png/.pgw     quick-look of the placed frames on RD New
    annotations.json           every placed frame of the site
    rcps.json                  every RCP of the site
    report.json                per-frame status, residuals, IIIF pixel ledger
"""

from __future__ import annotations

import json
from pathlib import Path

from luchtfotos_kadaster.annotations import annotation_page, georeference_annotation, resource_control_points
from luchtfotos_kadaster.catalog import KadasterSettings, parse_sheet_index, read_image_index, site_frames
from luchtfotos_kadaster.georef import (
    Placement,
    coarse_placements,
    epoch_ties,
    footprint,
    refine_placements,
    render_reference,
    save_placement,
)
from luchtfotos_kadaster.iiif import IIIFClient
from luchtfotos_kadaster.reference import PdokReference, RasterGrid
from luchtfotos_kadaster.matching import Matcher


def _log(message: str) -> None:
    print(message, flush=True)


def run_site(
    settings: KadasterSettings,
    site: str,
    *,
    years: list[int] | None = None,
    force: bool = False,
) -> dict:
    if site not in settings.sites:
        raise ValueError(f"Unknown site {site!r}; settings define: {', '.join(settings.sites)}")
    site_settings = settings.sites[site]
    sheet = parse_sheet_index(site_settings.sheet.read_text())
    root = settings.output_root / site
    cache = settings.output_root / "_cache"
    iiif = IIIFClient(cache / "iiif")
    pdok = PdokReference(settings.reference.wms_url, settings.reference.layer, cache / "pdok")
    m = settings.matching
    matcher = Matcher(cache / "romav2", setting=m.romav2_setting, max_matches=m.max_matches, seed=m.seed)
    _log(f"[{site}] matcher on {matcher.describe_device()}")

    manifest = f"{settings.manifest_base.rstrip('/')}/{site}"
    frames = site_frames(site, iiif.manifest(manifest, refresh=force), read_image_index(settings.image_index))
    available = sorted({frame.year for frame in frames}, reverse=True)
    wanted = years or site_settings.years or available
    order = [year for year in available if year in set(wanted)]
    _log(f"[{site}] {len(frames)} frames in {manifest}; epochs {order}")
    for year, names in sorted(sheet.epochs.items()):
        found = {f.name for f in frames if f.year == year}
        found |= {name.split("_", 1)[-1] for name in found}
        missing = [name for name in names if name not in found]
        if missing:
            _log(f"[{site}] warning: sheet index lists {year} frames not in the manifest: {missing}")

    placed: list[Placement] = []  # ok placements, nearest-newer epoch first
    everything: list[Placement] = []
    all_ties: list[dict] = []
    # Every epoch is walked so that a --year run still references the saved
    # placements of newer epochs and keeps the site-level outputs complete;
    # only the selected epochs are (re)processed.
    for year in available:
        selected = year in order
        epoch = [frame for frame in frames if frame.year == year]
        work = root / str(year)
        state = {frame.key: work / "frames" / f"{frame.key}.json" for frame in epoch}
        resumed = {}
        if not (force and selected):
            for frame in epoch:
                if state[frame.key].is_file():
                    saved = Placement.from_json(frame, json.loads(state[frame.key].read_text()))
                    # Failed frames are retried; the matcher cache makes the
                    # attempts that already failed cheap to repeat.
                    if saved.ok or not selected:
                        resumed[frame.key] = saved
        if not selected:
            kept = [resumed[frame.key] for frame in epoch if frame.key in resumed]
            everything.extend(kept)
            placed = [p for p in kept if p.ok] + placed
            if (work / "ties.json").is_file():
                all_ties.extend(json.loads((work / "ties.json").read_text()))
            continue
        todo = [frame for frame in epoch if frame.key not in resumed]
        reference = placed if settings.reference.chain_epochs else []
        source = "PDOK" if not reference else f"PDOK + {len(reference)} placed newer frames"
        _log(f"[{site} {year}] {len(epoch)} frames ({len(resumed)} resumed), reference: {source}")
        fresh = []
        if todo:
            fresh = coarse_placements(
                todo, sheet.centre_rd, reference, settings=settings, iiif=iiif, pdok=pdok, matcher=matcher, work=work
            )
            refine_placements(fresh, reference, settings=settings, iiif=iiif, pdok=pdok, matcher=matcher, work=work)
            for placement in fresh:
                save_placement(state[placement.frame.key], placement)
        by_key = {**resumed, **{p.frame.key: p for p in fresh}}
        epoch_placements = [by_key[frame.key] for frame in epoch]
        for placement in epoch_placements:
            stats = placement.stats
            detail = (
                f"RMSE {stats.get('rmse_m')} m, {stats.get('refine_inliers')} inliers, "
                f"{len(placement.gcps)} GCPs, {len(placement.links)} links"
                if placement.ok
                else placement.reason
            )
            _log(f"[{site} {year}] {placement.frame.name}: {placement.status} - {detail}")

        ties = epoch_ties(epoch_placements, settings=settings, iiif=iiif, matcher=matcher, work=work)
        all_ties.extend(ties)
        work.mkdir(parents=True, exist_ok=True)
        (work / "ties.json").write_text(json.dumps(ties))
        ok = [p for p in epoch_placements if p.ok]
        _write_epoch(work, ok, ties, placed, settings)
        if ok:
            _write_mosaic(work, ok, iiif, pdok)
        everything.extend(epoch_placements)
        placed = ok + placed

    out = settings.output
    ids = {"base": out.annotation_base, "suffix": out.annotation_suffix}
    ok_all = [p for p in everything if p.ok]
    (root / "annotations.json").write_text(
        json.dumps(
            annotation_page([georeference_annotation(p, transformation=out.transformation, **ids) for p in ok_all]),
            indent=1,
        )
    )
    rcps = resource_control_points(ok_all, all_ties, **ids)
    (root / "rcps.json").write_text(json.dumps(rcps, indent=1))
    report = {
        "site": site,
        "sheet": sheet.sheets,
        "centre_rd": sheet.centre_rd,
        "manifest": manifest,
        "frames": [
            {
                "key": p.frame.key,
                "year": p.frame.year,
                "name": p.frame.name,
                "image_service": p.frame.image.service,
                "canvas": p.frame.image.canvas,
                "status": p.status,
                "reason": p.reason,
                "gcps": len(p.gcps),
                "links": len(p.links),
                **p.stats,
            }
            for p in everything
        ],
        "ties": [{k: v for k, v in tie.items() if k != "points"} | {"rcps": len(tie["points"])} for tie in all_ties],
        "rcps": len(rcps),
        "iiif": iiif.ledger.summary(),
        "pdok_requests": pdok.requests,
        "matcher": matcher.stats,
    }
    (root / "report.json").write_text(json.dumps(report, indent=1))
    ledger = report["iiif"]
    _log(
        f"[{site}] placed {len(ok_all)}/{len(everything)} frames, {len(rcps)} RCPs; "
        f"IIIF delivered {ledger['delivered_fraction']:.1%} of the full-resolution pixels "
        f"({ledger['requests']} requests, {ledger['bytes_downloaded'] / 2**20:.1f} MiB)"
    )
    return report


def _write_epoch(work: Path, ok: list[Placement], ties: list[dict], newer: list[Placement], settings) -> None:
    out = settings.output
    ids = {"base": out.annotation_base, "suffix": out.annotation_suffix}
    work.mkdir(parents=True, exist_ok=True)
    page = annotation_page([georeference_annotation(p, transformation=out.transformation, **ids) for p in ok])
    (work / "annotations.json").write_text(json.dumps(page, indent=1))
    # Links point into newer epochs, so their frames are needed for map ids,
    # but only this epoch's ties and links are written here.
    rcps = resource_control_points(ok, ties, **ids)
    keys = {p.frame.key for p in ok}
    rcps += [
        rcp
        for rcp in resource_control_points(ok + newer, [], **ids)
        if rcp["id"].split(":")[1] in keys
    ]
    (work / "rcps.json").write_text(json.dumps(rcps, indent=1))


def _write_mosaic(work: Path, ok: list[Placement], iiif: IIIFClient, pdok: PdokReference, resolution: float = 4.0):
    import cv2
    from shapely.ops import unary_union

    grid = RasterGrid.covering(unary_union([footprint(p) for p in ok]).bounds, resolution, margin=200)
    image, owner = render_reference(iiif, pdok, grid, ok)
    # PDOK shown faded so the placed frames stand out.
    image[owner < 0] = (image[owner < 0] * 0.4 + 150).astype(image.dtype)
    cv2.imwrite(str(work / "mosaic.png"), image)
    (work / "mosaic.pgw").write_text(grid.world_file())
