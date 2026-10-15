"""Copy a site's outputs into the static site under ``docs/`` (GitHub Pages).

Layout under the target directory::

    maps/<image-id><suffix>         one Georeference Annotation, served under its own id
    <site>/annotations.json         AnnotationPage of the whole site
    <site>/rcps.json                RCPs of the whole site
    <site>/report.json              per-frame statistics
    <site>/<year>/annotations.json  AnnotationPage of one epoch
    <site>/<year>/rcps.json         RCPs of one epoch
    catalog.json                    sites, epochs and frames, read by the index page

Annotation ids are ``<annotation_base>/<image-id><suffix>``, so they resolve
once the target is served at ``annotation_base`` minus its last segment.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from luchtfotos_kadaster.catalog import KadasterSettings


def publish_site(settings: KadasterSettings, site: str, target: Path | None = None) -> dict:
    target = Path(target or settings.site_dir)
    root = settings.output_root / site
    if not (root / "annotations.json").is_file():
        raise FileNotFoundError(f"No outputs for {site!r} in {root}; run `luchtfotos-kadaster run {site}` first")
    base = settings.output.annotation_base.rstrip("/") + "/"
    page = json.loads((root / "annotations.json").read_text())

    maps = target / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    map_files = {}
    for annotation in page["items"]:
        if not annotation["id"].startswith(base):
            raise ValueError(
                f"Annotation id {annotation['id']} is not under annotation_base {base}; rerun the site first"
            )
        name = annotation["id"][len(base):]
        (maps / name).write_text(json.dumps(annotation, indent=1))
        map_files[annotation["target"]["source"]["id"]] = f"maps/{name}"

    out = target / site
    out.mkdir(parents=True, exist_ok=True)
    for name in ("annotations.json", "rcps.json", "report.json"):
        shutil.copyfile(root / name, out / name)
    years = []
    for work in sorted((p for p in root.iterdir() if p.name.isdigit()), key=lambda p: int(p.name), reverse=True):
        if not (work / "annotations.json").is_file():
            continue
        (out / work.name).mkdir(exist_ok=True)
        for name in ("annotations.json", "rcps.json"):
            shutil.copyfile(work / name, out / work.name / name)
        epoch = json.loads((work / "annotations.json").read_text())
        years.append(
            {
                "year": int(work.name),
                "annotations": f"{site}/{work.name}/annotations.json",
                "rcps": f"{site}/{work.name}/rcps.json",
                "maps": len(epoch["items"]),
            }
        )

    report = json.loads((root / "report.json").read_text())
    entry = {
        "site": site,
        "manifest": report["manifest"],
        "annotations": f"{site}/annotations.json",
        "rcps": f"{site}/rcps.json",
        "report": f"{site}/report.json",
        "years": years,
        "frames": [
            {
                "name": frame["name"],
                "year": frame["year"],
                "status": frame["status"],
                "rmse_m": frame.get("rmse_m"),
                "map": map_files.get(frame["image_service"]),
            }
            for frame in report["frames"]
        ],
    }
    catalog_path = target / "catalog.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.is_file() else {"sites": []}
    catalog["sites"] = sorted(
        [s for s in catalog["sites"] if s["site"] != site] + [entry], key=lambda s: s["site"]
    )
    catalog_path.write_text(json.dumps(catalog, indent=1))
    return entry
