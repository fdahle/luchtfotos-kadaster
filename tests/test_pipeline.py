from pathlib import Path

import numpy as np
import pytest

from luchtfotos_kadaster.annotations import georeference_annotation, resource_control_points
from luchtfotos_kadaster.catalog import Frame, KadasterSettings, load_settings, parse_sheet_index, site_frames
from luchtfotos_kadaster.georef import Placement, balanced_subset, fit_homography, local_gsd, transform
from luchtfotos_kadaster.iiif import IIIFImage, parse_manifest, region_url
from luchtfotos_kadaster.publish import publish_site
from luchtfotos_kadaster.reference import RasterGrid

REPO = Path(__file__).resolve().parents[1]

SHEET = """260917 TUDelft

Kaartbladnummers: b459 37Oost
RD: 084490 447676
DMS: 4°21'39"E 52°0'47"N

1933   2x 1933_459-608   1933_459-609
1981   3x 37111   37216   37218   (GEROTEERD)
2002   4x 37111   37113   37216   37218
"""

SERVICE = "https://dlc.services/iiif-img/7/40/0de855ee51500a189d7669bb32759f31"

MANIFEST_V2 = {
    "@id": "https://example.org/manifest",
    "sequences": [
        {
            "canvases": [
                {
                    "@id": "https://example.org/canvas/1",
                    "width": 11935,
                    "height": 11269,
                    "metadata": [
                        {"label": "String 2", "value": "delft"},
                        {"label": "String 3", "value": "2002"},
                    ],
                    "images": [
                        {
                            "resource": {
                                "@id": f"{SERVICE}/full/756,1024/0/default.jpg",
                                "service": {
                                    "@context": "http://iiif.io/api/image/2/context.json",
                                    "@id": SERVICE,
                                    "profile": "http://iiif.io/api/image/2/level2.json",
                                    "width": 11935,
                                    "height": 11269,
                                },
                            }
                        }
                    ],
                }
            ]
        }
    ],
}

MANIFEST_V3 = {
    "id": "https://example.org/v3",
    "items": [
        {
            "id": "https://example.org/v3/canvas/1",
            "width": 100,
            "height": 80,
            "metadata": [{"label": {"en": ["String 3"]}, "value": {"none": ["1954"]}}],
            "items": [
                {
                    "items": [
                        {
                            "body": {
                                "id": "https://example.org/img/abc/full/max/0/default.jpg",
                                "service": [{"id": "https://example.org/img/abc", "type": "ImageService3"}],
                            }
                        }
                    ]
                }
            ],
        }
    ],
}


def test_parse_sheet_index():
    sheet = parse_sheet_index(SHEET)
    assert sheet.centre_rd == (84490.0, 447676.0)
    assert sheet.sheets == "b459 37Oost"
    assert sheet.epochs[2002] == ("37111", "37113", "37216", "37218")
    assert sheet.epochs[1981] == ("37111", "37216", "37218")
    assert sheet.rotated_years == {1981}


def test_parse_manifest_v2_and_v3():
    (image,) = parse_manifest(MANIFEST_V2)
    assert image.service == SERVICE
    assert image.service_type == "ImageService2"
    assert (image.width, image.height) == (11935, 11269)
    assert image.canvas == "https://example.org/canvas/1"
    assert image.manifest == "https://example.org/manifest"
    assert image.meta("String 3") == "2002"
    assert image.image_id == "0de855ee51500a189d7669bb32759f31"

    (image,) = parse_manifest(MANIFEST_V3)
    assert image.service_type == "ImageService3"
    assert (image.width, image.height) == (100, 80)
    assert image.meta("String 3") == "1954"


def test_region_urls_never_request_the_full_scan_at_native_size():
    (image,) = parse_manifest(MANIFEST_V2)
    assert region_url(image, 0, 0, image.width, image.height, 1600) == f"{SERVICE}/full/1600,/0/default.jpg"
    assert region_url(image, 10, 20, 3000, 2000, 1024) == f"{SERVICE}/10,20,3000,2000/1024,/0/default.jpg"


def test_site_frames_use_index_names_and_metadata_year():
    images = parse_manifest(MANIFEST_V2)
    (frame,) = site_frames("delft", images, {images[0].image_id: "2002_37111"})
    assert (frame.year, frame.name, frame.key) == (2002, "2002_37111", "delft_2002_37111")


def test_settings_file_loads_and_resolves_paths():
    settings = load_settings(REPO / "config" / "settings.yml")
    assert set(settings.sites) == {"delft", "rotterdam"}
    for site in settings.sites.values():
        assert site.sheet.is_file()
        parse_sheet_index(site.sheet.read_text())
    assert settings.image_index.is_file()
    assert settings.site_dir == REPO / "docs"


def test_raster_grid_pixel_centres():
    grid = RasterGrid.around(1000.0, 2000.0, 50.0, 1.0)
    assert (grid.width, grid.height) == (100, 100)
    corner = transform(grid.pixel_to_world(), [(0, 0)])[0]
    assert corner == pytest.approx((950.5, 2049.5))
    assert grid.bounds == (950.0, 1950.0, 1050.0, 2050.0)


def test_fit_homography_handles_reflection_into_northing():
    rng = np.random.default_rng(1)
    resource = rng.uniform(0, 10000, (500, 2))
    truth = np.array([[0.38, 0.01, 83000.0], [0.012, -0.379, 450000.0], [1e-8, 2e-8, 1.0]])
    world = transform(truth, resource)
    world[:100] += rng.uniform(-500, 500, (100, 2))
    matrix, inliers = fit_homography(resource, world, 2.0)
    assert inliers[100:].all() and inliers[:100].sum() < 5
    assert np.abs(transform(matrix, resource[100:]) - world[100:]).max() < 0.05
    assert local_gsd(matrix, (5000, 5000)) == pytest.approx(0.38, rel=0.02)


def test_balanced_subset_spreads_points():
    points = np.r_[np.zeros((500, 2)) + 1, np.array([[100.0, 100.0], [100.0, 1.0], [1.0, 100.0]])]
    chosen = balanced_subset(points, 4)
    assert len(chosen) == 4
    assert {500, 501, 502} <= set(chosen.tolist())


def _placement(key, image_id):
    image = IIIFImage(
        service=f"https://example.org/img/{image_id}",
        service_type="ImageService2",
        width=1000,
        height=800,
        canvas=f"https://example.org/canvas/{image_id}",
        manifest="https://example.org/manifest",
    )
    year, name = key.split("_", 2)[1:]
    placement = Placement(Frame("delft", int(year), f"{year}_{name}", image), "ok")
    placement.homography = [[0.4, 0, 84000.0], [0, -0.4, 448000.0], [0, 0, 1.0]]
    placement.mask = [[0, 0], [1000, 0], [1000, 800], [0, 800]]
    placement.gcps = [[10.0, 20.0, 84004.0, 447992.0, "pdok"]]
    return placement


def test_georeference_annotation_shape():
    placement = _placement("delft_2002_37111", "a")
    annotation = georeference_annotation(placement, base="https://x/maps", transformation="projective")
    assert annotation["id"] == "https://x/maps/a"
    assert annotation["motivation"] == "georeferencing"
    source = annotation["target"]["source"]
    assert source["type"] == "ImageService2" and source["width"] == 1000
    assert source["partOf"][0]["partOf"][0]["type"] == "Manifest"
    assert annotation["target"]["selector"]["value"].startswith('<svg width="1000" height="800">')
    (feature,) = annotation["body"]["features"]
    assert feature["properties"]["resourceCoords"] == [10.0, 20.0]
    lon, lat = feature["geometry"]["coordinates"]
    assert 4.0 < lon < 4.6 and 51.8 < lat < 52.2
    assert annotation["body"]["transformation"] == {"type": "projective"}


def test_resource_control_points_pair_ties_and_links():
    newer = _placement("delft_2002_37111", "a")
    older = _placement("delft_1992_37110", "b")
    older.links = [[5.0, 6.0, newer.frame.key, 7.0, 8.0]]
    ties = [{"a": newer.frame.key, "b": older.frame.key, "points": [[1.0, 2.0, 3.0, 4.0]]}]
    rcps = resource_control_points([newer, older], ties, base="https://x/maps")
    assert len(rcps) == 4
    by_id = {}
    for rcp in rcps:
        assert rcp["type"] == "rcp"
        by_id.setdefault(rcp["id"], []).append(rcp)
    assert all(len(pair) == 2 for pair in by_id.values())
    link = next(pair for key, pair in by_id.items() if key.startswith("link:"))
    assert [r["mapId"] for r in link] == ["https://x/maps/b", "https://x/maps/a"]
    assert [r["resource"] for r in link] == [[5.0, 6.0], [7.0, 8.0]]


def test_publish_serves_each_annotation_under_its_id(tmp_path):
    import json

    base = "https://site.example/repo/maps"
    newer, older = _placement("delft_2002_37111", "a"), _placement("delft_1992_37110", "b")
    annotations = [georeference_annotation(p, base=base, transformation="projective", suffix=".json") for p in (newer, older)]
    assert annotations[0]["id"] == f"{base}/a.json"
    root = tmp_path / "runs" / "delft"
    (root / "2002").mkdir(parents=True)
    page = {"type": "AnnotationPage", "items": annotations}
    (root / "annotations.json").write_text(json.dumps(page))
    (root / "rcps.json").write_text("[]")
    (root / "2002" / "annotations.json").write_text(json.dumps(page | {"items": annotations[:1]}))
    (root / "2002" / "rcps.json").write_text("[]")
    frames = [
        {"name": p.frame.name, "year": p.frame.year, "status": "ok", "rmse_m": 2.0, "image_service": p.frame.image.service}
        for p in (newer, older)
    ]
    (root / "report.json").write_text(json.dumps({"manifest": "https://m", "frames": frames}))
    settings = KadasterSettings.model_validate(
        {
            "manifest_base": "https://m",
            "output_root": tmp_path / "runs",
            "sites": {"delft": {"sheet": tmp_path / "sheet.txt"}},
            "output": {"annotation_base": base, "annotation_suffix": ".json"},
        }
    )
    target = tmp_path / "site"
    publish_site(settings, "delft", target)
    assert json.loads((target / "maps" / "a.json").read_text())["id"] == f"{base}/a.json"
    (site,) = json.loads((target / "catalog.json").read_text())["sites"]
    assert [y["year"] for y in site["years"]] == [2002]
    assert [f["map"] for f in site["frames"]] == ["maps/a.json", "maps/b.json"]


def test_tiled_region_matches_direct_resample(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from luchtfotos_kadaster.iiif import IIIFClient

    rng = np.random.default_rng(3)
    scan = cv2.GaussianBlur(rng.integers(0, 255, (3000, 2600), dtype=np.uint8), (0, 0), 6)
    image = IIIFImage("https://example.org/img/x", "ImageService2", 2600, 3000, "", "")
    client = IIIFClient(tmp_path, tile_px=256)
    calls = []

    def fake_request(_image, x, y, w, h, width):
        calls.append((x, y, w, h, width))
        crop = scan[y : y + h, x : x + w]
        return cv2.resize(crop, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)

    client._request = fake_request
    region, (sx, sy) = client.region(image, 700, 900, 1200, 800, 300)
    assert region.shape == (200, 300) and sx == pytest.approx(0.25)
    # Every request is a level-4 tile on the fixed 1024 px grid.
    assert all(x % 1024 == 0 and y % 1024 == 0 for x, y, *_ in calls)
    direct = cv2.resize(scan[900:1700, 700:1900], (300, 200), interpolation=cv2.INTER_AREA)
    assert np.abs(region.astype(int) - direct).mean() < 3
    first = set(calls)
    calls.clear()
    client.region(image, 800, 1000, 600, 400, 150)
    # An overlapping request at the same level asks for tiles already fetched,
    # which the real client then serves from its URL cache.
    assert set(calls) <= first
