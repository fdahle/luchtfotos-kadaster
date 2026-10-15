"""Georeference Annotations and Resource Control Points for Allmaps.

A placed frame becomes one Georeference Annotation
(https://iiif.io/api/extension/georef/) on its IIIF image service, with the
canvas and manifest it was read from under ``partOf``. Tie points become
``@allmaps/attach`` Resource Control Points: two RCPs with the same ``id`` on
two maps make one attachment, and ``mapId`` is the annotation id of the frame
the point lies on.
"""

from __future__ import annotations

from luchtfotos_kadaster.georef import Placement

GEOREF_CONTEXT = [
    "http://iiif.io/api/extension/georef/1/context.json",
    "http://iiif.io/api/presentation/3/context.json",
]

_TO_WGS84 = None


def rd_to_wgs84(points):
    """EPSG:28992 -> (lon, lat). PROJ uses the RDNAPTRANS grid when it has it."""
    global _TO_WGS84
    from pyproj import Transformer

    if _TO_WGS84 is None:
        _TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    return [_TO_WGS84.transform(x, y) for x, y in points]


def map_id(placement: Placement, base: str, suffix: str = "") -> str:
    return f"{base.rstrip('/')}/{placement.frame.image.image_id}{suffix}"


def georeference_annotation(placement: Placement, *, base: str, transformation: str, suffix: str = "") -> dict:
    image = placement.frame.image
    polygon = " ".join(f"{round(x)},{round(y)}" for x, y in placement.mask)
    lonlat = rd_to_wgs84([(gcp[2], gcp[3]) for gcp in placement.gcps])
    source = {
        "id": image.service,
        "type": image.service_type,
        "height": image.height,
        "width": image.width,
    }
    if image.canvas:
        canvas = {"id": image.canvas, "type": "Canvas"}
        if image.manifest:
            canvas["partOf"] = [{"id": image.manifest, "type": "Manifest"}]
        source["partOf"] = [canvas]
    return {
        "@context": GEOREF_CONTEXT,
        "id": map_id(placement, base, suffix),
        "type": "Annotation",
        "motivation": "georeferencing",
        "target": {
            "type": "SpecificResource",
            "source": source,
            "selector": {
                "type": "SvgSelector",
                "value": (
                    f'<svg width="{image.width}" height="{image.height}">'
                    f'<polygon points="{polygon}" /></svg>'
                ),
            },
        },
        "body": {
            "type": "FeatureCollection",
            "transformation": {"type": transformation},
            "features": [
                {
                    "type": "Feature",
                    "properties": {"resourceCoords": [gcp[0], gcp[1]]},
                    "geometry": {"type": "Point", "coordinates": [round(lon, 8), round(lat, 8)]},
                }
                for gcp, (lon, lat) in zip(placement.gcps, lonlat)
            ],
        },
    }


def annotation_page(annotations: list[dict]) -> dict:
    return {"@context": GEOREF_CONTEXT, "type": "AnnotationPage", "items": annotations}


def resource_control_points(
    placements: list[Placement], ties: list[dict], *, base: str, suffix: str = ""
) -> list[dict]:
    """RCPs from within-epoch ties and from cross-epoch links of the given placements.

    ``placements`` must include every frame a link points at, since its map id
    is needed for the other half of the pair.
    """
    by_key = {p.frame.key: p for p in placements}
    rcps = []

    def pair(identifier, first, first_xy, second, second_xy):
        for placement, (x, y) in ((first, first_xy), (second, second_xy)):
            rcps.append(
                {"type": "rcp", "id": identifier, "mapId": map_id(placement, base, suffix), "resource": [x, y]}
            )

    for tie in ties:
        first, second = by_key[tie["a"]], by_key[tie["b"]]
        for index, (ax, ay, bx, by) in enumerate(tie["points"]):
            pair(f"tie:{tie['a']}:{tie['b']}:{index}", first, (ax, ay), second, (bx, by))
    for placement in placements:
        for index, (rx, ry, other, ox, oy) in enumerate(placement.links):
            if other in by_key:
                pair(f"link:{placement.frame.key}:{other}:{index}", placement, (rx, ry), by_key[other], (ox, oy))
    return rcps
