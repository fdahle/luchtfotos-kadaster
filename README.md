# Luchtfoto's Kadaster

Historical aerial photographs from the Kadaster, georeferenced through IIIF
and published as [Allmaps](https://allmaps.org) Georeference Annotations.
Resource Control Points tie the photographs to each other.

**Browse:** <https://fdahle.github.io/luchtfotos-kadaster/>. Every year and
every photo opens directly in the Allmaps Viewer.

This repository holds only annotations and statistics. The photographs remain
the Kadaster's. They are served through IIIF from the DLCS heritage
infrastructure while the Kadaster sets up its own.

## Test set (October 2026)

A vertical cut through time for two cities, every available epoch from the
1930s to 2002, 39 photographs in total:

| | Delft (sheet b459 37 Oost) | | Rotterdam (sheet b502 37 Oost) | |
|---|---|---|---|---|
| **Year** | **Photos** | **RMSE (m)** | **Photos** | **RMSE (m)** |
| 2002 | 4 | 1.9–2.2 | 3 | 1.8–1.9 |
| 1992 | 4 | 1.5–2.2 | 2 | 1.6–1.9 |
| 1981 | 3 | 2.2 | 4 | 1.8–2.5 |
| 1971 | 2 | 1.7–2.0 | 3 | 2.2–2.3 |
| 1961 | 2 | 1.7–2.2 | 3 | 1.8–2.1 |
| 1954 | 2 | 1.5–2.3 | 2 | 2.2–2.4 |
| 1933 / 1934 | 2 | 1.6–1.9 | 3 | 1.8–2.5 |
| **RCPs** | 2158 | | 2260 | |

All photos are placed. The RMSE is the fit of each photo's transformation to
its control points. That control comes from the current PDOK orthophoto for
2002, and from the already placed newer photos for the older years. It measures
agreement with modern imagery, not surveyed ground control.

## Files

| Path | Contents |
|---|---|
| `maps/<image-id>.json` | One Georeference Annotation. Its `id` is its own URL on this site |
| `<site>/annotations.json` | AnnotationPage with every photo of the site |
| `<site>/<year>/annotations.json` | AnnotationPage with one year |
| `<site>/rcps.json`, `<site>/<year>/rcps.json` | Resource Control Points, see below |
| `<site>/report.json` | Per-photo statistics: RMSE and median residual, matches and inliers, ground resolution, share of control taken from newer photos |
| `catalog.json` | Sites, years and photos, used by the index page |

Each annotation targets the photo's IIIF image service. Under `partOf` it also
records the canvas and the manifest the photo came from. The transformation
is `projective`, with 40 GCPs in WGS84 that are spread over the photo, and an
SVG mask that excludes the margins and data strip.

To load a year in Allmaps: `https://viewer.allmaps.org/?url=https://fdahle.github.io/luchtfotos-kadaster/delft/1954/annotations.json`

## Resource Control Points

The RCPs are written in the format of
[`@allmaps/attach`](https://github.com/allmaps/allmaps/tree/main/packages/attach):

```json
{"type": "rcp", "id": "tie:delft_2002_37111:delft_2002_37113:0", "mapId": "https://fdahle.github.io/luchtfotos-kadaster/maps/<image-id>.json", "resource": [x, y]}
```

Two RCPs with the same `id` on two maps form one attachment. The prefix of the
id says where the point comes from:

- `tie:` pairs two overlapping photos of the same year.
- `link:` pairs a photo with the newer photo it took its control from.

```js
import { parseAnnotation } from '@allmaps/annotation'
import { AttachedTransformation } from '@allmaps/attach'

const maps = parseAnnotation(await (await fetch(annotationsUrl)).json())
const rcps = await (await fetch(rcpsUrl)).json()
const adjusted = AttachedTransformation.fromGeoreferencedMaps(maps, rcps).toGeoreferencedMaps()
```

The annotations themselves are not adjusted jointly. Each one is fitted on
its own, and the RCPs are there for the joint adjustment.

## How the photos were placed

1. **IIIF first.** No scan is downloaded whole. A reduced overview (1600 px
   wide) gives each photo's approximate position. After that, only image
   segments are read, as fixed tiles at reduced resolution. A cold run of a
   city fetched about a quarter of the full-resolution pixels. The `iiif`
   ledger in `report.json` covers only the latest run, which was served from
   cache.
2. **Location hint.** The only hint is one RD coordinate per city, the
   sheet-index centre. Each overview is matched against an 8 × 8 km PDOK
   window around that point. If that fails, the window is shifted by 3 km.
   Photos that still don't match are placed by matching them against
   neighbouring photos that are already placed.
3. **Recent first, then backwards.** 2002 is matched against PDOK's current
   orthophoto. Each older year is matched against the newer years' photos,
   already placed and rendered from IIIF, with PDOK filling the gaps.
4. **Refinement.** The photo is split into a 4 × 4 grid and every cell is
   matched at about 1 m/px. A projective transformation is then fitted on all
   inliers. Matching uses RoMa v2.

## Known limitations

- The 1930s masks still include the black corner marks.
- Rotterdam 1934_37O-739 and 37O-749 have far fewer refinement inliers than the
  other photos (4473 and 2545). They look plausible, but deserve a check.
- One projective transformation per photo ignores relief and lens distortion.
  Both cities are flat, so relief displacement shows up mainly on tall
  buildings.

## Producing the files

The files are generated by the `kadaster` subproject of Felix Dahle's
historical-SfM pipeline (`hist-sfm kadaster run <site>`, then
`hist-sfm kadaster publish <site> --to <this checkout>`). Don't edit them by
hand: the annotation ids and the RCP `mapId`s must stay in sync.
