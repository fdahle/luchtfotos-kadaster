# Luchtfoto's Kadaster

Historical aerial photographs from the Kadaster, georeferenced through IIIF
and published as [Allmaps](https://allmaps.org) Georeference Annotations.
Resource Control Points tie the photographs to each other.

The repository holds the pipeline (`src/`, `config/`) and its published
results (`docs/`). The photographs themselves remain the Kadaster's. They are
served through IIIF from the DLCS heritage infrastructure while the Kadaster
sets up its own, and the pipeline never downloads a full scan.

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

Known limitations:

- The 1930s masks still include the black corner marks.
- Rotterdam 1934_37O-739 and 37O-749 have far fewer refinement inliers than the
  other photos (4473 and 2545). They look plausible, but deserve a check.
- One projective transformation per photo ignores relief and lens distortion.
  Both cities are flat, so relief displacement shows up mainly on tall
  buildings.

## Published files (`docs/`)

`docs/` is a static site. `docs/index.html` lists every year and photo, with
links that open them in the Allmaps Viewer. It is not online yet. Once
publishing is agreed with the Kadaster, it is served by GitHub Pages from
`docs/` at `https://fdahle.github.io/luchtfotos-kadaster/`. The annotation ids
already point there.

| Path | Contents |
|---|---|
| `maps/<image-id>.json` | One Georeference Annotation. Its `id` is its own URL on the site |
| `<site>/annotations.json` | AnnotationPage with every photo of the site |
| `<site>/<year>/annotations.json` | AnnotationPage with one year |
| `<site>/rcps.json`, `<site>/<year>/rcps.json` | Resource Control Points, see below |
| `<site>/report.json` | Per-photo statistics: RMSE and median residual, matches and inliers, ground resolution, share of control taken from newer photos |
| `catalog.json` | Sites, years and photos, used by the index page |

Each annotation targets the photo's IIIF image service. Under `partOf` it also
records the canvas and the manifest the photo came from. The transformation
is `projective`, with 40 GCPs in WGS84 that are spread over the photo, and an
SVG mask that excludes the margins and data strip.

### Resource Control Points

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

## Running the pipeline

```
pip install -e .          # RoMa v2 wants a CUDA GPU; on CPU it is very slow
luchtfotos-kadaster run delft
luchtfotos-kadaster run rotterdam --year 2002 --year 1992
luchtfotos-kadaster run delft --force      # ignore saved placements, refetch the manifest
luchtfotos-kadaster publish delft          # copy the results into docs/
pytest
```

`config/settings.yml` sets the manifests, sites, reference, matcher and
thresholds. `output_root` is the working directory: resume state, per-epoch
outputs, quick-look mosaics and the IIIF, PDOK and matcher caches. Change it
to a local path. A run resumes every photo that is already placed. A rerun
after changing `annotation_base` comes from cache and only rewrites the ids.

`config/images.csv` is the DLCS export that came with the photographs. It maps
each image-service id to its original file name, because the generated
manifest labels canvases only "Canvas n". `config/sheets/*.txt` are the
sheet-index notes that came with them: sheet number, RD centre and the
photos of each year.

## How a photo is placed

1. **IIIF first, never the whole scan.** The manifest
   `<manifest_base>/<site>` gives every photo's image service. The year comes
   from the canvas metadata (`String 3`). A reduced overview
   (`full/1600,/0/default.jpg`) gives the approximate position. After that only
   image segments are read, as fixed 1024 px tiles at power-of-two scale
   levels, the way a deep-zoom viewer reads an image. Overlapping requests,
   such as a newer photo rendered again as reference for every older year,
   then hit tiles already cached. A cold run of a city fetched about a quarter
   of the full-resolution pixels. The `iiif` ledger in `report.json` covers
   only the latest run.
2. **Coarse placement from one location hint.** The only hint is one RD
   coordinate per city, the `RD:` line of the sheet-index file. Each overview
   is matched against an 8 × 8 km reference at 4 m/px around that point,
   giving one projective transformation from image pixels to RD New
   (EPSG:28992). Matches on the black margins and data strip are excluded
   from the photo's mask. If the centred window fails, windows shifted by 3 km
   are tried, then other rotations. A photo that still fails is matched
   overview-to-overview against photos already placed, same year first, then
   newer years. A reference built from many photos over PDOK can be much harder
   to match than one of the photos it is made of. The 1930s photos and Delft
   1961_37O-II-39 needed this fallback.
3. **Recent first, then backwards.** The newest year is matched against PDOK's
   current orthophoto (`Actueel_ortho25`). Each older year is matched against
   the photos already placed from the newer years, warped onto RD from their
   own IIIF tiles, with PDOK filling whatever they don't cover. A control point
   taken from a newer photo is both a GCP and an RCP `link:` between the two
   photos.
4. **Refinement.** The placed photo is split into a 4 × 4 grid. Each cell is
   read at about 1 m/px and matched against a reference rendered for exactly
   that RD box. The transformation is then refitted on every inlier. A photo
   passes at ≤ 6 m RMSE with at least 20 GCPs.

Matching uses [RoMa v2](https://pypi.org/project/romav2/) (`precise`
preset), a dense matcher that holds up across decades of change and
arbitrary rotation. Results are cached under a hash of both images' bytes and
the settings, and every pair is seeded, so a rerun reproduces them.
