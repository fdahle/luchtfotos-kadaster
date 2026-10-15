"""Kadaster aerial photographs (Delft, Rotterdam) georeferenced through IIIF.

Frames are read from IIIF manifests and Image API services rather than
downloaded, placed on RD New against PDOK and against each other, and written
out as Allmaps Georeference Annotations with Resource Control Points between
frames.
"""

from luchtfotos_kadaster.catalog import KadasterSettings, load_settings
from luchtfotos_kadaster.pipeline import run_site
from luchtfotos_kadaster.publish import publish_site

__all__ = ["KadasterSettings", "load_settings", "publish_site", "run_site"]
