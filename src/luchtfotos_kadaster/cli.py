"""``luchtfotos-kadaster run <site>`` and ``luchtfotos-kadaster publish <site>``."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SETTINGS = Path(__file__).resolve().parents[2] / "config" / "settings.yml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="luchtfotos-kadaster", description="Georeference Kadaster IIIF aerial photographs for Allmaps"
    )
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="Settings file")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Place one site's frames, newest epoch first, and write annotations")
    run.add_argument("site", help="Site under sites: in the settings, e.g. delft")
    run.add_argument("--year", type=int, action="append", dest="years", help="Only this epoch (repeatable)")
    run.add_argument("--force", action="store_true", help="Ignore saved placements and refetch the manifest")
    publish = commands.add_parser("publish", help="Copy one site's annotations, RCPs and report into docs/")
    publish.add_argument("site", help="Site under sites: in the settings, e.g. delft")
    publish.add_argument("--to", type=Path, help="Target directory (default: site_dir from the settings)")
    args = parser.parse_args(argv)

    from luchtfotos_kadaster import load_settings, publish_site, run_site

    settings = load_settings(args.settings)
    if args.command == "publish":
        entry = publish_site(settings, args.site, args.to)
        target = args.to or settings.site_dir
        print(f"[{args.site}] published {len(entry['frames'])} frames, {len(entry['years'])} epochs to {target}")
    else:
        run_site(settings, args.site, years=args.years, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
