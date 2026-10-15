#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Stream one analysis from a deployed server and print what it found.

./example.py https://<your-web-url> --preset scottsdale
./example.py https://<your-web-url> --lat 33.615 --lon -111.889 --geojson out.geojson
"""

from __future__ import annotations

import argparse
import collections
import json

import httpx


def stream(base: str, params: dict[str, str]):
    with httpx.stream("GET", f"{base.rstrip('/')}/api/analyze", params=params, timeout=120) as r:
        r.raise_for_status()
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                yield event, json.loads(line[5:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--preset")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--geojson", help="write the structures to this file")
    args = ap.parse_args()
    params = {"preset": args.preset} if args.preset else {"lat": str(args.lat), "lon": str(args.lon)}

    items: dict[int, dict] = {}
    for event, data in stream(args.url, params):
        if event == "image":
            print(
                f"imagery: {data['source']}, {data['captured']}, {data['width']}x{data['height']} px ({data['fetch_ms']} ms)"
            )
        elif event == "structures":
            items = {it["id"]: it for it in data["items"]}
            print(f"outlines: {len(items)} structures ({data['detect_ms']} ms)")
        elif event == "class":
            items[data["id"]].update(data)
        elif event == "done":
            print(f"classified {data['classified']} buildings ({data['classify_ms']} ms); total {data['total_ms']} ms")
        elif event == "failed":
            raise SystemExit(data["message"])

    kinds = collections.Counter(it.get("structure_type", it["kind"]) for it in items.values())
    print(json.dumps(dict(kinds.most_common()), indent=1))
    biggest = max(items.values(), key=lambda it: it["area_m2"])
    print(json.dumps({k: v for k, v in biggest.items() if k not in ("ring", "px")}, indent=1))
    if args.geojson:
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [it["ring"]]},
                "properties": {k: v for k, v in it.items() if k not in ("ring", "px")},
            }
            for it in items.values()
        ]
        with open(args.geojson, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)
        print(f"wrote {len(features)} features to {args.geojson}")


if __name__ == "__main__":
    main()
