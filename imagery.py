"""Georeferenced RGB chips on a common grid: square pixels, 0.3 m, local UTM.

Two sources, both public:
- USDA NAIP aerial orthoimagery over the contiguous US (public domain), read as Cloud-Optimized
  GeoTIFF windows from Microsoft's Planetary Computer.
- SpaceNet 2 WorldView-3 satellite chips (CC BY-SA 4.0) from the SpaceNet bucket on AWS, which come
  with hand-drawn building footprints.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import httpx
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import from_bounds

GSD = 0.3  # metres per pixel for every chip the models see
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"
SN2 = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN2_buildings/train"
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "VSI_CACHE": "TRUE",
}

_sas: list[tuple[str, float]] = []


@dataclass
class Chip:
    rgb: np.ndarray  # H x W x 3 uint8
    transform: rasterio.Affine  # pixel -> metres in `crs`
    crs: str  # a UTM zone
    source: str
    captured: str
    native_gsd: float  # the source's own resolution, before resampling to GSD

    def to_lonlat(self) -> Transformer:
        return Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)


def utm_crs(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _sas_token() -> str:
    if _sas and _sas[0][1] > time.time() + 120:
        return _sas[0][0]
    r = httpx.get(PC_SAS, timeout=20).raise_for_status().json()
    _sas[:] = [(r["token"], time.time() + 30 * 60)]
    return r["token"]


def naip_chip(lon: float, lat: float, size_m: float = 240.0) -> Chip:
    """The newest NAIP acquisition around (lon, lat), mosaicked across tile edges."""
    half_deg = size_m / 111_320
    bbox = [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]
    body = {
        "collections": ["naip"],
        "bbox": bbox,
        "limit": 40,
        "sortby": [{"field": "datetime", "direction": "desc"}],
    }
    feats = httpx.post(PC_STAC, json=body, timeout=30).raise_for_status().json()["features"]
    if not feats:
        raise LookupError("No NAIP imagery here. NAIP covers the contiguous United States.")
    newest = feats[0]["properties"]
    year, state = newest["datetime"][:4], newest.get("naip:state")
    items = [f for f in feats if f["properties"]["datetime"][:4] == year and f["properties"].get("naip:state") == state]

    crs = utm_crs(lon, lat)
    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    n = int(round(size_m / GSD))
    x0, y1 = cx - n * GSD / 2, cy + n * GSD / 2
    dst_transform = from_origin(x0, y1, GSD, GSD)
    out = np.zeros((3, n, n), np.uint8)
    token = _sas_token()
    with rasterio.Env(**GDAL_ENV):
        for f in items:
            with rasterio.open(f["assets"]["image"]["href"] + "?" + token) as ds:
                band = np.zeros_like(out)
                # Reproject a window around the chip only; NAIP tiles are already in a UTM zone,
                # usually this one, so this is a resample, not a real warp.
                left, bottom, right, top = Transformer.from_crs(crs, ds.crs, always_xy=True).transform_bounds(
                    x0, y1 - n * GSD, x0 + n * GSD, y1
                )
                win = from_bounds(left, bottom, right, top, ds.transform).round_offsets().round_lengths()
                win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
                src = ds.read([1, 2, 3], window=win)
                reproject(
                    src,
                    band,
                    src_transform=ds.window_transform(win),
                    src_crs=ds.crs,
                    dst_transform=dst_transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                    src_nodata=0,
                    dst_nodata=0,
                )
                fill = (out.sum(0) == 0) & (band.sum(0) > 0)
                out[:, fill] = band[:, fill]
                native = abs(ds.transform.a)
            if (out.sum(0) > 0).all():
                break
    if (out.sum(0) == 0).mean() > 0.5:
        raise LookupError("NAIP has no coverage for most of this area.")
    return Chip(
        rgb=np.ascontiguousarray(out.transpose(1, 2, 0)),
        transform=dst_transform,
        crs=crs,
        source=f"USDA NAIP {year}",
        captured=newest["datetime"][:10],
        native_gsd=native,
    )


def _stretch(arr: np.ndarray) -> np.ndarray:
    """16-bit pan-sharpened bands to 8-bit with a 2-98 percentile stretch per band."""
    out = np.zeros(arr.shape, np.uint8)
    for b in range(arr.shape[0]):
        band = arr[b].astype(np.float32)
        valid = band[band > 0]
        lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
        out[b] = np.clip((band - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)
    return out


def spacenet_chip(aoi: str, img: int) -> tuple[Chip, list[list[list[float]]]]:
    """A SpaceNet 2 chip (e.g. aoi='AOI_2_Vegas') and its labelled footprints as lon/lat rings."""
    url = f"/vsicurl/{SN2}/{aoi}/PS-RGB/SN2_buildings_train_{aoi}_PS-RGB_img{img}.tif"
    with rasterio.Env(**GDAL_ENV), rasterio.open(url) as ds:
        src = _stretch(ds.read([1, 2, 3]))
        b = ds.bounds
        lon, lat = (b.left + b.right) / 2, (b.top + b.bottom) / 2
        crs = utm_crs(lon, lat)
        to_utm = Transformer.from_crs(ds.crs, crs, always_xy=True)
        left, bottom, right, top = to_utm.transform_bounds(*b)
        w, h = int(math.ceil((right - left) / GSD)), int(math.ceil((top - bottom) / GSD))
        dst_transform = from_origin(left, top, GSD, GSD)
        out = np.zeros((3, h, w), np.uint8)
        reproject(
            src,
            out,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
        )
        native = abs(ds.transform.a) * 111_320 * math.cos(math.radians(lat))
    label_url = f"{SN2}/{aoi}/geojson_buildings/SN2_buildings_train_{aoi}_geojson_buildings_img{img}.geojson"
    labels = []
    for f in httpx.get(label_url, timeout=30).raise_for_status().json()["features"]:
        g = f.get("geometry") or {}
        if g.get("type") == "Polygon":
            labels.append([[round(x, 7), round(y, 7)] for x, y, *_ in g["coordinates"][0]])
    chip = Chip(
        rgb=np.ascontiguousarray(out.transpose(1, 2, 0)),
        transform=dst_transform,
        crs=crs,
        source="SpaceNet 2 / Maxar WorldView-3",
        captured="2015",
        native_gsd=round(native, 2),
    )
    return chip, labels
