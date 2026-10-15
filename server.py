"""Structure extraction from aerial and satellite imagery.

GET /api/analyze?preset=<id> or ?lat=..&lon=.. streams Server-Sent Events:
  image      the chip (JPEG data URL), its source, capture date and lon/lat bounds
  structures every building and pool outline as a lon/lat ring with area, perimeter, length,
             width and orientation (and, for SpaceNet chips, the hand-labelled footprints)
  scene      the vision-language model's two-sentence read of the whole chip
  class      one per building as the model answers: type, roof shape, roof colour
  done       counts and per-stage timings
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image, ImageDraw
from pyproj import Transformer

from imagery import GSD, Chip, naip_chip, spacenet_chip
from vision import Models, Structure, measure

INDEX_HTML = Path(__file__).with_name("index.html")
VLM_URL = os.environ.get("VLM_URL", "http://localhost:8000").rstrip("/")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct-FP8")
MAX_CLASSIFIED = 160  # largest buildings first; the rest keep their geometry only
VLM_CONCURRENCY = 32
CUSTOM_PER_10_MIN = 60  # replica-wide budget for visitor-chosen locations
# The contiguous US, where NAIP has coverage.
CONUS = (-125.0, 24.4, -66.9, 49.5)

PRESETS: list[dict[str, Any]] = [
    # Aerial: USDA NAIP, public domain.
    {"id": "scottsdale", "name": "Scottsdale, AZ", "kind": "aerial", "lon": -111.8890, "lat": 33.6150,
     "blurb": "Tile roofs and backyard pools"},
    {"id": "cape-coral", "name": "Cape Coral, FL", "kind": "aerial", "lon": -81.9650, "lat": 26.5650,
     "blurb": "Gulf Coast homes, tarps after a hurricane season"},
    {"id": "mesa", "name": "Mesa, AZ", "kind": "aerial", "lon": -111.7400, "lat": 33.4100,
     "blurb": "Townhomes around a community pool"},
    {"id": "mobile-park", "name": "Mesa, AZ", "kind": "aerial", "lon": -111.7689, "lat": 33.41211,
     "blurb": "A manufactured home park"},
    {"id": "des-moines", "name": "Des Moines, IA", "kind": "aerial", "lon": -93.6500, "lat": 41.6000,
     "blurb": "Shops and offices beside houses"},
    {"id": "elk-grove", "name": "Elk Grove Village, IL", "kind": "aerial", "lon": -87.9750, "lat": 42.0050,
     "blurb": "Warehouses and distribution centers"},
    {"id": "fridley", "name": "Fridley, MN", "kind": "aerial", "lon": -93.25404, "lat": 45.07707,
     "blurb": "A high school and its sports fields"},
    {"id": "farm", "name": "Polk County, IA", "kind": "aerial", "lon": -93.50835, "lat": 41.7314,
     "blurb": "A farmstead: machine sheds and grain bins"},
    # Satellite: SpaceNet 2, Maxar WorldView-3, CC BY-SA 4.0. Hand-labelled footprints included.
    {"id": "vegas", "name": "Las Vegas, NV", "kind": "satellite", "aoi": "AOI_2_Vegas", "img": 1061,
     "blurb": "Townhomes, houses and pools"},
    {"id": "paris", "name": "Paris suburbs", "kind": "satellite", "aoi": "AOI_3_Paris", "img": 1086,
     "blurb": "Pavillons under trees, shot off-nadir"},
    {"id": "shanghai", "name": "Shanghai", "kind": "satellite", "aoi": "AOI_4_Shanghai", "img": 1097,
     "blurb": "Villas with red and slate roofs"},
]  # fmt: skip

state: dict[str, Any] = {}
custom_calls: deque[float] = deque()


class Pipeline:
    def __init__(self) -> None:
        super().__init__()
        import torch

        self.models = Models("cuda" if torch.cuda.is_available() else "cpu")
        # One chip on the GPU at a time; a second visitor waits a few seconds behind the first.
        self.lock = threading.Lock()
        self.chips: dict[str, tuple[Chip, list]] = {}

    def chip_for(self, preset: dict[str, Any]) -> tuple[Chip, list]:
        if preset["id"] not in self.chips:
            if preset["kind"] == "satellite":
                self.chips[preset["id"]] = spacenet_chip(preset["aoi"], preset["img"])
            else:
                self.chips[preset["id"]] = (naip_chip(preset["lon"], preset["lat"]), [])
        return self.chips[preset["id"]]

    def detect(self, chip: Chip) -> list[Structure]:
        with self.lock:
            return self.models.detect(chip)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # uvicorn does not accept connections until this returns, so the replica only reports ready
    # with the models loaded, warm, and every preset's imagery cached.
    pipe = await asyncio.to_thread(Pipeline)
    for p in PRESETS:
        try:
            chip, _ = await asyncio.to_thread(pipe.chip_for, p)
        except Exception as e:  # noqa: BLE001  imagery services can blip; fetch on first use instead
            print(f"preset {p['id']} not prefetched: {e!r}", flush=True)
    await asyncio.to_thread(pipe.detect, pipe.chip_for(PRESETS[0])[0])
    state["pipe"] = pipe
    state["http"] = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    state["vlm"] = asyncio.Semaphore(VLM_CONCURRENCY)
    yield
    await state["http"].aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse({"ok": "pipe" in state, "presets": len(PRESETS)})


@app.get("/api/presets")
async def presets() -> JSONResponse:
    return JSONResponse([{k: v for k, v in p.items() if k not in ("aoi", "img")} for p in PRESETS])


# -- classification -------------------------------------------------------------------------------

TYPES = [
    "single_family_house",
    "multi_family",
    "manufactured_home",
    "commercial",
    "industrial",
    "institutional",
    "agricultural",
    "accessory",
]
COLORS = [
    "grey",
    "dark_grey_or_black",
    "brown",
    "tan_or_beige",
    "orange_or_terracotta",
    "red",
    "white",
    "green",
    "blue",
]
SCHEMA = {
    "type": "object",
    "properties": {
        # Colour first: naming what it sees before deciding on a type keeps the model grounded.
        "roof_color": {"type": "string", "enum": COLORS},
        "structure_type": {"type": "string", "enum": TYPES},
        "roof_shape": {"type": "string", "enum": ["gable", "hip", "flat", "complex"]},
    },
    "required": ["roof_color", "structure_type", "roof_shape"],
    "additionalProperties": False,
}
SCENE_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string", "maxLength": 420}},
    "required": ["description"],
    "additionalProperties": False,
}
SCENE_PROMPT = (
    "Top-down {kind} image, {gsd:.1f} m per pixel, {w:.0f} m across. In at most 50 words, describe the land use "
    "and the kinds of buildings: for example detached houses on lots, attached townhomes, apartment "
    "complexes, a mobile home park, a commercial strip, downtown blocks, an industrial park, farmland. "
    "If parts of the image differ, say which part is which. Answer in English."
)
# The definitions are a system message, identical on every call, so vLLM's prefix cache computes
# them once instead of once per building.
CLASS_SYSTEM = (
    "You classify one structure in a top-down aerial or satellite image. It is outlined in magenta.\n"
    "structure_type:\n"
    "- single_family_house: one detached home on its own lot, usually with a driveway and yard.\n"
    "- multi_family: attached homes or apartments: townhouse rows, duplexes, condo and apartment buildings. "
    "Look for repeated roof sections, several identical vents or entries, shared parking lots or a shared pool.\n"
    "- manufactured_home: mobile or manufactured homes: long narrow rectangles, often 4-6 m wide and "
    "50-130 m², set in rows in a park, each on its own small lot.\n"
    "- commercial: shops, restaurants, offices, hotels, gas stations, downtown buildings; usually with a "
    "customer parking lot or street frontage.\n"
    "- industrial: warehouses, factories and distribution centres: very large plain roofs, truck docks, "
    "trailer yards. Usually over 1500 m².\n"
    "- institutional: schools, churches, hospitals, government and community buildings.\n"
    "- agricultural: barns, silos, grain bins, livestock and machine sheds on farmland.\n"
    "- accessory: detached garages, sheds, carports, pool houses: a small structure in the yard or lot of a "
    "larger main building.\n"
    "The neighbourhood summary may not hold for every building; decide from this structure and its own lot.\n"
    "roof_shape: gable (two slopes meeting at a ridge), hip (slopes on all sides), flat, complex "
    "(many intersecting sections).\n"
    "roof_color: the roof's dominant colour as it appears."
)
CLASS_PROMPT = (
    "Neighbourhood, from a wider view: {scene}\n"
    "Top-down {kind} image, {gsd:.1f} m per pixel; the crop is {w:.0f} m across. "
    "Classify the structure outlined in magenta. Its footprint is {area:.0f} m²."
)


def crop_for(rgb: np.ndarray, s: Structure) -> tuple[str, float]:
    """A JPEG data URL of the structure with its neighbours, outlined, and the crop's width in metres."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = s.poly_px.bounds
    size = max(x1 - x0, y1 - y0)
    # Wide on purpose: whether a roof is a house, a townhome row or a shop shows in its lot, its
    # parking and its neighbours more than in the roof itself. The margin is capped so a large roof
    # keeps enough pixels for its own details.
    side = size + min(size * 1.6, 120) + 60
    out_px = 448 if side <= 448 else 672
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    bx1, by1 = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    img = Image.fromarray(rgb[by0:by1, bx0:bx1])
    scale = out_px / max(img.size)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC)
    ImageDraw.Draw(img).line(
        [((x - bx0) * scale, (y - by0) * scale) for x, y in s.ring_px], fill=(255, 0, 255), width=2
    )
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), (bx1 - bx0) * GSD


def _auth_headers() -> dict[str, str]:
    # The model group is authenticated; this replica's workload identity token rotates on disk.
    path = os.environ.get("CHALK_WEB_IDENTITY_TOKEN_FILE")
    return {"Authorization": f"Bearer {Path(path).read_text().strip()}"} if path else {}


async def ask(system: str | None, image_url: str, text: str, schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    messages.append(
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": text}],
        }
    )
    body = {
        "model": VLM_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
    }
    async with state["vlm"]:
        r = await state["http"].post(f"{VLM_URL}/v1/chat/completions", json=body, headers=_auth_headers())
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


async def describe_scene(chip: Chip, kind: str) -> str:
    """One look at the whole chip. Every building's prompt carries it, because a townhome row or a
    mobile home park is obvious from above and nearly invisible in a single roof's crop."""
    h, w = chip.rgb.shape[:2]
    text = SCENE_PROMPT.format(kind=kind, gsd=GSD, w=max(h, w) * GSD)
    out = await ask(None, jpeg_data_url(chip.rgb), text, SCENE_SCHEMA, 300)
    desc = out["description"].strip()
    # The length cap can cut the last sentence off; keep whole sentences only.
    return desc[: desc.rfind(".") + 1] if "." in desc else desc


async def classify(chip: Chip, kind: str, scene: str, idx: int, s: Structure, area_m2: float) -> dict[str, Any]:
    url, width_m = crop_for(chip.rgb, s)
    text = CLASS_PROMPT.format(scene=scene, kind=kind, gsd=GSD, w=width_m, area=area_m2)
    t = time.perf_counter()
    out = await ask(CLASS_SYSTEM, url, text, SCHEMA, 120)
    return {"id": idx, **out, "ms": round((time.perf_counter() - t) * 1000)}


# -- analyze ---------------------------------------------------------------------------------------


def sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def jpeg_data_url(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def run(preset: dict[str, Any] | None, lon: float | None, lat: float | None) -> AsyncIterator[bytes]:
    pipe: Pipeline = state["pipe"]
    t0 = time.perf_counter()
    try:
        if preset is not None:
            chip, labels = await asyncio.to_thread(pipe.chip_for, preset)
            kind = preset["kind"]
        else:
            chip, labels, kind = await asyncio.to_thread(naip_chip, lon, lat), [], "aerial"
    except LookupError as e:
        yield sse("failed", {"message": str(e)})
        return
    except Exception:  # noqa: BLE001
        yield sse("failed", {"message": "The imagery service did not answer. Try again in a moment."})
        return
    fetch_ms = round((time.perf_counter() - t0) * 1000)
    h, w = chip.rgb.shape[:2]
    to_ll = chip.to_lonlat()
    corners = [to_ll.transform(*(chip.transform * (x, y))) for x, y in ((0, 0), (w, 0), (w, h), (0, h))]
    yield sse(
        "image",
        {
            "url": jpeg_data_url(chip.rgb),
            "width": w,
            "height": h,
            "gsd": GSD,
            "native_gsd": chip.native_gsd,
            "source": chip.source,
            "captured": chip.captured,
            "kind": kind,
            "corners": [[round(x, 7), round(y, 7)] for x, y in corners],
            "fetch_ms": fetch_ms,
        },
    )

    t = time.perf_counter()
    scene_task = asyncio.create_task(describe_scene(chip, kind))
    found = await asyncio.to_thread(pipe.detect, chip)
    detect_ms = round((time.perf_counter() - t) * 1000)
    items = []
    for i, s in enumerate(found):
        items.append(
            {
                "id": i,
                "kind": s.kind,
                "score": s.score,
                "px": [[x, y] for x, y in s.ring_px],
                **measure(chip, s.ring_px),
            }
        )
    to_crs = Transformer.from_crs("EPSG:4326", chip.crs, always_xy=True)
    inv = ~chip.transform
    labels_px = [[[round(v, 1) for v in inv * to_crs.transform(lo, la)] for lo, la in ring] for ring in labels]
    yield sse("structures", {"items": items, "labels": labels_px, "detect_ms": detect_ms})

    t = time.perf_counter()
    try:
        scene = await scene_task
    except Exception:  # noqa: BLE001  classify without it rather than not at all
        scene = "unknown"
    yield sse("scene", {"description": scene})
    buildings = sorted((it for it in items if it["kind"] == "building"), key=lambda it: -it["area_m2"])
    tasks = [
        asyncio.create_task(classify(chip, kind, scene, it["id"], found[it["id"]], it["area_m2"]))
        for it in buildings[:MAX_CLASSIFIED]
    ]
    failed = 0
    try:
        for fut in asyncio.as_completed(tasks):
            try:
                yield sse("class", await fut)
            except Exception:  # noqa: BLE001
                failed += 1
    finally:
        for task in tasks:
            task.cancel()
    yield sse(
        "done",
        {
            "buildings": len(buildings),
            "pools": sum(it["kind"] == "pool" for it in items),
            "classified": len(tasks) - failed,
            "fetch_ms": fetch_ms,
            "detect_ms": detect_ms,
            "classify_ms": round((time.perf_counter() - t) * 1000),
            "total_ms": round((time.perf_counter() - t0) * 1000),
        },
    )


@app.get("/api/analyze")
async def analyze(
    preset: str | None = None,
    lat: float | None = Query(None, ge=-90, le=90),
    lon: float | None = Query(None, ge=-180, le=180),
) -> StreamingResponse:
    p = next((x for x in PRESETS if x["id"] == preset), None) if preset else None
    if p is None:
        if lat is None or lon is None:
            raise HTTPException(400, "Pass a preset, or lat and lon.")
        if not (CONUS[0] <= lon <= CONUS[2] and CONUS[1] <= lat <= CONUS[3]):
            raise HTTPException(400, "Custom locations need NAIP imagery, which covers the contiguous US.")
        now = time.monotonic()
        while custom_calls and now - custom_calls[0] > 600:
            custom_calls.popleft()
        # Client IPs are not visible behind the gateway, so the budget is replica-wide.
        if len(custom_calls) >= CUSTOM_PER_10_MIN:
            raise HTTPException(429, "The demo is busy. Try a preset, or come back in a few minutes.")
        custom_calls.append(now)
    headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    return StreamingResponse(run(p, lon, lat), media_type="text/event-stream", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
