"""Find structures in a 0.3 m chip and turn their masks into measured polygons.

- Buildings: an RF-DETR instance segmentation model fine-tuned on overhead building footprints,
  run on the whole chip and on overlapping native-resolution tiles, then merged.
- Pools: Grounding DINO with the prompt "swimming pool".
- Every box from either detector is re-segmented by SAM 2.1, whose masks follow roof edges more
  closely, and the mask is traced, simplified and squared up into a polygon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from shapely.affinity import rotate
from shapely.geometry import Polygon
from transformers import (
    AutoImageProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    RfDetrForInstanceSegmentation,
    Sam2Model,
    Sam2Processor,
)

from imagery import GSD, Chip

BUILDING_MODEL = ("merve/rf-detr-seg-satellite-buildings", "05b80dce9a57701724ad6a0fc052827c8b724257")
SAM_MODEL = ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610")
POOL_MODEL = ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb")
BUILDING_THRESHOLD = 0.5
POOL_THRESHOLD = 0.15  # low on purpose: a water-colour check on the mask does the filtering
TILE, OVERLAP = 432, 96
MIN_BUILDING_M2 = 12.0


@dataclass
class Structure:
    kind: str  # "building" | "pool"
    score: float
    ring_px: list[tuple[float, float]]  # regularized outline, chip pixels
    poly_px: Polygon


class Models:
    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        name, rev = BUILDING_MODEL
        self.rf_proc = AutoImageProcessor.from_pretrained(name, revision=rev)
        self.rf = RfDetrForInstanceSegmentation.from_pretrained(name, revision=rev).to(device).eval()
        name, rev = SAM_MODEL
        self.sam_proc = Sam2Processor.from_pretrained(name, revision=rev)
        self.sam = Sam2Model.from_pretrained(name, revision=rev, torch_dtype=torch.bfloat16).to(device).eval()
        name, rev = POOL_MODEL
        self.gd_proc = AutoProcessor.from_pretrained(name, revision=rev)
        self.gd = AutoModelForZeroShotObjectDetection.from_pretrained(name, revision=rev).to(device).eval()

    # -- buildings ---------------------------------------------------------------------------

    @torch.inference_mode()
    def _rfdetr(self, img: Image.Image) -> list[tuple[np.ndarray, float]]:
        inputs = self.rf_proc(images=img, return_tensors="pt").to(self.device)
        out = self.rf(**inputs)
        res = self.rf_proc.post_process_instance_segmentation(
            out, threshold=BUILDING_THRESHOLD, target_sizes=[img.size[::-1]]
        )[0]
        seg = res["segmentation"].cpu().numpy()
        return [(seg == s["id"], float(s["score"])) for s in res["segments_info"]]

    def _tiled(self, img: Image.Image) -> list[tuple[np.ndarray, float]]:
        w, h = img.size
        step = TILE - OVERLAP
        xs = sorted({*range(0, max(w - TILE, 0) + 1, step), max(w - TILE, 0)})
        ys = sorted({*range(0, max(h - TILE, 0) + 1, step), max(h - TILE, 0)})
        dets = []
        for y in ys:
            for x in xs:
                for m, s in self._rfdetr(img.crop((x, y, x + TILE, y + TILE))):
                    yy, xx = np.nonzero(m)
                    if len(xx) == 0:
                        continue
                    # An instance touching an interior tile edge is cut off; the overlapping
                    # neighbour (or the whole-chip pass) sees it whole.
                    if (x > 0 and xx.min() < 2) or (y > 0 and yy.min() < 2):
                        continue
                    if (x + TILE < w and xx.max() >= TILE - 2) or (y + TILE < h and yy.max() >= TILE - 2):
                        continue
                    full = np.zeros((h, w), bool)
                    full[y : y + TILE, x : x + TILE] = m[: min(TILE, h - y), : min(TILE, w - x)]
                    dets.append((full, s))
        return dets

    def buildings(self, img: Image.Image) -> list[tuple[np.ndarray, float]]:
        # The whole-chip pass sees large roofs in one piece; the native-resolution tiles separate
        # neighbouring houses that the downscaled pass merges. Keep the best of both.
        return mask_nms(self._rfdetr(img) + self._tiled(img), 0.5)

    # -- pools -------------------------------------------------------------------------------

    @torch.inference_mode()
    def pools(self, img: Image.Image) -> list[tuple[list[float], float]]:
        inputs = self.gd_proc(images=img, text="swimming pool.", return_tensors="pt").to(self.device)
        out = self.gd(**inputs)
        res = self.gd_proc.post_process_grounded_object_detection(
            out,
            inputs.input_ids,
            threshold=POOL_THRESHOLD,
            text_threshold=POOL_THRESHOLD,
            target_sizes=[img.size[::-1]],
        )[0]
        boxes = []
        for b, s in zip(res["boxes"].tolist(), res["scores"].tolist()):
            area_m2 = (b[2] - b[0]) * (b[3] - b[1]) * GSD * GSD
            if 6 <= area_m2 <= 400:  # residential to community pools; drops whole-yard boxes
                boxes.append((b, float(s)))
        return boxes

    # -- refinement --------------------------------------------------------------------------

    @torch.inference_mode()
    def sam_masks(self, img: Image.Image, boxes: list[list[float]]) -> list[np.ndarray]:
        if not boxes:
            return []
        out_masks: list[np.ndarray] = []
        for i in range(0, len(boxes), 64):
            inputs = self.sam_proc(images=img, input_boxes=[boxes[i : i + 64]], return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            out = self.sam(**inputs, multimask_output=False)
            ms = self.sam_proc.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"].cpu())[0]
            out_masks.extend(ms[j, 0].numpy() > 0 for j in range(ms.shape[0]))
        return out_masks

    def detect(self, chip: Chip) -> list[Structure]:
        img = Image.fromarray(chip.rgb)
        bmasks = self.buildings(img)
        refined = self.sam_masks(img, [mask_box(m, pad=3) for m, _ in bmasks])
        found: list[Structure] = []
        for (src, score), mask in zip(bmasks, refined):
            # SAM sometimes grabs the whole lot or a neighbour; keep the detector's own mask
            # when the refined one strays far from it.
            if iou(mask, src) < 0.5:
                mask = src
            poly = mask_to_polygon(mask)
            if poly is None or poly.area * GSD * GSD < MIN_BUILDING_M2:
                continue
            # The building model sometimes outlines a pool; the pool pass below finds it properly.
            if water_share(chip.rgb, mask) > 0.5:
                continue
            ring = regularize(poly, square=True)
            found.append(Structure(kind="building", score=round(score, 3), ring_px=ring, poly_px=Polygon(ring)))
        buildings = [s.poly_px for s in found]
        pools: list[Structure] = []
        for box, score in sorted(self.pools(img), key=lambda d: -d[1]):
            # The pool's outline is its water: SAM's masks for pools tend to take the deck too.
            mask = water_mask(chip.rgb, box)
            poly = mask_to_polygon(mask) if mask is not None else None
            if poly is None:
                continue
            ring = regularize(poly, square=False)
            p = Polygon(ring)
            if not p.is_valid or p.area * GSD * GSD < 6:
                continue
            # Blue roofs (tarps, enclosures) and duplicate boxes are not new pools.
            if any(p.intersection(b).area > 0.3 * p.area for b in buildings):
                continue
            if any(p.intersection(q.poly_px).area > 0.3 * min(p.area, q.poly_px.area) for q in pools):
                continue
            pools.append(Structure(kind="pool", score=round(score, 3), ring_px=ring, poly_px=p))
        return found + pools


# -- geometry ------------------------------------------------------------------------------------


def is_water(rgb: np.ndarray, strict: bool) -> np.ndarray:
    """Pool-water colour per pixel.

    The loose test (cyan to blue, a little saturation) traces a pool inside a box the pool detector
    already proposed. The strict test decides whether a *building* is really a pool without that
    prior: pools sit at hue 160-200 with saturation mostly above 75, while slate roofs are bluer
    (hue 205-225) and metal roofs and grain bins much greyer (saturation 30-50).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(int)
    hue, sat, val = hsv[..., 0] * 2, hsv[..., 1], hsv[..., 2]  # OpenCV hue is degrees / 2
    if strict:
        return (hue >= 160) & (hue <= 205) & (sat >= 70) & (val >= 90)
    return (hue >= 150) & (hue <= 235) & (sat >= 30) & (val >= 50)


def water_share(rgb: np.ndarray, mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return float(is_water(rgb[y0:y1, x0:x1], strict=True)[mask[y0:y1, x0:x1]].mean())


def water_mask(rgb: np.ndarray, box: list[float]) -> np.ndarray | None:
    """The largest pool-water-coloured blob in a box, or None."""
    h, w = rgb.shape[:2]
    x0, y0 = max(int(box[0]) - 2, 0), max(int(box[1]) - 2, 0)
    x1, y1 = min(int(box[2]) + 3, w), min(int(box[3]) + 3, h)
    water = is_water(rgb[y0:y1, x0:x1], strict=False).astype(np.uint8)
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(water)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < 0.2 * (box[2] - box[0]) * (box[3] - box[1]):
        return None
    out = np.zeros((h, w), bool)
    out[y0:y1, x0:x1] = labels == best
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def mask_nms(dets: list[tuple[np.ndarray, float]], thr: float) -> list[tuple[np.ndarray, float]]:
    """Greedy NMS on masks by intersection over the smaller mask; compares overlapping boxes only."""
    keep: list[tuple[np.ndarray, float, tuple[int, int, int, int], int]] = []
    for m, s in sorted(dets, key=lambda d: -d[1]):
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        area = len(xs)
        ok = True
        for k, _, kb, ka in keep:
            x0, y0, x1, y1 = max(box[0], kb[0]), max(box[1], kb[1]), min(box[2], kb[2]), min(box[3], kb[3])
            if x0 >= x1 or y0 >= y1:
                continue
            inter = int((m[y0:y1, x0:x1] & k[y0:y1, x0:x1]).sum())
            if inter / max(min(area, ka), 1) > thr:
                ok = False
                break
        if ok:
            keep.append((m, s, box, area))
    return [(m, s) for m, s, _, _ in keep]


def mask_box(m: np.ndarray, pad: int) -> list[float]:
    ys, xs = np.nonzero(m)
    return [float(xs.min() - pad), float(ys.min() - pad), float(xs.max() + pad), float(ys.max() + pad)]


def mask_to_polygon(m: np.ndarray) -> Polygon | None:
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 30 or len(c) < 4:
        return None
    p = Polygon(c[:, 0, :].astype(float)).buffer(0)
    if p.geom_type == "MultiPolygon":
        p = max(p.geoms, key=lambda g: g.area)
    return p if p.is_valid and not p.is_empty else None


def dominant_angle(p: Polygon) -> float:
    """Direction (radians, mod pi/2) of the polygon's long edges, from its minimum rotated rectangle."""
    rect = list(p.minimum_rotated_rectangle.exterior.coords)
    (x0, y0), (x1, y1), (x2, y2) = rect[0], rect[1], rect[2]
    if math.hypot(x1 - x0, y1 - y0) >= math.hypot(x2 - x1, y2 - y1):
        return math.atan2(y1 - y0, x1 - x0)
    return math.atan2(y2 - y1, x2 - x1)


def regularize(p: Polygon, square: bool) -> list[tuple[float, float]]:
    """Simplify an outline and, for buildings, snap edges near the dominant axes to exactly 90°."""
    simple = p.simplify(1.5, preserve_topology=True)
    if not square or simple.geom_type != "Polygon" or len(simple.exterior.coords) < 5:
        return [(round(x, 1), round(y, 1)) for x, y in simple.exterior.coords]
    theta = dominant_angle(simple)
    # Work in a frame where the dominant axes are x and y.
    r = rotate(simple, -theta, origin=simple.centroid, use_radians=True)
    pts = list(r.exterior.coords)[:-1]
    lines: list[tuple[str, float, float]] = []  # (axis, offset, length) or ("free", ...)
    for a, b in zip(pts, pts[1:] + pts[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        ang = abs(math.degrees(math.atan2(dy, dx))) % 180
        if ang < 20 or ang > 160:
            lines.append(("h", (a[1] + b[1]) / 2, length))
        elif 70 < ang < 110:
            lines.append(("v", (a[0] + b[0]) / 2, length))
        elif length < 8:
            continue  # a chamfered corner from the tracer; squaring removes it
        else:
            lines.append(("free", 0.0, length))
    if any(kind == "free" for kind, _, _ in lines):
        # Diagonal walls (bay windows, angled wings): keep the plain simplification.
        return [(round(x, 1), round(y, 1)) for x, y in simple.exterior.coords]
    # Merge runs of parallel edges (a short jog between them becomes one wall at their weighted offset).
    merged: list[list[float | str]] = []
    for kind, off, length in lines:
        if merged and merged[-1][0] == kind:
            k, o, ln = merged[-1]
            merged[-1] = [k, (o * ln + off * length) / (ln + length), ln + length]
        else:
            merged.append([kind, off, length])
    if len(merged) > 1 and merged[0][0] == merged[-1][0]:
        k, o, ln = merged.pop()
        k0, o0, ln0 = merged[0]
        merged[0] = [k0, (o0 * ln0 + o * ln) / (ln0 + ln), ln0 + ln]
    if len(merged) < 4 or len(merged) % 2:
        return [(round(x, 1), round(y, 1)) for x, y in simple.exterior.coords]
    corners = []
    for (k1, o1, _), (_, o2, _) in zip(merged, merged[1:] + merged[:1]):
        corners.append((o2, o1) if k1 == "h" else (o1, o2))
    sq = Polygon(corners)
    if not sq.is_valid or abs(sq.area - simple.area) > 0.2 * simple.area:
        return [(round(x, 1), round(y, 1)) for x, y in simple.exterior.coords]
    back = rotate(sq, theta, origin=simple.centroid, use_radians=True)
    return [(round(x, 1), round(y, 1)) for x, y in back.exterior.coords]


def measure(chip: Chip, ring_px: list[tuple[float, float]]) -> dict:
    """Pixel outline -> lon/lat ring plus metric measurements (the chip grid is metric UTM)."""
    to_ll = chip.to_lonlat()
    xy = [chip.transform * (x, y) for x, y in ring_px]
    ll = [to_ll.transform(x, y) for x, y in xy]
    poly_m = Polygon(xy)
    rect = poly_m.minimum_rotated_rectangle
    rc = list(rect.exterior.coords)
    e1 = math.hypot(rc[1][0] - rc[0][0], rc[1][1] - rc[0][1])
    e2 = math.hypot(rc[2][0] - rc[1][0], rc[2][1] - rc[1][1])
    long_edge = (rc[0], rc[1]) if e1 >= e2 else (rc[1], rc[2])
    dx, dy = long_edge[1][0] - long_edge[0][0], long_edge[1][1] - long_edge[0][1]
    bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 180  # long axis, clockwise from grid north
    c = poly_m.centroid
    clon, clat = to_ll.transform(c.x, c.y)
    return {
        "ring": [[round(x, 7), round(y, 7)] for x, y in ll],
        "centroid": [round(clon, 7), round(clat, 7)],
        "area_m2": round(poly_m.area, 1),
        "perimeter_m": round(poly_m.length, 1),
        "length_m": round(max(e1, e2), 1),
        "width_m": round(min(e1, e2), 1),
        "orientation_deg": round(bearing, 1),
        "rectangularity": round(poly_m.area / max(rect.area, 1e-6), 2),
        "vertices": len(ring_px) - 1,
    }
