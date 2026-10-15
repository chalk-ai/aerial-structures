# Structures from aerial and satellite imagery

Pick a point. The server pulls the newest overhead imagery for it, outlines every building and swimming
pool, measures each outline in meters, and asks a vision-language model what each building is. Results
stream to the browser as they are ready and download as GeoJSON.

It runs as two [Chalk](https://chalk.ai) ScalingGroups deployed from one script: a vision app on an
NVIDIA L4 and Qwen3-VL-32B on an NVIDIA L40S.

Live: [chalk.ai/demos/aerial-structures](https://chalk.ai/demos/aerial-structures)

| File | What it is |
| --- | --- |
| `deploy.py` | Deploys both ScalingGroups and prints the app's URL |
| `server.py` | FastAPI app: presets, the streaming `/api/analyze` endpoint, and the classification prompts |
| `vision.py` | Building and pool detection, SAM 2.1 refinement, outline tracing, squaring and measuring |
| `imagery.py` | Reads NAIP aerial imagery and SpaceNet satellite chips onto one 0.3 m UTM grid |
| `index.html` | The page the app serves (`?embed=1` for the iframe on chalk.ai) |
| `example.py` | A client that streams one tile and optionally writes GeoJSON |

## Run it

```bash
chalk login                     # or export CHALK_CLIENT_ID / CHALK_CLIENT_SECRET
./deploy.py                     # both groups; prints web_url
./deploy.py --app-only          # redeploy the app only, keep the running model group
./example.py https://<web_url> --preset scottsdale
./example.py https://<web_url> --lat 33.615 --lon -111.889 --geojson tile.geojson
```

A farmstead in Polk County, Iowa, against the live demo:

```text
$ ./example.py https://<web_url> --lat 41.7314 --lon -93.50835
imagery: USDA NAIP 2023, 2023-09-02, 800x800 px (928 ms)
outlines: 10 structures (685 ms)
classified 10 buildings (4086 ms); total 5707 ms
{"agricultural": 8, "single_family_house": 1, "accessory": 1}
{"id": 1, "kind": "building", "score": 0.922, "centroid": [-93.5092498, 41.7311168],
 "area_m2": 396.2, "perimeter_m": 81.8, "length_m": 25.2, "width_m": 15.7,
 "orientation_deg": 90.8, "rectangularity": 1.0, "vertices": 4,
 "roof_color": "white", "structure_type": "agricultural", "roof_shape": "gable", "ms": 1582}
```

## How it works

1. **Imagery.** For a point in the contiguous US, `imagery.py` finds the newest USDA NAIP acquisition
   in Microsoft Planetary Computer's STAC catalog, reads a 240 m window straight out of the
   cloud-optimized GeoTIFFs (mosaicking across tile edges), and resamples it to 0.3 m in the local
   UTM zone. NAIP is 0.3 m in many states since 2021 and 0.6 m elsewhere. The satellite presets are
   SpaceNet 2 WorldView-3 chips, reprojected onto the same grid, with their hand-drawn footprints.
2. **Buildings.** An RF-DETR instance segmentation model fine-tuned on overhead buildings runs on the
   whole chip and on overlapping native-resolution tiles; the two passes are merged with mask NMS.
   Each detection's box goes to SAM 2.1, whose masks follow roof edges more closely. A mask that is
   mostly pool-coloured water is dropped (the pool pass handles it).
3. **Pools.** Grounding DINO proposes boxes for "swimming pool" at a low threshold; the pool's outline
   is the largest water-coloured blob inside the box, which rejects most false boxes and keeps the deck
   out of the outline.
4. **Geometry.** Masks are traced, simplified, and building outlines are squared up: edges within 20°
   of the footprint's dominant axes snap to it. Every outline is measured in UTM: area, perimeter,
   length and width of the minimum rotated rectangle, the long axis's bearing, rectangularity and
   centroid, and returned as a lon/lat ring.
5. **Classification.** Qwen3-VL-32B first describes the whole tile in two sentences. Each building is
   then sent as a crop with its lot and neighbours, outlined in magenta, together with that
   description, and answers in a JSON schema: `structure_type` (single_family_house, multi_family,
   manufactured_home, commercial, industrial, institutional, agricultural, accessory), `roof_shape`
   and `roof_color`. Up to 160 buildings per tile are classified, largest first.

## API

`GET /api/analyze?preset=<id>` or `GET /api/analyze?lat=<lat>&lon=<lon>` returns Server-Sent Events:

| Event | Data |
| --- | --- |
| `image` | the chip as a JPEG data URL, size, source, capture date, corner coordinates, `fetch_ms` |
| `structures` | `items`: `id`, `kind` (`building` or `pool`), `px` (outline in image pixels), `ring` (lon/lat), `area_m2`, `perimeter_m`, `length_m`, `width_m`, `orientation_deg`, `rectangularity`, `centroid`; `labels` (SpaceNet only); `detect_ms` |
| `scene` | `description`: the model's read of the whole tile |
| `class` | one per building: `id`, `structure_type`, `roof_shape`, `roof_color`, `ms` |
| `done` | counts and `fetch_ms`, `detect_ms`, `classify_ms`, `total_ms` |
| `failed` | `message` |

`GET /api/presets` lists the presets; `GET /api/status` is the readiness probe.

## Measured

All on the deployed demo (one NVIDIA L4 for the app, one NVIDIA L40S for the model) or against labels
we did not tune on.

**Footprints.** F1 at IoU 0.5 (the SpaceNet metric) for the full building pipeline, on 40 random
SpaceNet 2 chips per city (preset chips excluded):

| City | Labelled buildings | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Las Vegas | 1,335 | 0.86 | 0.77 | 0.81 |
| Paris | 515 | 0.67 | 0.62 | 0.64 |
| Shanghai | 418 | 0.51 | 0.43 | 0.47 |
| Khartoum | 1,057 | 0.41 | 0.36 | 0.38 |

The fine-tuning data of the building model is not documented, so it may overlap SpaceNet.

**Building types.** 112 of 118 buildings (95%) correct on a held-out set we labelled by hand, without
seeing the model's answers, from seven presets (Scottsdale, the Mesa mobile home park, Fridley,
the farmstead, Las Vegas, Paris, Shanghai); buildings too ambiguous to call at 0.3 m were left out.
On the set the prompts were developed against (Cape Coral, Mesa townhomes, Des Moines, Elk Grove;
dominated by hard attached-townhome and mobile-home cases) it is 88 of 101 (87%). Before the
whole-tile description and wider crops were added, Qwen3-VL-8B scored about 50% on that set.
Roof colour comes back from the same call but was not scored, and can differ between runs because
batched inference is not bit-for-bit deterministic.

**Latency**, three passes over every preset:

| Stage | Time |
| --- | --- |
| Imagery for a new location (NAIP via Planetary Computer) | 2.6 to 3.5 s |
| Outlines and measurements on the L4 | 0.6 s (farmstead) to 2.8 s (156 mobile homes) |
| Classification on the L40S, scene plus every building | 4.5 s (6 buildings) to 21 s (156 buildings); about 10 s for 50 |

## Notes

- **Where it is weak.** The footprint model was fine-tuned on satellite imagery of a style close to
  US suburbs; it misses more buildings in dense or non-US settings, under heavy tree canopy, and on
  off-nadir imagery. Attached townhomes and detached houses can look alike at 0.3 m, and so can small
  manufactured homes and sheds. Roof material and rooftop solar were tried and dropped: the model was
  not reliable enough at this resolution.
- **Abuse limits.** The demo is unauthenticated, so it bounds load per replica: custom locations are
  limited to 60 per 10 minutes and to the contiguous US, one tile is processed on the GPU at a time,
  and at most 32 model calls are in flight. Client IPs are not visible behind the gateway.
- **Image builds.** The app image starts from `pytorch/pytorch` so the CUDA stack comes from base
  layers; building it with `pip` can overrun the image builder's deadline. The vision weights are
  baked in at pinned revisions; Qwen3-VL is downloaded by vLLM at startup.
- **Cleaning up.** With the same credentials:
  `python -c "from chalkcompute import ScalingGroup; [ScalingGroup.from_name(n).delete() for n in ('aerial-structures', 'aerial-structures-vlm')]"`

## Credits and licenses

- Imagery: [USDA NAIP](https://naip-usdaonline.hub.arcgis.com/) via
  [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/naip), public domain.
  [SpaceNet](https://spacenet.ai/) 2 imagery and labels from Maxar WorldView-3, CC BY-SA 4.0.
- Models: [merve/rf-detr-seg-satellite-buildings](https://huggingface.co/merve/rf-detr-seg-satellite-buildings)
  (an RF-DETR segmentation model by Roboflow, fine-tuned by Merve Noyan),
  [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-large) by Meta,
  [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) by IDEA Research, and
  [Qwen3-VL-32B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) by the Qwen team.
  All Apache-2.0.
- This code: Apache-2.0, see `LICENSE`.
