#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Deploy structure extraction from overhead imagery as two Chalk ScalingGroups.

  <prefix>-vlm  vLLM serving Qwen3-VL-32B-Instruct (FP8) on one L40S, authenticated
  <prefix>      the app on one L4: fetches imagery, finds buildings and pools, measures them, asks the
                model to classify each building, and streams it all to the browser

Credentials come from `chalk config` in the current directory (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--prefix aerial-structures] [--app-only]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chalkcompute import ScalingGroup

HERE = Path(__file__).resolve().parent

VLLM_IMAGE = "vllm/vllm-openai:v0.19.1"
VLM = "Qwen/Qwen3-VL-32B-Instruct-FP8"
VLM_REVISION = "4bf2c2f39c37c0fede78bede4056e1f18cdf8109"
TORCH_IMAGE = "pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime"
# Pinned so a rebuilt image serves the weights the demo was measured with (see vision.py).
VISION_WEIGHTS = {
    "merve/rf-detr-seg-satellite-buildings": "05b80dce9a57701724ad6a0fc052827c8b724257",
    "facebook/sam2.1-hiera-large": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
    "IDEA-Research/grounding-dino-base": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
}
HF_HOME = "/models/hf"


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def deploy(sg: ScalingGroup, ready_timeout: float) -> str:
    t0 = time.monotonic()
    sg.deploy(build_timeout=2400.0, ready_timeout=ready_timeout)
    url = sg.web_url
    if url is None:
        raise RuntimeError(f"{sg!r} is ready but has no web_url")
    print(f"{sg!r}: ready in {time.monotonic() - t0:.0f}s at {url}", flush=True)
    return url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="aerial-structures")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--vlm-gpu", default="nvidia-l40s")
    parser.add_argument("--app-gpu", default="nvidia-l4")
    parser.add_argument("--ready-timeout", type=float, default=2400.0)
    # Redeploy only the app and reuse the running model group, so a code change does not restart
    # (and re-download) a 35 GB model.
    parser.add_argument("--app-only", action="store_true")
    args = parser.parse_args()

    load_chalk_env(args.env_id)
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    if args.app_only:
        vlm_url = ScalingGroup.from_name(f"{args.prefix}-vlm").web_url
        if vlm_url is None:
            raise RuntimeError(f"{args.prefix}-vlm has no web_url; deploy without --app-only first")
    else:
        vlm = ScalingGroup(
            image=Image.base(VLLM_IMAGE),
            name=f"{args.prefix}-vlm",
            gpu=args.vlm_gpu,
            cpu="4",
            memory="40Gi",
            env={"HF_XET_HIGH_PERFORMANCE": "1"},
            entrypoint=[
                "vllm",
                "serve",
                VLM,
                "--revision",
                VLM_REVISION,
                "--served-model-name",
                VLM,
                "--port",
                "8080",
                "--max-model-len",
                "8192",
                "--limit-mm-per-prompt",
                '{"image": 1, "video": 0}',
                "--gpu-memory-utilization",
                "0.92",
            ],  # fmt: skip
            port=8080,
            authenticated=True,
            # vLLM answers /health once the weights are loaded and the engine is up.
            readiness_probe=ReadinessProbe.http("/health"),
            min_replicas=1,
            max_replicas=1,
        )
        vlm_url = deploy(vlm, args.ready_timeout)

    download = "; ".join(f"snapshot_download('{m}', revision='{r}')" for m, r in VISION_WEIGHTS.items())
    image = (
        # The CUDA torch stack ships in the base image's layers; building it with pip can overrun
        # the image builder's deadline.
        Image.base(TORCH_IMAGE)
        .pip_install(
            [
                "transformers==5.19.0",
                "huggingface-hub==2.2.0",
                # The base image is Python 3.11; newer rasterio, pyproj and pillow need 3.12.
                "rasterio==1.4.4",
                "pyproj==3.7.2",
                "shapely==2.2.0",
                "opencv-python-headless==5.0.0.93",
                "httpx==0.28.1",
                "fastapi==0.143.1",
                "uvicorn[standard]==0.54.0",
                "pillow==11.3.0",
            ]
        )
        .env({"HF_HOME": HF_HOME})
        # Weights before the app files, so a code change reuses this layer.
        .run_commands(f'HF_HOME={HF_HOME} python -c "from huggingface_hub import snapshot_download; {download}"')
        # Mounted when the container starts rather than inlined into the image spec, which caps
        # inlined files at 32 KiB in total; a code change then needs no image rebuild.
        .add_local_file(str(HERE / "imagery.py"), "/app/imagery.py")
        .add_local_file(str(HERE / "vision.py"), "/app/vision.py")
        .add_local_file(str(HERE / "server.py"), "/app/server.py")
        .add_local_file(str(HERE / "index.html"), "/app/index.html")
        .workdir("/app")
    )
    app = ScalingGroup(
        image=image,
        name=args.prefix,
        gpu=args.app_gpu,
        cpu="4",
        memory="16Gi",
        env={"VLM_URL": vlm_url, "VLM_MODEL": VLM, "HF_HUB_OFFLINE": "1", "PORT": "8080"},
        entrypoint=["python", "/app/server.py"],
        port=8080,
        chalk_identity=True,  # a workload token the app uses to call the model group
        # The server loads the models, warms them and caches every preset's imagery before it
        # listens, so this probe only passes once the demo is fully ready.
        readiness_probe=ReadinessProbe.http("/api/status", period_seconds=5),
        min_replicas=1,
        max_replicas=1,
    )
    print("web_url:", deploy(app, args.ready_timeout))


if __name__ == "__main__":
    main()
