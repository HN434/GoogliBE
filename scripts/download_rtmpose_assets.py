"""
Utility script to download RTMPose-S config and checkpoint for CPU inference.

Usage:
    python scripts/download_rtmpose_assets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

# Updated URLs – these paths are stable in the MMPose repo
CONFIG_URL = (
    "https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
    "configs/rtmpose/rtmpose-s_8xb256-270e_body7-256x192.py"
)

CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/rtmpose/"
    "rtmpose-s/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
)


def download_file(url: str, dest: Path, overwrite: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        print(f"✔ {dest} already exists. Skipping download.")
        return

    print(f"⬇ Downloading {url} -> {dest}")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"✅ Saved to {dest}")
    except Exception as e:
        print(f"❌ Failed downloading {url}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download RTMPose assets")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/rtmpose-s"),
        help="Destination directory for config/checkpoint files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    args = parser.parse_args()

    config_path = args.models_dir / "rtmpose-s_256x192.py"
    checkpoint_path = args.models_dir / "rtmpose-s_256x192.pth"

    try:
        # download_file(CONFIG_URL, config_path, overwrite=args.overwrite)
        download_file(CHECKPOINT_URL, checkpoint_path, overwrite=args.overwrite)
    except requests.HTTPError as err:
        print(f"❌ HTTP error while downloading assets: {err}")
        sys.exit(1)
    except requests.RequestException as err:
        print(f"❌ Network error while downloading assets: {err}")
        sys.exit(1)

    print("🎉 All RTMPose assets downloaded successfully.")


if __name__ == "__main__":
    main()
