from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
import json


@dataclass
class SampleArtifactPaths:
    sample_id: str
    sample_dir: Path
    mer_json: Optional[Path]
    audio_json: Optional[Path]
    video_json: Optional[Path]
    image_json: Optional[Path]
    au_json: Optional[Path]
    au_csv: Optional[Path]
    audio_wav: Optional[Path]
    peak_frame_image: Optional[Path]


def _looks_like_mer_json(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith("_merr_data.json") or name.endswith("_mer_data.json")


def _looks_like_audio_json(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith("_audio_analysis.json") or name.endswith("_audio_data.json")


def _looks_like_video_json(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith("_video_analysis.json") or name.endswith("_video_data.json")


def _looks_like_image_json(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith("_image_analysis.json") or name.endswith("_image_data.json")


def _looks_like_au_json(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith("_au_analysis.json") or name.endswith("_au_data.json")


def find_samples(root: Path) -> Iterator[SampleArtifactPaths]:
    for sample_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        # ignore the error_logs and .llm_cache folders
        if "error_logs" in sample_dir.name or ".llm_cache" in sample_dir.name:
            continue
        # Try to discover artifacts by simple naming conventions
        mer_json = None
        audio_json = None
        video_json = None
        image_json = None
        au_json = None
        au_csv = None
        audio_wav = None
        peak_frame_image = None

        for p in sample_dir.iterdir():
            low = p.name.lower()
            if p.is_file():
                if _looks_like_mer_json(p):
                    mer_json = p
                elif _looks_like_audio_json(p):
                    audio_json = p
                elif _looks_like_video_json(p):
                    video_json = p
                elif _looks_like_image_json(p):
                    image_json = p
                elif _looks_like_au_json(p):
                    au_json = p
                elif low.endswith("_au_data.csv"):
                    au_csv = p
                elif low.endswith(".wav"):
                    audio_wav = p
                elif (low.endswith("_peak_frame.jpg") or low.endswith("_peak_frame.png")):
                    peak_frame_image = p

        # Try to get audio_wav path from JSON files if available
        # Priority: MER JSON > Audio JSON > discovered file
        for json_file in [mer_json, audio_json]:
            if json_file and json_file.exists():
                try:
                    json_data = load_mer_output(json_file)
                    if "audio_path" in json_data:
                        json_audio_path = Path(json_data["audio_path"])
                        if json_audio_path.exists():
                            audio_wav = json_audio_path
                            break  # Use the first valid path found
                except Exception:
                    # If reading JSON fails, continue to next or keep discovered audio_wav
                    continue

        sample_id = sample_dir.name
        yield SampleArtifactPaths(
            sample_id=sample_id,
            sample_dir=sample_dir,
            mer_json=mer_json,
            audio_json=audio_json,
            video_json=video_json,
            image_json=image_json,
            au_json=au_json,
            au_csv=au_csv,
            audio_wav=audio_wav,
            peak_frame_image=peak_frame_image,
        )


def load_mer_output(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


