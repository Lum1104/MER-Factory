from __future__ import annotations

from typing import Dict, Optional, Tuple
import re
import math
import pandas as pd


# Simple lexicon mapping textual mentions to AU codes.
MENTION_TO_AUS = {
    "smile": ["AU12_r"],
    "cheek raiser": ["AU06_r"],
    "inner brow raiser": ["AU01_r"],
    "outer brow raiser": ["AU02_r"],
    "brow lowerer": ["AU04_r"],
    "upper lid raiser": ["AU05_r"],
    "lid tightener": ["AU07_r"],
    "nose wrinkler": ["AU09_r"],
    "upper lip raiser": ["AU10_r"],
    "dimpler": ["AU14_r"],
    "lip corner depressor": ["AU15_r"],
    "chin raiser": ["AU17_r"],
    "lip stretcher": ["AU20_r"],
    "lip tightener": ["AU23_r"],
    "lips part": ["AU25_r"],
    "jaw drop": ["AU26_r"],
    "lip suck": ["AU28_r"],
    "blink": ["AU45_r"],
}


def _extract_mentioned_aus(text: str) -> Dict[str, float]:
    if not text:
        return {}
    text_low = text.lower()
    aus: Dict[str, float] = {}
    for phrase, codes in MENTION_TO_AUS.items():
        if phrase in text_low:
            for code in codes:
                # assign a heuristic weight 1.0 for presence
                aus[code] = max(aus.get(code, 0.0), 1.0)
    # Also capture explicit AU mentions like "AU12" or "AU12_r"
    for m in re.finditer(r"au(\d+)(?:_r|_c)?", text_low):
        code = f"AU{m.group(1)}_r"
        aus[code] = max(aus.get(code, 0.0), 1.0)
    return aus


def _presence_from_intensity_row(row: pd.Series, threshold: float) -> Dict[str, int]:
    presence = {}
    for col in row.index:
        if col.endswith("_r") and col.startswith("AU"):
            presence[col] = 1 if row[col] >= threshold else 0
    return presence


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)
    return p, r, f1


def compute_au_alignment_metrics(
    au_csv_path: Optional[str],
    peak_frame_index: Optional[int],
    peak_frame_au_text: Optional[str],
    presence_threshold: float = 0.8,
    peak_au_intensities: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compare AUs mentioned in text to actual OpenFace AUs around the peak frame.

    Returns precision/recall/f1 on AU presence.
    """
    results: Dict[str, float] = {
        "au_pr": 0.0,
        "au_re": 0.0,
        "au_f1": 0.0,
    }
    presence: Dict[str, int] = {}
    if peak_au_intensities is not None and isinstance(peak_au_intensities, dict) and peak_au_intensities:
        # Build presence directly from provided intensities
        presence = {
            k if k.endswith("_r") else f"{k}_r": int(float(v) >= presence_threshold)
            for k, v in peak_au_intensities.items()
            if str(k).upper().startswith("AU")
        }
        # If after filtering we got no AUs, return zeros
        if not presence:
            return results
    elif au_csv_path and peak_frame_index is not None:
        try:
            df = pd.read_csv(au_csv_path)
        except Exception:
            return results
        if df.empty:
            return results
        idx = max(0, min(int(peak_frame_index), len(df) - 1))
        row = df.iloc[idx]
        presence = _presence_from_intensity_row(row, presence_threshold)
        # If no valid AU columns found in CSV, return zeros
        if not presence:
            return results
    else:
        return results
    mentioned = _extract_mentioned_aus(peak_frame_au_text or "")

    # Build sets
    predicted_set = {k for k, v in presence.items() if v == 1}
    mentioned_set = set(mentioned.keys())
    
    # If both sets are empty, return zeros (no data to evaluate)
    if not predicted_set and not mentioned_set:
        return results

    tp = len(predicted_set & mentioned_set)
    fp = len(mentioned_set - predicted_set)
    fn = len(predicted_set - mentioned_set)

    p, r, f1 = _precision_recall_f1(tp, fp, fn)
    results.update({"au_pr": p, "au_re": r, "au_f1": f1})
    return results


