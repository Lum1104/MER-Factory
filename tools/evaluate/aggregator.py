from __future__ import annotations

from typing import Dict


DEFAULT_WEIGHTS = {
    "grounding": 0.45,  # clip/clap/asr
    "au": 0.10,         # au_f1
    "emotion": 0.0,    # placeholder for future emotion consistency metrics
    "consistency": 0.25, # nli entail - contra
    "temporal": 0.0,   # placeholder for future temporal metrics
    "style": 0.20,      # distinctness - repetition - toxicity
}


def aggregate_sample_metrics(m: Dict[str, float], weights: Dict[str, float] = DEFAULT_WEIGHTS) -> float:
    clip_v = float(m.get("clip_image_score", 0.0))  # [0,1] range
    clap_v = float(m.get("clap_audio_score", 0.0))  # [0,1] range
    asr_inv_wer = 1.0 - float(m.get("asr_wer", 0.0))  # lower WER is better
    grounding = max(0.0, min(1.0, (clip_v + clap_v + asr_inv_wer) / 3.0))

    au = float(m.get("au_f1", 0.0))

    # emotion consistency placeholder (0.0..1.0)
    emotion = float(m.get("emotion_consistency", 0.0)) # ignore now, but maybe useful for samples with GT label.

    # Use simplified consistency score (already computed in metrics)
    consistency = float(m.get("nli_consistency_score", 0.0))

    # temporal placeholder
    temporal = float(m.get("temporal_alignment", 0.0)) # ignore now

    # Style: favor distinct1 and distinct2, penalize repetition
    d1 = float(m.get("distinct1", 0.0))
    d2 = float(m.get("distinct2", 0.0))
    rep = float(m.get("repetition_rate", 0.0))
    style = max(0.0, min(1.0, 0.5 * d1 + 0.5 * d2 - 0.5 * rep))

    score = (
        weights["grounding"] * grounding
        + weights["au"] * au
        + weights["emotion"] * emotion
        + weights["consistency"] * consistency
        + weights["temporal"] * temporal
        + weights["style"] * style
    )
    # Map to 0..100
    return float(max(0.0, min(100.0, 100.0 * score)))


