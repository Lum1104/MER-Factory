from __future__ import annotations

from typing import Dict, Optional
import math
import re


def _distinct_ngram_ratio(text: str, n: int) -> float:
    if not text:
        return 0.0
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    unique = len(set(ngrams))
    total = len(ngrams)
    return unique / total if total else 0.0


def _repetition_rate(text: str) -> float:
    if not text:
        return 0.0
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    return 1.0 - unique / len(tokens)


def _fkgl(text: str) -> float:
    # Simple FKGL approximation without external deps
    if not text:
        return 0.0
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r"\w+", text)
    syllables = sum(_count_syllables(w) for w in words)
    num_sent = max(1, len(sentences))
    num_words = max(1, len(words))
    fkgl = 0.39 * (num_words / num_sent) + 11.8 * (syllables / num_words) - 15.59
    return float(fkgl)


def _count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev:
            count += 1
        prev = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def compute_text_style_metrics(text: Optional[str]) -> Dict[str, float]:
    text = text or ""
    return {
        "distinct1": _distinct_ngram_ratio(text, 1),
        "distinct2": _distinct_ngram_ratio(text, 2),
        "repetition_rate": _repetition_rate(text),
        "fkgl": _fkgl(text),
    }


