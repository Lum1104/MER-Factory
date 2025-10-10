# MER-Factory Evaluation Suite

A comprehensive reference-free evaluation toolkit for MER-Factory outputs. This suite provides automated metrics to assess annotation quality without human ratings, supporting MER (full), audio, video, image, and AU analysis pipelines with graceful degradation when artifacts or dependencies are missing.

## Overview

### Why automated, reference‑free evaluation?
- **Scalability**: Evaluate large datasets without human raters
- **Objectivity**: Measure grounding, consistency, and structure with reproducible metrics
- **Debuggability**: Identify failure modes (hallucination, poor grounding, weak AU alignment) quickly
- **Model‑agnostic**: Works across providers (Gemini, ChatGPT, Ollama, HuggingFace) and pipeline types
- **Real-time feedback**: Beautiful progress bars and color-coded results for immediate insights

## Quick Start

### Basic Usage
```bash
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv
```

### With Verbose Output
```bash
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv --verbose
```

### Advanced Options
```bash
python tools/evaluate.py output/ \
    --export-csv output/evaluation_summary.csv \
    --write-per-sample \
    --verbose \
    --batch-size 16
```

**Performance**: Batch processing provides 3-8x speedup on GPU compared to single-sample mode. All evaluation functions (CLIP, CLAP, NLI, ASR) automatically detect and handle batch inputs.

## Supported Pipeline Types

The evaluation system automatically detects and adapts to different pipeline types:

| Pipeline Type | Detection Logic | Applicable Metrics |
|---------------|----------------|-------------------|
| **MER (full)** | `*_merr_data.json` present | All metrics (CLIP, CLAP, AU, NLI, ASR, Style) |
| **Audio** | `*_audio_analysis.json` present | CLAP, ASR WER, NLI, Style |
| **Video** | `*_video_analysis.json` present | CLIP, NLI, Style |
| **Image** | `*_image_analysis.json` present | CLIP, NLI, Style |
| **AU** | `*_au_analysis.json` present | AU alignment, Style |

## Metric Categories

### 🖼️ Grounding Metrics (Media ↔ Text)

#### CLIP Image-Text Score (`clip_image_score`)
- **Purpose**: Measures visual grounding between peak frame and text description
- **Method**: Cosine similarity via OpenAI CLIP (ViT-B-32)
- **Range**: 0-1 (normalized from -1 to 1)
- **Interpretation**: Higher scores indicate better visual alignment
- **Dependencies**: `open_clip_torch`, `torch`, `PIL`
- **Fallback**: Returns 0.0 if dependencies missing or no image available

#### CLAP Audio-Text Score (`clap_audio_score`)  
- **Purpose**: Measures audio-text alignment between WAV file and transcript/description
- **Method**: Cosine similarity via LAION-CLAP (HTSAT-base)
- **Range**: 0-1 (normalized from -1 to 1)
- **Interpretation**: Higher scores indicate better audio grounding
- **Dependencies**: `laion-clap`, `torch`
- **Fallback**: Returns 0.0 if dependencies missing or no audio available

#### ASR Word Error Rate (`asr_wer`)
- **Purpose**: Validates transcript quality against strong ASR baseline
- **Method**: WER comparison with Whisper/faster-whisper transcription
- **Range**: 0-1 (lower is better)
- **Interpretation**: Lower WER indicates more accurate transcription
- **Dependencies**: `faster-whisper` (preferred) or `whisper`
- **Special Features**: 
  - Chinese language support with character-level tokenization
  - Traditional to Simplified Chinese normalization
  - Intelligent fallback between ASR engines

### 😊 AU Alignment Metrics (Facial Expression ↔ Text)

#### AU Precision/Recall/F1 (`au_pr`, `au_re`, `au_f1`)
- **Purpose**: Matches textual AU descriptions to detected facial actions
- **Method**: Lexicon-based extraction + OpenFace AU intensity comparison
- **Range**: 0-1 (higher is better)
- **AU Lexicon**: Maps phrases like "smile" → AU12_r, "brow raiser" → AU01_r
- **Threshold**: Default 0.8 for AU presence detection
- **Data Sources**: 
  - OpenFace CSV files (`*_au_data.csv`)
  - Peak frame AU intensities from JSON
- **Fallback**: Returns 0.0 if no AU data or text available

### 🔗 Cross-Modal Consistency

#### NLI Consistency Score (`nli_consistency_score`, `nli_entail_rate`, `nli_contra_rate`)
- **Purpose**: Measures logical consistency between multimodal summary and unimodal descriptions
- **Method**: Natural Language Inference via DeBERTa-large-MNLI
- **Logic**: Entailment rate - Contradiction rate (only positive values kept)
- **Range**: 0-1 (higher is better)
- **Dependencies**: `transformers`, `torch`
- **Fallback**: Returns 0.0 if dependencies missing or insufficient text

### 📝 Text Quality & Style

#### Distinct N-grams (`distinct1`, `distinct2`)
- **Purpose**: Measures lexical diversity in generated text
- **Method**: Ratio of unique unigrams/bigrams to total
- **Range**: 0-1 (higher indicates more diversity)
- **Interpretation**: Low values may indicate repetitive generation

#### Repetition Rate (`repetition_rate`)
- **Purpose**: Detects overly repetitive text generation
- **Method**: 1 - (unique tokens / total tokens)
- **Range**: 0-1 (lower is better)
- **Interpretation**: Higher values indicate more repetition

#### Readability Score (`fkgl`)
- **Purpose**: Measures text complexity via Flesch-Kincaid Grade Level
- **Method**: Approximation based on sentence/word/syllable counts
- **Interpretation**: Extremely high/low values may indicate poor generation quality

### 🎯 Composite Score

The overall quality score (0-100) combines all metrics with carefully tuned weights:

```python
DEFAULT_WEIGHTS = {
    "grounding": 0.45,  # clip/clap/asr
    "au": 0.10,         # au_f1
    "emotion": 0.0,    # placeholder for future emotion consistency metrics
    "consistency": 0.25, # nli entail - contra
    "temporal": 0.0,   # placeholder for future temporal metrics
    "style": 0.20,      # distinctness - repetition - toxicity
}
```

**Score Interpretation**:
- 🎉 **80-100**: Excellent quality
- 👍 **60-79**: Good quality  
- 📈 **0-59**: Needs improvement

## File Structure & Data Loading

### Expected Directory Structure
```
output/
├── sample_00000000/
│   ├── sample_00000000_merr_data.json     # MER pipeline output
│   ├── sample_00000000_peak_frame.png     # Peak frame image
│   ├── sample_00000000.wav               # Audio file
│   ├── sample_00000000_au_data.csv       # OpenFace AU data
│   └── evaluation.json                   # Generated metrics
├── sample_00000001/
│   └── ...
└── evaluation_summary.csv                # Dataset-level results
```

### Artifact Discovery Logic

The `SampleArtifactPaths` class automatically discovers relevant files:

| File Pattern | Purpose | Priority |
|-------------|---------|----------|
| `*_merr_data.json` | MER pipeline output | Highest |
| `*_audio_analysis.json` | Audio-only pipeline | Medium |
| `*_video_analysis.json` | Video-only pipeline | Medium |
| `*_image_analysis.json` | Image-only pipeline | Medium |
| `*_au_analysis.json` | AU-only pipeline | Low |
| `*_peak_frame.{png,jpg}` | Peak frame image | - |
| `*.wav` | Audio file | - |
| `*_au_data.csv` | OpenFace AU data | - |

## Output Formats

### Per-Sample Output (`evaluation.json`)
```json
{
  "clip_image_score": 0.785,
  "clap_audio_score": 0.692,
  "asr_wer": 0.123,
  "au_pr": 0.834,
  "au_re": 0.756,
  "au_f1": 0.793,
  "nli_consistency_score": 0.667,
  "nli_entail_rate": 0.750,
  "nli_contra_rate": 0.083,
  "distinct1": 0.923,
  "distinct2": 0.887,
  "repetition_rate": 0.076,
  "fkgl": 8.2,
  "composite_score": 73.4
}
```

### Dataset Summary (`evaluation_summary.csv`)
- Sorted by `composite_score` (descending)
- All metrics included for statistical analysis
- Easy import into analysis tools

### Console Output
- 🏆 Top 10 performing samples table
- 📈 Overall statistics panel
- 📚 Metric explanations table
- Color-coded values (green/yellow/red thresholds)

## Advanced Configuration

### Custom Weights
Modify the aggregation weights in `aggregator.py`:
```python
custom_weights = {
    "grounding": 0.30,    # Emphasize grounding
    "au": 0.25,           # Increase AU importance  
    "consistency": 0.20,  # More consistency weight
    "style": 0.15,        # Higher style weight
    "emotion": 0.10,      # Reserve for future
    "temporal": 0.00,     # Disable temporal
}
```

### AU Presence Threshold
Adjust AU detection sensitivity:
```python
compute_au_alignment_metrics(
    au_csv_path=csv_path,
    peak_frame_index=frame_idx,
    peak_frame_au_text=text,
    presence_threshold=0.9  # Stricter threshold
)
```
