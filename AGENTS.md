# AGENS.md - Agent Guide for MER-Factory

> Where to modify code when adding features to MER-Factory

## Overview

MER-Factory is a Python toolkit for Multimodal Emotion Recognition and Reasoning (MERR) dataset creation. It uses LangGraph for pipeline orchestration and supports multiple LLM providers (Gemini, ChatGPT, Ollama, HuggingFace).

**Key features**: Automated preprocessing, LLM-based annotation, reference-free evaluation (CLIP/CLAP/AU F1/NLI/ASR), Gate Agent for quality control (dev), and training pipeline integration.

**Tech stack**: Python 3.12+, LangChain/LangGraph, PyTorch, Transformers, Typer CLI, Rich.

## Project Architecture

```
Entry Point (main.py)
    ↓
Config (utils/config.py) → ProcessingType (MER/AU/audio/video/image)
    ↓
Graph (mer_factory/graph.py) → StateGraph with conditional routing
    ↓
Nodes (mer_factory/nodes/) → Pipeline stages (async/sync)
    ↓
Models (mer_factory/models/) → LLM providers (Gemini/ChatGPT/Ollama/HF)
    ↓
Prompts (utils/prompts/) → Template-based LLM instructions
    ↓
Export (export.py) → CSV/ShareGPT/Emotion-LLaMA formats
    ↓
Evaluate (tools/evaluate.py) → Reference-free quality metrics (CLIP, CLAP, AU F1, NLI, ASR WER)
    ↓
Training (train.sh → LLaMA-Factory) → Prepare dataset & launch training UI
```

## Quick Reference

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point (Typer), arg parsing |
| `mer_factory/graph.py` | LangGraph pipeline definition and routing |
| `mer_factory/nodes/async_nodes.py` | Pipeline implementations (default) |
| `mer_factory/nodes/sync_nodes.py` | Sync implementations (HF models only) |
| `mer_factory/models/__init__.py` | LLM provider factory |
| `utils/config.py` | Configuration dataclasses |
| `utils/prompts/prompts.json` | LLM prompt templates |
| `export.py` | Dataset export (CSV/ShareGPT/Emotion-LLaMA) |
| `tools/evaluate.py` | **Reference-free evaluation** (CLIP, CLAP, AU F1, NLI, ASR WER, composite score) |
| `train.sh` | Training prep → LLaMA-Factory Web UI |

**Gate Agent** (dev feature): Enable with `--use-gate-agent` or `-uga`. Reviews intermediate analysis results and rejects low-quality outputs, prompting sub-agents to refine. See `mer_factory/nodes/gate_agent.py`.

## Project Structure

```
MER-Factory/
├── main.py                     # CLI entry point
├── export.py                   # Dataset export (CSV/ShareGPT/Emotion-LLaMA)
├── dashboard.py                # Flask web server for data curation
├── train.sh                    # Training preparation script
├── requirements.txt            # Python dependencies
│
├── mer_factory/                # Core package
│   ├── graph.py               # LangGraph pipeline + routing
│   ├── state.py               # MERRState definition
│   ├── prompts.py             # Prompt template loader
│   ├── tools.py               # Tool functions
│   ├── nodes/                 # Pipeline nodes
│   │   ├── async_nodes.py     # Async implementations (default)
│   │   ├── sync_nodes.py      # Sync for HF models
│   │   └── gate_agent.py      # Gate Agent (quality control, dev feature)
│   └── models/                # LLM provider integrations
│       ├── __init__.py        # LLMModels factory
│       ├── hf_models/         # HuggingFace models
│       ├── hf_api_server.py   # HF model API server
│       └── api_models/        # Gemini, ChatGPT, Ollama
│
├── utils/                      # Utilities
│   ├── config.py              # Configuration dataclasses
│   ├── file_handler.py        # File discovery
│   ├── processing_manager.py  # Pipeline orchestration
│   ├── register_dataset.py    # LLaMA-Factory dataset registration
│   ├── caching.py             # LLM response caching
│   └── prompts/               # Prompt templates
│       └── prompts.json       # Default prompts
│
├── tools/                      # Standalone tools
│   ├── evaluate.py            # Reference-free evaluation
│   ├── ffmpeg_adapter.py      # FFmpeg wrapper
│   ├── openface_adapter.py    # OpenFace wrapper
│   ├── facial_analyzer.py     # Facial analysis
│   └── emotion_analyzer.py    # Emotion analysis
│
├── test/                       # Test scripts
│   ├── test_ffmpeg.py         # FFmpeg integration test
│   └── test_openface.py       # OpenFace integration test
│
├── docs/                       # Jekyll documentation
├── examples/                   # Example outputs
└── LLaMA-Factory/              # Git submodule: training framework
```

## Adding Features: Where to Modify

### 1. Add a New LLM Provider

**Files to modify**:
1. `mer_factory/models/api_models/` or `mer_factory/models/hf_models/` - Implement model class
2. `mer_factory/models/__init__.py` - Register in `LLMModels.__init__()`
3. `main.py` - Add CLI argument (if needed)

**Steps**:
```python
# 1. Create mer_factory/models/api_models/your_provider.py
# Implement: analyze_audio(), describe_video(), describe_image(),
#           synthesize_summary(), describe_facial_expression()

# 2. Edit mer_factory/models/__init__.py:
#    - Import your model class
#    - Add entry to model_factory dict in __init__()

# 3. Edit main.py (if CLI arg needed):
#    @app.command()
#    def process(
#        ...
#        your_provider_model: str = typer.Option(None, "--your-provider", "-yp")
#    )
```

### 2. Add a New Processing Type (Pipeline)

**Files to modify**:
1. `utils/config.py` - Add to `ProcessingType` enum
2. `mer_factory/graph.py` - Add route in `route_by_processing_type()`
3. `mer_factory/nodes/async_nodes.py` - Implement node functions
4. `main.py` - Update help text if needed

**Steps**:
```python
# 1. utils/config.py
class ProcessingType(str, Enum):
    # ... existing types
    YOUR_NEW_TYPE = "your_new_type"

# 2. mer_factory/graph.py
def route_by_processing_type(state: MERRState) -> str:
    # ... existing routes
    if proc_type == "YOUR_NEW_TYPE":
        return "your_new_pipeline"

# 3. mer_factory/nodes/async_nodes.py
# Implement node functions for your pipeline
# Pattern: async def your_node_name(state: MERRState) -> MERRState:

# 4. mer_factory/graph.py - in create_graph()
# Add nodes and edges for your pipeline
```

### 3. Modify Pipeline Behavior / Add Node

**Files to modify**:
- `mer_factory/nodes/async_nodes.py` - Default for new/changed nodes
- `mer_factory/nodes/sync_nodes.py` - If HF model support needed
- `mer_factory/graph.py` - Wire up new nodes to the graph

**Node pattern**:
```python
# mer_factory/nodes/async_nodes.py
async def your_new_node(state: MERRState) -> MERRState:
    """Description of what this node does."""
    # Access state data
    input_path = state["input_path"]
    config = state["config"]

    # Do work...
    result = "some result"

    # Update state
    state["your_field"] = result
    return state
```

**Graph wiring** (`mer_factory/graph.py`):
```python
# In create_graph():
workflow.add_node("your_new_node", nodes.your_new_node)

# Add edge or conditional edge to connect your node
workflow.add_edge("previous_node", "your_new_node")
# OR
workflow.add_conditional_edges(
    "some_node",
    your_routing_function,
    {"your_new_node": "your_new_node", ...}
)
```

### 4. Change LLM Prompts

**Files to modify**:
- `utils/prompts/prompts.json` - Edit prompt templates
- `mer_factory/prompts.py` - If prompt loading logic needs changes

**Prompt structure** (`prompts.json`):
```json
{
  "user_questions": {
    "au": ["prompt1", "prompt2"],
    "mer": ["prompt1", "prompt2"],
    "audio": ["..."],
    "video": ["..."],
    "image": ["..."]
  }
}
```

### 5. Add Export Format

**Files to modify**:
1. `export.py` - Add export function and argument choice
2. `README.md` - Document new format

**Steps**:
```python
# 1. export.py - Add to argument choices
parser.add_argument(
    "--export_format",
    choices=["sharegpt", "emotion-llama", "emotion-llama-fine", "your_format"],
    ...
)

# 2. export.py - Implement export function
def export_to_your_format(all_data, export_path, ...):
    """Export data to your format."""
    # Implement export logic
    pass

# 3. export.py - Add handling in export_to_json()
if export_format == "your_format":
    export_to_your_format(all_data, export_path, ...)
    return
```

### 6. Add CLI Argument / Option

**Files to modify**:
- `main.py` - Add Typer option
- `utils/config.py` - Add to `AppConfig` if it's a config value

**Pattern**:
```python
# main.py - In @app.command() function signature
your_option: str = typer.Option(
    "default_value",
    "--your-option",
    "-yo",
    help="Description of your option"
)

# utils/config.py - If storing in config
@dataclass
class AppConfig:
    # ... existing fields
    your_option: str = "default_value"
```

### 7. Modify Evaluation Metrics

**Files to modify**:
- `tools/evaluate.py` - Main evaluation logic
- `tools/evaluate/` - Evaluation submodules

**Pattern**:
```python
# tools/evaluate.py
def compute_your_metric(data) -> float:
    """Compute your custom metric."""
    # Implement metric
    return score

# Add to DEFAULT_WEIGHTS and aggregate_scores()
```

## Code Conventions

### Async vs Sync Nodes
- **Default**: Use `async_nodes.py` (all providers except HF)
- **HF models only**: Use `sync_nodes.py` (automatically selected when `--huggingface-model` is used)
- **Graph selection**: `create_graph(use_sync_nodes=is_sync_model)` in `main.py`

### State Management
- All state flows through `MERRState` (defined in `mer_factory/state.py`)
- Nodes receive state, modify it, and return it
- Use `state.get()` for optional fields, `state["key"]` for required ones

### Error Handling
```python
# Pattern for error handling in nodes
try:
    # ... work
except Exception as e:
    state["error"] = str(e)
    console.print(f"[bold red]Error: {e}[/bold red]")
    return state
```

### Terminal Output
```python
from rich.console import Console
console = Console(stderr=True)

# Logging (quiet)
console.log("Message")

# Styled output
console.print("[bold green]Success![/bold green]")
console.print(f"[yellow]{variable}[/yellow]")
```

## Dependencies & External Tools

| Tool | Purpose | Config |
|------|---------|--------|
| FFmpeg | Video/audio processing | Must be in PATH |
| OpenFace | Facial Action Units | `OPENFACE_EXECUTABLE` in `.env` |
| Gemini API | Default LLM | `GOOGLE_API_KEY` in `.env` |
| OpenAI API | ChatGPT models | `OPENAI_API_KEY` in `.env` |
| Ollama | Local models | `--ollama-vision-model` / `--ollama-text-model` |
| HF API Server | HF models | Run separately, set `HF_API_BASE_URL` |

## Git Submodules

```bash
# LLaMA-Factory (training framework)
git submodule update --init --recursive

# If submodule is empty after pull
git submodule sync --recursive
git submodule update --init --recursive
```

## Before Making Changes

1. **Read the existing code** - Don't modify files you haven't read
2. **Identify async vs sync** - Most changes need both `async_nodes.py` and `sync_nodes.py`
3. **Check routing** - Pipeline changes usually require graph routing updates in `graph.py`
4. **Consider backward compatibility** - Export format changes affect existing datasets
