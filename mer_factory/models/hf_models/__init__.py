import importlib
from typing import Type

# Registry of Hugging Face models.
# This avoids importing them automatically for the incompatible modules dependency issues.
HUGGINGFACE_MODEL_REGISTRY = {
    "google/gemma-3n-E4B-it": (".gemma_multimodal", "GemmaMultimodalModel"),
    "google/gemma-3n-E2B-it": (".gemma_multimodal", "GemmaMultimodalModel"),
    "Qwen/Qwen2-Audio-7B-Instruct": (".qwen2_audio", "Qwen2AudioModel"),
    "zhifeixie/Audio-Reasoner": (".audio_reasoner", "AudioReasonerModel"),
    "Qwen/Qwen2.5-Omni-7B": (".qwen2_5_omni", "Qwen2_5OmniModel"),
    "Qwen/Qwen2.5-Omni-3B": (".qwen2_5_omni", "Qwen2_5OmniModel"),
    # DistilGPT2轻量级文本模型（英文）
    "distilbert/distilgpt2": (".custom_text_model", "DistilGPT2Model"),
    # Qwen2.5-0.5B-Instruct轻量级中文文本模型
    "Qwen/Qwen2.5-0.5B-Instruct": (".qwen2_5_0_5b", "Qwen2_5_0_5BModel"),
}


def get_hf_model_class(model_id: str) -> Type:
    """
    Dynamically finds and imports the correct model class from the registry.

    This function uses lazy loading. The module for a given model is only
    imported when this function is called, preventing dependency conflicts
    if you only use a subset of the available models.

    Args:
        model_id (str): The full model ID from Hugging Face.

    Returns:
        The corresponding model class if a match is found.

    Raises:
        ValueError: If the given model ID is not in the registry.
    """
    registry_entry = HUGGINGFACE_MODEL_REGISTRY.get(model_id)
    if not registry_entry:
        raise ValueError(
            f"Unsupported Hugging Face model ID: '{model_id}'. "
            f"Supported models are: {list(HUGGINGFACE_MODEL_REGISTRY.keys())}"
        )

    module_name, class_name = registry_entry

    try:
        module = importlib.import_module(module_name, package=__name__)
        model_class = getattr(module, class_name)
        return model_class
    except ImportError as e:
        print(
            f"Error: Failed to import dependencies for model '{model_id}'. "
            f"Please ensure all required libraries for this specific model are installed. "
            f"Original error: {e}"
        )
        raise e
