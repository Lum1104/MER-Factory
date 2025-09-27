from rich.console import Console
import diskcache

from .api_models.gemini import GeminiModel
from .api_models.ollama import OllamaModel
from .api_models.chatgpt import ChatGptModel
from .hf_models.hf_client import HFGradioClientModel
from utils.caching import cache_llm_call


console = Console(stderr=True)


class LLMModels:
    def __init__(
        self,
        api_key: str = None,
        ollama_text_model_name: str = None,
        ollama_vision_model_name: str = None,
        chatgpt_model_name: str = None,
        huggingface_model_id: str = None,
        cache: diskcache.Cache = None,
        verbose: bool = True,
    ):
        """
        Initializes the appropriate model and applies caching if provided.

        Args:
            api_key (str, optional): A generic API key.
            ollama_text_model_name (str, optional): Name of the Ollama text model.
            ollama_vision_model_name (str, optional): Name of the Ollama vision model.
            chatgpt_model_name (str, optional): Name of the ChatGPT model (e.g., 'gpt-4o').
            huggingface_model_id (str, optional): ID of the Hugging Face model.
            cache (diskcache.Cache, optional): A diskcache instance for LLM caching.
            verbose (bool): Whether to print verbose logs.
        """
        self.verbose = verbose
        self.model_instance = None
        self.model_type = None

        hf_model_class = None
        if huggingface_model_id:
            try:
                hf_model_class = HFGradioClientModel
            except ValueError as e:
                raise e

        model_factory = {
            "chatgpt": {
                "condition": chatgpt_model_name and api_key,
                "class": ChatGptModel,
                "args": {
                    "api_key": api_key,
                    "model_name": chatgpt_model_name,
                    "verbose": verbose,
                },
            },
            "huggingface": {
                "condition": huggingface_model_id and hf_model_class,
                "class": hf_model_class,
                "args": {"model_id": huggingface_model_id, "verbose": verbose},
            },
            "ollama": {
                "condition": ollama_vision_model_name or ollama_text_model_name,
                "class": OllamaModel,
                "args": {
                    "text_model_name": ollama_text_model_name,
                    "vision_model_name": ollama_vision_model_name,
                    "verbose": verbose,
                },
            },
            "gemini": {
                "condition": api_key
                and not chatgpt_model_name
                and not ollama_text_model_name,
                "class": GeminiModel,
                "args": {"api_key": api_key, "verbose": verbose},
            },
        }

        initialized = False
        for model_type, config in model_factory.items():
            if config["condition"]:
                self.model_type = model_type
                try:
                    self.model_instance = config["class"](**config["args"])
                    if cache is not None:
                        console.log(
                            f"Applying LLM cache to model: [yellow]{model_type}[/yellow]"
                        )
                        self._apply_caching(cache)
                    initialized = True
                except (ValueError, ImportError) as e:
                    console.print(
                        f"[bold red]Failed to initialize {config['class'].__name__}: {e}[/bold red]"
                    )
                    raise
                break

        if not initialized:
            active_conditions = {k: v["condition"] for k, v in model_factory.items()}
            if not any(active_conditions.values()):
                raise ValueError(
                    "No model could be initialized. Please provide the necessary arguments."
                )

    def _apply_caching(self, cache: diskcache.Cache):
        """
        Monkey-patches the model instance's methods with a caching decorator.
        This dynamically adds caching to any LLM provider's methods,
        supporting both sync and async methods.
        """
        methods_to_cache = [
            "analyze_audio",
            "describe_video",
            "describe_image",
            "synthesize_summary",
            "describe_facial_expression",
        ]

        for method_name in methods_to_cache:
            if hasattr(self.model_instance, method_name):
                original_method = getattr(self.model_instance, method_name)
                cached_method = cache_llm_call(cache)(original_method)
                setattr(self.model_instance, method_name, cached_method)
                if self.verbose:
                    console.log(
                        f"Applied LLM cache to method: [yellow]{method_name}[/yellow]"
                    )
