# HF models server
# Use this to serve the HF models for the API
# Avoid mess up the environment and enable asynchronous calls
# Author: Yuxiang Lin

import argparse
import gradio as gr
import json
from rich.console import Console
from pathlib import Path
from .hf_models import get_hf_model_class

console = Console()

# --- Globals ---
MODEL_INSTANCE = None


def predict(payload_str: str) -> str:
    """
    Dynamically calls a method on the loaded model instance in a thread-safe manner.
    """
    if not MODEL_INSTANCE:
        return json.dumps({"error": "Model is not initialized."})

    try:
        payload = json.loads(payload_str)
        method_name = payload["method"]
        params = payload["params"]
    except (json.JSONDecodeError, KeyError) as e:
        return json.dumps({"error": f"Invalid payload: {e}"})

    console.log(
        f"Received request for method: [cyan]{method_name}[/cyan] with params: {params}"
    )

    # Convert file paths from string to Path objects where needed
    for key, value in params.items():
        if isinstance(value, str) and ("path" in key or "url" in key):
            params[key] = Path(value)

    try:
        method = getattr(MODEL_INSTANCE, method_name)
        result = method(**params)
        console.log(f"Method: [yellow]{method_name}[/yellow] executed successfully.")

        return json.dumps({"result": result})
    except Exception as e:
        console.log(f"[bold red]Error executing method '{method_name}': {e}[/bold red]")
        return json.dumps({"error": str(e)})


def main():
    global MODEL_INSTANCE

    parser = argparse.ArgumentParser(
        description="Gradio API Server for Hugging Face Models"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="The Hugging Face model ID to serve.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on."
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the server on."
    )
    args = parser.parse_args()

    console.rule(
        f"[bold green]Starting Gradio Server for: {args.model_id}[/bold green]"
    )

    try:
        ModelClass = get_hf_model_class(args.model_id)
        MODEL_INSTANCE = ModelClass(model_id=args.model_id, verbose=True)
        console.log(f"Successfully initialized model [cyan]{args.model_id}[/cyan].")
    except (ValueError, ImportError) as e:
        console.log(f"[bold red]Failed to load model: {e}[/bold red]")
        return

    interface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=10, label="JSON Payload"),
        outputs=gr.Textbox(label="JSON Output"),
        title=f"API for {args.model_id}",
        description=f"Accepts a JSON payload with 'method' and 'params' to call the model.",
    )

    interface.queue().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
