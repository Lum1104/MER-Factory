# utils/register_dataset.py

import json
import argparse
import sys
import hashlib
from pathlib import Path


def calculate_sha1(file_path: Path) -> str:
    """Calculates the SHA1 hash of a file."""
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)  # Read in 64k chunks
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def register_llama_factory(dataset_name: str, file_path: Path, file_type: str):
    """
    Handles dynamic registration for LLaMA-Factory based on file_type.
    """
    llama_factory_path = Path(__file__).parent.parent / "LLaMA-Factory"
    if not llama_factory_path.is_dir():
        print("Error: LLaMA-Factory submodule not found. Please run 'git submodule update --init --recursive' first.")
        sys.exit(1)

    dataset_info_path = llama_factory_path / "data" / "dataset_info.json"
    if not dataset_info_path.exists():
        print(f"Error: dataset_info.json not found at {dataset_info_path}")
        sys.exit(1)

    try:
        print(f"Calculating SHA1 hash for LLaMA-Factory file...")
        file_sha1 = calculate_sha1(file_path)
        print(f"File SHA1 hash: {file_sha1}")

        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)

        if dataset_name in dataset_info:
            print(f"Warning: Dataset '{dataset_name}' already exists. It will be overwritten.")

        # --- Core dynamic configuration logic ---
        columns = {
            "messages": "messages",
            "files": "files"  # 'files' is a general field
        }

        if file_type == "image":
            columns["images"] = "images"
            print("Image task detected, added 'images' field to columns.")
        elif file_type == "video":
            columns["videos"] = "videos"
            print("Video task detected, added 'videos' field to columns.")
        elif file_type == "audio":
            columns["audios"] = "audios"
            print("Audio task detected, added 'audios' field to columns.")
        elif file_type == "mer":  # Assuming 'mer' task combines audio and video
            columns["videos"] = "videos"
            columns["audios"] = "audios"
            print("MER task detected, added 'videos' and 'audios' fields to columns.")
        # --- End ---

        tags = {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "gpt"
        }

        dataset_info[dataset_name] = {
            "file_name": str(file_path.resolve()),  # Use absolute path to avoid confusion
            "file_sha1": file_sha1,
            "formatting": "sharegpt",
            "columns": columns,
            "tags": tags
        }

        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=4, ensure_ascii=False)

        print(f"✅ Successfully registered dataset '{dataset_name}' (type: {file_type}) to LLaMA-Factory.")

    except Exception as e:
        print(f"Error: An error occurred while updating dataset_info.json: {e}")
        sys.exit(1)


def register_ms_swift(file_path: Path):
    """
    Generates the dataset command-line argument for ms-swift.
    """
    abs_path = file_path.resolve()
    # Print this argument to be captured by the train.sh script
    print(f"--dataset {abs_path}")
    print(f"✅ Dataset parameter generated for ms-swift.")


def main():
    parser = argparse.ArgumentParser(description="Registers an exported dataset with a specified training framework.")
    parser.add_argument("--framework", required=True, choices=["llama_factory", "ms_swift"], help="The target training framework.")
    parser.add_argument("--dataset_name", required=True, help="A unique name to assign to the dataset.")
    parser.add_argument("--file_path", required=True, type=Path, help="Path to the exported JSON or JSONL file.")
    # Added file_type argument to dynamically generate columns
    parser.add_argument("--file_type", required=True, help="The type of the exported file (e.g., 'image', 'video', 'mer').")

    args = parser.parse_args()

    if not args.file_path.exists():
        print(f"Error: Dataset file not found at {args.file_path}")
        sys.exit(1)

    if args.framework == "llama_factory":
        register_llama_factory(args.dataset_name, args.file_path, args.file_type)
    elif args.framework == "ms_swift":
        register_ms_swift(args.file_path)


if __name__ == "__main__":
    main()