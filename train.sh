#!/bin/bash

# ===================================================================================
# MER-Factory: Data Preparation and UI Launcher
#
# This script prepares a dataset from MER-Factory outputs, registers it,
# and then launches the LLaMA-Factory Web UI for manual training configuration.
# ===================================================================================

set -e # Exit immediately if any command exits with a non-zero status.
set -o pipefail # The return value of a pipeline is the status of the last command to exit with a non-zero status.

# --- Configuration ---
FRAMEWORK="llama_factory" # Specify the training framework, llama_factory / ms-swift
OUTPUT_DIR_BASE="./output_models" # # Specify where training model output stored
DATA_SOURCE_DIR="/path/to/your/origin/dataset" # Path to the MER-Factory analysis results folder
FILE_TYPE="mer" # The type of analysis file to process
DATASET_NAME="OriginDatasetName_ModelName_MissionType" # eg: mer2025_llava_llama3.2_MER
EXPORT_DIR="./training_data" # Path to the final results folder
mkdir -p "${EXPORT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/${FRAMEWORK}_${DATASET_NAME}"
INTERMEDIATE_CSV_PATH="${EXPORT_DIR}/${FILE_TYPE}_export_data.csv"

# --- Help Function ---
usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -f, --framework <name>      Specify the training framework: 'llama_factory' (currently supported) or ms-swift. (Default: ${FRAMEWORK})"
    echo "  -o, --output_dir <path>     Specify the root directory to save trained models. (Default: ${OUTPUT_DIR_BASE})"
    echo "  -d, --data_source <path>    Path to the MER-Factory analysis results folder. (Default: ${DATA_SOURCE_DIR})"
    echo "  -t, --file_type <type>      The type of analysis file to process (e.g., 'mer', 'image', 'video'). (Default: ${FILE_TYPE})"
    echo "  -n, --dataset_name <name>   Specify a unique name for your dataset. (Default: ${DATASET_NAME})"
    echo "  -h, --help                  Show this help message."
    echo
    echo "Example: ./train.sh -n my_video_dataset -t video"
}

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--framework) FRAMEWORK="$2"; shift ;;
        -o|--output_dir) OUTPUT_DIR_BASE="$2"; shift ;;
        -d|--data_source) DATA_SOURCE_DIR="$2"; shift ;;
        -t|--file_type) FILE_TYPE="$2"; shift ;;
        -n|--dataset_name) DATASET_NAME="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown parameter: $1"; usage; exit 1 ;;
    esac
    shift
done

# --- Step 1: Export Dataset ---
echo "🚀 [Step 1/3] Exporting dataset for framework: ${FRAMEWORK}..."
echo "   - (1/2) Consolidating analysis results into an intermediate CSV..."
python export.py \
    --output_folder "${DATA_SOURCE_DIR}" \
    --file_type "${FILE_TYPE}" \
    --export_path "${EXPORT_DIR}" \
    --export_csv
echo "   - Intermediate CSV file created: ${INTERMEDIATE_CSV_PATH}"

echo "   - (2/2) Converting CSV to the final training format..."
if [ "$FRAMEWORK" == "llama_factory" ]; then
    EXPORT_FORMAT="sharegpt"
    EXPORT_FILE_EXT="json"
else
    echo "❌ Error: Only 'llama_factory' framework is currently supported."
    exit 1
fi
EXPORT_FILE_PATH="${EXPORT_DIR}/${FILE_TYPE}_${EXPORT_FORMAT}_export.${EXPORT_FILE_EXT}"

python export.py \
    --input_csv "${INTERMEDIATE_CSV_PATH}" \
    --export_format "${EXPORT_FORMAT}" \
    --json_format "${EXPORT_FILE_EXT}" \
    --export_path "${EXPORT_DIR}" \
    --file_type "${FILE_TYPE}"
echo "✅ Dataset exported to: ${EXPORT_FILE_PATH}"
echo

# --- Step 2: Register Dataset ---
echo "📝 [Step 2/3] Registering dataset..."
if [ "$FRAMEWORK" == "llama_factory" ]; then
    python utils/register_dataset.py \
        --framework llama_factory \
        --dataset_name "${DATASET_NAME}" \
        --file_path "${EXPORT_FILE_PATH}" \
        --file_type "${FILE_TYPE}"
fi
echo "✅ Dataset registration complete."
echo

# --- Step 3: Launch Graphical Training Interface (Web UI) ---
echo "🌐 [Step 3/3] Launching LLaMA-Factory Web UI..."
echo "=============================================="
echo "Newly registered dataset:   ${DATASET_NAME}"
echo "Suggested model output dir:   ${OUTPUT_DIR}"
echo "=============================================="
echo "👉 In the opened web page, please [Manually select the model], [Select the dataset: ${DATASET_NAME}], and [Fill in the output directory]."
echo "👉 After confirming all parameters are correct, click the 'Start' button to launch training."

if [ "$FRAMEWORK" == "llama_factory" ]; then
    cd LLaMA-Factory
    # Launch a clean Web UI for the user to configure
    llamafactory-cli webui
fi

echo "🎉 Web UI has been closed."