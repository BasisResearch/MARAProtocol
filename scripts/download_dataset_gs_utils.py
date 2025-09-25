#!/usr/bin/env bash

# This script downloads the benchmark dataset from the specified Google Cloud Storage bucket.
# It first downloads a manifest file and then downloads the files listed in the manifest.
#
# Prerequisites:
# 1. Google Cloud SDK installed. This script relies on the `gsutil` command.
#    Installation instructions: https://cloud.google.com/sdk/docs/install
# 2. Authenticated with GCP. Run `gcloud auth login` and `gcloud auth application-default login`.
# 3. `jq` installed (https://stedolan.github.io/jq/download/). On macOS: `brew install jq`.
#
# Usage:
# ./scripts/download_dataset.sh

set -euo pipefail

# --- Configuration ---

# The GCP bucket where the dataset is stored.
GCP_BUCKET="prolific-server-123.firebasestorage.app"

# The path to the root of the dataset inside the GCP bucket.
# Based on the screenshot, the manifest and data are at the root, so this is empty.
DATASET_ROOT_IN_BUCKET=""

# The local directory where the dataset will be downloaded.
# This will be created if it doesn't exist.
LOCAL_DESTINATION="python_examples/autumnbench/example_benchmark"

# The name of the manifest file.
MANIFEST_FILE="manifest_public.json"

# --- Script ---

# Check for gsutil
if ! command -v gsutil &> /dev/null
then
    echo "ERROR: 'gsutil' is not installed or not in your PATH."
    echo "Please install the Google Cloud SDK to continue: https://cloud.google.com/sdk/docs/install"
    exit 1
fi


# Check for jq
if ! command -v jq &> /dev/null
then
    echo "ERROR: 'jq' is not installed, but is required to parse the manifest file."
    echo "Please install jq to continue. See https://stedolan.github.io/jq/download/"
    exit 1
fi

# Construct the base URL for gsutil.
# This handles the case where the root path is empty.
if [[ -z "${DATASET_ROOT_IN_BUCKET}" ]]; then
    GS_BASE_URL="gs://${GCP_BUCKET}"
else
    # gsutil is sensitive to trailing slashes, so remove it if present
    CLEANED_ROOT_PATH=${DATASET_ROOT_IN_BUCKET%/}
    GS_BASE_URL="gs://${GCP_BUCKET}/${CLEANED_ROOT_PATH}"
fi

# Create destination if it doesn't exist
mkdir -p "${LOCAL_DESTINATION}"

MANIFEST_SOURCE_URL="${GS_BASE_URL}/${MANIFEST_FILE}"
LOCAL_MANIFEST_PATH="${LOCAL_DESTINATION}/${MANIFEST_FILE}"

echo "Downloading manifest from ${MANIFEST_SOURCE_URL}..."
gsutil -q cp -n "${MANIFEST_SOURCE_URL}" "${LOCAL_MANIFEST_PATH}"

if [[ ! -f "${LOCAL_MANIFEST_PATH}" ]]; then
    echo "ERROR: Manifest file could not be downloaded."
    exit 1
fi

echo "Manifest downloaded to ${LOCAL_MANIFEST_PATH}"
echo "Parsing manifest and downloading files..."

# This function downloads a single file from GCS
download_file() {
    local relative_path=$1
    local remote_url="${GS_BASE_URL}/${relative_path}"
    local local_path="${LOCAL_DESTINATION}/${relative_path}"

    # Create subdirectory if it doesn't exist
    mkdir -p "$(dirname "${local_path}")"

    echo "Downloading ${remote_url}"
    # Use -q for quieter output, and add -n to not overwrite existing files
    gsutil -q cp -n "${remote_url}" "${local_path}"
}

# Read the 'files' array from the manifest
# For each object, construct the paths and download the files.
# We assume the extensions are .json for answers/prompts and .sexp for programs.

jq -c '.files[]' "${LOCAL_MANIFEST_PATH}" | while read -r file_obj; do
    task_id=$(jq -r '.id' <<< "$file_obj")
    program_id=$(jq -r '.program' <<< "$file_obj")
    prompt_relative_path="prompts/${task_id}.json"
    local_prompt_path="${LOCAL_DESTINATION}/${prompt_relative_path}"

    # Download answer file
    download_file "answers/${task_id}.json"

    # Download prompt file
    download_file "${prompt_relative_path}"

    # Download program file
    download_file "programs/${program_id}.sexp"

    # For 'change_detection' tasks, also download the 'wrong_program'
    if [[ -f "${local_prompt_path}" ]]; then
        task_type=$(jq -r '.type' "${local_prompt_path}")
        if [[ "${task_type}" == "change_detection" ]]; then
            wrong_program_id=$(jq -r '.wrong_program // empty' "${local_prompt_path}")
            if [[ -n "${wrong_program_id}" ]]; then
                 download_file "programs/${wrong_program_id}.sexp"
            fi
        fi
    fi
done


echo "Download complete."
echo "Dataset content is in: ${LOCAL_DESTINATION}"
