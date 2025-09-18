#!/usr/bin/env bash

# This script downloads the benchmark dataset from the specified Google Cloud Storage bucket.
# It first downloads a manifest file and then downloads the files listed in the manifest.
#
# Prerequisites:
# 1. `curl` installed for downloading files.
# 2. `jq` installed (https://stedolan.github.io/jq/download/).
#
# You can run the `scripts/setup.sh` script to install these dependencies.
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

# Check for curl
if ! command -v curl &> /dev/null
then
    echo "ERROR: 'curl' is not installed or not in your PATH."
    echo "Please install curl to continue, or run 'scripts/setup.sh'."
    exit 1
fi


# Check for jq
if ! command -v jq &> /dev/null
then
    echo "ERROR: 'jq' is not installed, but is required to parse the manifest file."
    echo "Please install jq to continue (e.g., 'brew install jq' or 'sudo apt-get install jq'), or run 'scripts/setup.sh'."
    exit 1
fi

# Construct the base URL for Firebase Storage HTTPS access.
# This handles the case where the root path is empty.
if [[ -z "${DATASET_ROOT_IN_BUCKET}" ]]; then
    HTTPS_BASE_URL="https://firebasestorage.googleapis.com/v0/b/${GCP_BUCKET}/o/"
else
    # gsutil is sensitive to trailing slashes, so remove it if present
    CLEANED_ROOT_PATH=${DATASET_ROOT_IN_BUCKET%/}
    HTTPS_BASE_URL="https://firebasestorage.googleapis.com/v0/b/${GCP_BUCKET}/o/${CLEANED_ROOT_PATH}%2F"
fi

# Create destination if it doesn't exist
mkdir -p "${LOCAL_DESTINATION}"

MANIFEST_SOURCE_URL="${HTTPS_BASE_URL}${MANIFEST_FILE}?alt=media"
LOCAL_MANIFEST_PATH="${LOCAL_DESTINATION}/${MANIFEST_FILE}"

echo "Downloading manifest from ${MANIFEST_SOURCE_URL}..."
# Use curl to download the file. -f fails silently on server errors. -L follows redirects.
# -n is not directly supported, but we can check if the file exists first.
if [[ -f "${LOCAL_MANIFEST_PATH}" ]]; then
    echo "Manifest file already exists. Skipping download."
else
    curl -fL "${MANIFEST_SOURCE_URL}" -o "${LOCAL_MANIFEST_PATH}"
fi

if [[ ! -f "${LOCAL_MANIFEST_PATH}" ]]; then
    echo "ERROR: Manifest file could not be downloaded."
    exit 1
fi

echo "Manifest downloaded to ${LOCAL_MANIFEST_PATH}"
echo "Parsing manifest and downloading files..."

# This function downloads a single file from GCS via HTTPS
download_file() {
    local relative_path=$1
    
    # URL-encode the relative path, especially the slashes
    local encoded_relative_path
    encoded_relative_path=$(echo -n "${relative_path}" | jq -sRr @uri)

    local remote_url="${HTTPS_BASE_URL}${encoded_relative_path}?alt=media"
    local local_path="${LOCAL_DESTINATION}/${relative_path}"

    # Create subdirectory if it doesn't exist
    mkdir -p "$(dirname "${local_path}")"

    # Skip download if file exists
    if [[ -f "${local_path}" ]]; then
        echo "Skipping existing file: ${local_path}"
        return
    fi

    echo "Downloading ${remote_url}"
    # Use -f to fail on server errors, -L to follow redirects.
    curl -fL "${remote_url}" -o "${local_path}"
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
