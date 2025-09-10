#!/bin/bash
set -e

# Initialize conda
source /root/miniconda3/etc/profile.d/conda.sh
conda activate univa

# Install Python dependencies
pip install -r /workspace/UniWorld-V1/requirements.txt
pip install flash_attn==2.7.3 --no-build-isolation
pip install google-genai
pip install mpi4py

# Create model weight directory
mkdir -p ${MODEL_WEIGHT_DIR}

export HF_HUB_DISABLE_XET="1"  # disable xet backend to avoid download freeze

# Download Hugging Face models
hf download LanguageBind/UniWorld-V1 \
    --local-dir ${MODEL_WEIGHT_DIR}/UniWorld-V1
hf download black-forest-labs/FLUX.1-dev \
    --local-dir ${MODEL_WEIGHT_DIR}/FLUX.1-dev
hf download google/siglip2-so400m-patch16-512 \
    --local-dir ${MODEL_WEIGHT_DIR}/siglip2-so400m-patch16-512


# Download training dataset
mkdir -p /workspace/UniWorld-V1/training_data
cd /workspace/UniWorld-V1/training_data
wget -O uniworld_removal_dataset_v2.0.0.zip https://training-dataset.utils.aniweave.ai/uniworld/uniworld_removal_dataset_v2.0.0.zip
unzip -o uniworld_removal_dataset_v2.0.0.zip


# modify data.txt from training dataset

# File to modify
FILE="/workspace/UniWorld-V1/training_data/uniworld_removal_dataset_v2.0.0/data.txt"

# Old and new path
OLD_PATH="/Users/rick/Desktop/AniWeave/scene-gen-train/data/uniworld_removal_dataset_v2.0.0"
NEW_PATH="/workspace/UniWorld-V1/training_data/uniworld_removal_dataset_v2.0.0"

# Replace paths in-place
sed -i.bak "s|$OLD_PATH|$NEW_PATH|g" "$FILE"

echo "Path replacement done. Backup saved as $FILE.bak"