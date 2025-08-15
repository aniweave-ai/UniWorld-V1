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

# Download Hugging Face models
huggingface-cli download --resume-download LanguageBind/UniWorld-V1 \
    --local-dir ${MODEL_WEIGHT_DIR}/UniWorld-V1
huggingface-cli download --resume-download black-forest-labs/FLUX.1-dev \
    --local-dir ${MODEL_WEIGHT_DIR}/FLUX.1-dev
huggingface-cli download --resume-download google/siglip2-so400m-patch16-512 \
    --local-dir ${MODEL_WEIGHT_DIR}/siglip2-so400m-patch16-512
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir ${MODEL_WEIGHT_DIR}/Qwen2.5-VL-7B-Instruct

# Download training dataset
mkdir -p /workspace/UniWorld-V1/training_data
cd /workspace/UniWorld-V1/training_data
wget -O uniworld_removal_dataset_v1.1.zip https://training-dataset.utils.aniweave.ai/uniworld/uniworld_removal_dataset_v1.1.zip
unzip -o uniworld_removal_dataset_v1.1.zip
