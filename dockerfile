FROM aniweaverick/cuda126-miniconda-devel:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PATH=/opt/conda/bin:$PATH
ENV MODEL_WEIGHT_DIR=/workspace/UniWorld-V1/model_weight

WORKDIR /workspace

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    libopenmpi-dev openmpi-bin \
    wget unzip git zsh tmux \
 && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

# Conda config and environment creation
RUN source /root/miniconda3/etc/profile.d/conda.sh && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --set channel_priority strict && \
    conda config --set always_yes yes && \
    conda config --set changeps1 True && \
    conda create -n univa python=3.10 && \
    echo "source /root/miniconda3/etc/profile.d/conda.sh && conda activate univa" >> ~/.bashrc && \
    echo "source /root/miniconda3/etc/profile.d/conda.sh && conda activate univa" >> ~/.zshrc

# Clone repo
RUN git clone https://github.com/aniweave-ai/UniWorld-V1.git /workspace/UniWorld-V1

# Copy helper script
COPY setup_univa.sh /workspace/setup_univa.sh

# Set default working directory and shell
WORKDIR /workspace/UniWorld-V1
CMD ["/bin/zsh"]
