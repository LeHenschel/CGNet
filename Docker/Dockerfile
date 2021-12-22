# Start with Docker pytorch base

FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# Copy requirements.txt
COPY requirements.txt /tmp/requirements.txt

# Install custom libraries
RUN apt-get update && apt-get install -y curl; \
        curl -sL https://deb.nodesource.com/setup_12.x | bash - ;\
	apt-get install -y nodejs

# Install python packages
RUN pip install -r /tmp/requirements.txt; rm -f /tmp/requirements.txt

# Add username upon docker creation
RUN useradd henschell --uid 1000

# Seems like this is automatically set in devel!
ENV TORCH_CUDA_ARCH_LIST=Maxwell \
    CUDA_HOME=/usr/local/cuda/ \
    PATH=/usr/local/cuda/:$PATH

WORKDIR "/"
