FROM python:3.8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /stage/

# Ensure the version of AllenNLP we want isn't overwritten when we install allennlp-models.
ENV ALLENNLP_VERSION_OVERRIDE allennlp

# Disable parallelism in tokenizers because it doesn't help, and sometimes hurts.
ENV TOKENIZERS_PARALLELISM 0

# Install torch ecosystem.
RUN pip install --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir fairscale==0.3.7

# Install other requirements.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
