#!/bin/sh

# Exit script if commands fail.
set -e

# Install latest commit of AllenNLP + AllenNLP Models from 'fairscale' branches.
pip install --no-deps --no-cache-dir git+https://github.com/allenai/allennlp.git@fairscale
pip install --no-deps --no-cache-dir git+https://github.com/allenai/allennlp-models.git@fairscale

# Print some debugging info.
pip freeze
allennlp test-install

# Train.
allennlp train https://raw.githubusercontent.com/epwalsh/allennlp-t5-fine-tuning/main/config.jsonnet -s /output/
