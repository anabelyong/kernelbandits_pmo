#!/bin/bash

# Force JAX to use CPU
export JAX_PLATFORM_NAME=cpu

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bandits_pmo

python -c "import jax; print('Backend:', jax.default_backend())"

python /home/anabel/kernelbandits_pmo/aug_fex_2.py
