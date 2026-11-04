# Picks the Python that has Goof-E's dependencies. Sourced by the .command launchers.
# Override with GOOFE_PYTHON=/path/to/python, or name your conda env with GOOFE_CONDA_ENV.
ENV_NAME="${GOOFE_CONDA_ENV:-lerobot-mac}"
PY="${GOOFE_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" /opt/homebrew/Caskroom/miniforge/base; do
    if [[ -x "$base/envs/$ENV_NAME/bin/python" ]]; then PY="$base/envs/$ENV_NAME/bin/python"; break; fi
  done
fi
if [[ -z "$PY" && -n "$CONDA_PREFIX" ]]; then PY="$CONDA_PREFIX/bin/python"; fi
PY="${PY:-python3}"
