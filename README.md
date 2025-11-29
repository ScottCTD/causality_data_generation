# Causal Pool Simulation

## Env Setup

New:
```sh
uv sync
```

Legacy:
On Linux:
```sh
# install poetry
curl -sSL https://install.python-poetry.org | python3 -
poetry config cache-dir /home/scottc/scratch/poetry_venvs/

module load python/3.13

cd pooltool
poetry install --with=dev,docs
source ~/scratch/poetry_venvs/virtualenvs/pooltool-billiards-47-DzVsq-py3.13/bin/activate
uv pip install \
  --pre \
  --extra-index-url https://archive.panda3d.org/simple/ \
  "panda3d==1.11.0.dev3702"

export POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
pip install -e .

uv pip install pandas tqdm
```

## Generate Video Shots

Testing:
```sh
uv run python tests/test.py
```

## Generate Questions


```sh
python question_gen.py \
  --seed 42 \
  --dataset 1k_simple \
  --output outputs/ds1/raw_qa.jsonl \
  --num-options 4 \
  --num-correct 2 \
  --num-descriptive-per-shot 1 \
  --num-predictive-per-shot 1 \
  --max-velocity-cfs-per-shot 1 \
  --max-position-cfs-per-shot 1 \
  --exclude-invalid-hits \
  --predictive-filter-fraction 0.2


# only predictive
python question_gen.py \
  --dataset ds1 \
  --output outputs/ds1/all_predictive.jsonl \
  --num-descriptive-per-shot 0 \
  --num-predictive-per-shot 3 \
  --max-velocity-cfs-per-shot 0 \
  --max-position-cfs-per-shot 0 \
  --exclude-invalid-hits \
  --predictive-filter-fraction 0.05
```
