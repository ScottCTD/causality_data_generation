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

Shots are stored in `./outputs/dataset_name/shots/shot_id/`. Each `shot_id` has multiple video files, one for each camera setting.

## Generate Questions

```sh
python question_gen/question_gen.py -d ds2 -D 5 -p 5 -v 5 -P 5 -e
python question_gen/generate_stats.py outputs/ds2/raw_qa.jsonl
```
