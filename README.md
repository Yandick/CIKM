# FaithRec RecBench

This repository is now kept intentionally small. It only contains the code needed for:

- building ReRec-style RecBench+ reranking data;
- collecting high-quality SFT traces from an OpenAI-compatible teacher model;
- training and RL fine-tuning in the local `verl` checkout.

The data policy follows ReRec's main setting: each query is converted into a candidate-aware reranking instance with 20 candidates, including 1 positive item and 19 random negatives by default.

## Layout

```text
data/recbench/
  books.dat
  movies.dat
scripts/
  prepare_recbench.py
src/faithrec/
  recbench.py
  reward.py
teacher/
  build_sft.py
verl/
  scripts/train_recbench_sft.sh
  scripts/train_recbench_rl.sh
  verl/utils/reward_score/recbench_json.py
```

Generated files live under `data/recbench/processed/` and are ignored by git.

## 1. Prepare RecBench Data

```powershell
python scripts\prepare_recbench.py `
  --recbench-root D:\SCUT\26_spring\RecBenchPlus-main `
  --item-dir data\recbench `
  --output-dir data\recbench\processed
```

By default the script builds at most `--train-size + --test-size` instances per domain. Use `--max-instances` if you want a smaller smoke test or a larger full export.

Outputs:

```text
data/recbench/processed/canonical/movie_train.jsonl
data/recbench/processed/canonical/movie_test.jsonl
data/recbench/processed/canonical/book_train.jsonl
data/recbench/processed/canonical/book_test.jsonl
data/recbench/processed/rl/movie_train.parquet
data/recbench/processed/rl/movie_test.parquet
data/recbench/processed/rl/book_train.parquet
data/recbench/processed/rl/book_test.parquet
```

The canonical JSONL is used for teacher generation. The RL parquet is directly consumed by verl.

## 2. Build Teacher SFT Data

Use an OpenAI-compatible closed teacher model to produce strong reasoning traces. The hidden gold candidate IDs are only given to the teacher-generation script for filtering and quality control; they are not included in the student prompt.

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"

python teacher\build_sft.py `
  --input data\recbench\processed\canonical\movie_train.jsonl `
  --output data\recbench\processed\sft\movie_train.parquet `
  --model gpt-4.1-mini
```

Run the same command for `movie_test.jsonl`, `book_train.jsonl`, and `book_test.jsonl` when you need validation files or both domains.

Accepted rows require the positive item at top-1, valid candidate-only ranking, and valid evidence references by default.

## 3. Train With verl

Run these from inside `verl` on the training machine.

```bash
cd verl

MODEL_PATH=/path/to/base-model \
TRAIN_FILE=../data/recbench/processed/sft/movie_train.parquet \
VAL_FILE=../data/recbench/processed/sft/movie_test.parquet \
bash scripts/train_recbench_sft.sh
```

Then run RL:

```bash
cd verl

MODEL_PATH=/path/to/sft-checkpoint \
TRAIN_FILE=../data/recbench/processed/rl/movie_train.parquet \
VAL_FILE=../data/recbench/processed/rl/movie_test.parquet \
bash scripts/train_recbench_rl.sh
```

The RL reward hook is `verl/verl/utils/reward_score/recbench_json.py`. It evaluates the final JSON answer for ranking quality, format validity, and evidence-reference validity.
The default RL launcher uses verl GRPO with the custom rule reward.

## Notes

- `data/recbench/books.dat` and `data/recbench/movies.dat` are the item metadata files used by this project.
- `scripts/prepare_recbench.py` reads query files from the cloned RecBench+ repository and item metadata from `data/recbench`.
- If the local RecBench+ clone has fewer available rows than `--train-size + --test-size`, the script writes all available rows after the train split.
