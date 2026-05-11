# FaithRec RecBench

This repository contains the code needed for:

- building ReRec-style RecBench+ reranking data;
- training the main RL policy in the local `verl` checkout;
- optionally building deterministic schema SFT warm-start data for ablations or format fallback.

The data policy follows ReRec's main setting: each query is converted into a candidate-aware reranking instance with 20 candidates, including 1 positive item and 19 random negatives by default.

## Layout

```text
data/recbench/
  books.dat
  movies.dat
scripts/
  prepare_recbench.py
  check_recbench_rl_data.py
src/faithrec/
  recbench.py
  reward.py
teacher/
  build_sft.py
verl/
  scripts/configs/recbench_rl.env
  scripts/train_recbench_sft.sh
  scripts/train_recbench_rl.sh
  verl/utils/reward_score/recbench_json.py
```

Generated files live under `data/recbench/processed/` and are ignored by git.

## 1. Prepare RecBench Data

Run this after cloning RecBench+ locally. Regenerate the parquet files whenever the prompt, output schema, reward metadata, or split policy changes, because the full prompt and `extra_info` are stored inside each parquet row.

```powershell
cd D:\SCUT\26_spring\CIKM
conda activate rec

python .\scripts\prepare_recbench.py `
  --recbench-root D:\SCUT\26_spring\RecBenchPlus-main `
  --item-dir .\data\recbench `
  --output-dir .\data\recbench\processed `
  --domains movie
```

Default split sizes are `--train-size 10000`, `--dev-size 1000`, and `--test-size 12000` per domain. `--max-instances` defaults to `train + dev + test`. The script filters rows whose positive item cannot be placed into the candidate set unless `--keep-no-positive-candidates` is set.

Expected movie-domain outputs:

```text
data/recbench/processed/canonical/movie_train.jsonl
data/recbench/processed/canonical/movie_dev.jsonl
data/recbench/processed/canonical/movie_test.jsonl
data/recbench/processed/rl/movie_train.parquet
data/recbench/processed/rl/movie_dev.parquet
data/recbench/processed/rl/movie_test.parquet
```

Before training, check the RL parquet schema:

```powershell
python .\scripts\check_recbench_rl_data.py `
  .\data\recbench\processed\rl\movie_train.parquet `
  .\data\recbench\processed\rl\movie_dev.parquet `
  .\data\recbench\processed\rl\movie_test.parquet
```

If this check reports missing `rationale` prompts, missing `extra_info` keys, or empty `positive_candidate_ids`, delete the old processed files and rerun `prepare_recbench.py`.

## 2. Main RL Training Flow

The main experiment is RL-first: start from a base instruct model and optimize recommendation correctness together with verifier-style rationale faithfulness on the RL parquet files. The model must return one JSON object:

```json
{
  "ranking": ["M", "N", "..."],
  "selected_candidate_id": "M",
  "evidence_refs": ["Q01", "Q02", "M"],
  "rationale": [
    {
      "candidate_id": "M",
      "claim": "matches_query",
      "support": ["Q01", "Q02", "M"]
    }
  ]
}
```

The `rationale` field is a compact audit object, not free-form chain-of-thought. The reward verifies parse success, schema validity, candidate IDs, support IDs, selected-candidate binding, full ranking validity, evidence grounding, and structured rationale validity.

The RL reward hook is `verl/verl/utils/reward_score/recbench_json.py`. `recommendation` is NDCG@`REWARD_K`, so the target item receives different credit at rank 1, 2, 3, and so on. Invalid rankings, including duplicates, missing candidates, unknown candidates, or `selected_candidate_id != ranking[0]`, receive zero recommendation reward. `faithfulness` is the response-level verifier score over the schema, evidence references, and rationale support.

### One-Time W&B Setup

On the GPU machine:

```powershell
cd D:\SCUT\26_spring\CIKM\verl
pip install wandb
wandb login
```

If your account uses a W&B team/entity, set it before training:

```powershell
$env:WANDB_ENTITY="your-wandb-entity"
```

### Start RL From Scratch

Use the shared RL config as the default single-GPU training recipe:

```powershell
cd D:\SCUT\26_spring\CIKM\verl
ray stop --force
$env:CUDA_VISIBLE_DEVICES="0"

bash ./scripts/train_recbench_rl.sh `
  --config ./scripts/configs/recbench_rl.env `
  trainer.resume_mode=disable
```

The config uses:

```text
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
TRAIN_FILE=../data/recbench/processed/rl/movie_train.parquet
VAL_FILE=../data/recbench/processed/rl/movie_dev.parquet
TRAIN_BATCH_SIZE=128
ROLLOUT_N=8
ACTOR_LR=5e-7
KL_LOSS_COEF=0.003
TOTAL_EPOCHS=8
SAVE_FREQ=50
TEST_FREQ=100
GDPO_REWARD_KEYS=['recommendation','faithfulness']
GDPO_REWARD_WEIGHTS=[1.0,1.0]
REWARD_MODE=faithrl_hybrid
REWARD_K=10
```

`movie_dev.parquet` is used for validation during training. Keep `movie_test.parquet` for final held-out evaluation.

### Resume From Checkpoint

Continue the same experiment from the latest checkpoint:

```powershell
cd D:\SCUT\26_spring\CIKM\verl
$env:CUDA_VISIBLE_DEVICES="0"

bash ./scripts/train_recbench_rl.sh `
  --config ./scripts/configs/recbench_rl.env `
  trainer.resume_mode=auto
```

### Multi-GPU Run

The shared config uses shell defaults, so override only the multi-GPU variables at launch:

```powershell
cd D:\SCUT\26_spring\CIKM\verl
$env:CUDA_VISIBLE_DEVICES="0,1"
$env:NGPUS_PER_NODE="2"
$env:ROLLOUT_TP="2"
$env:ROLLOUT_GPU_MEMORY_UTILIZATION="0.5"

bash ./scripts/train_recbench_rl.sh `
  --config ./scripts/configs/recbench_rl.env `
  trainer.resume_mode=disable
```

### Local Smoke Test

For a short wiring check without W&B, run without the full single-GPU config and override the small-run variables directly:

```powershell
cd D:\SCUT\26_spring\CIKM\verl
$env:CUDA_VISIBLE_DEVICES="0"
$env:TRAINER_LOGGER='["console"]'
$env:WANDB_MODE="disabled"
$env:TRAIN_BATCH_SIZE="8"
$env:PPO_MINI_BATCH_SIZE="4"
$env:ROLLOUT_N="2"
$env:TOTAL_EPOCHS="1"
$env:TEST_FREQ="1"

bash ./scripts/train_recbench_rl.sh trainer.resume_mode=disable
```

Clear those temporary PowerShell variables or open a fresh terminal before the real run.

## 3. Optional Schema SFT Warm-Start

SFT is not part of the main result. Use it only as an ablation or fallback if the base model's initial JSON/rationale parse rate is too low for stable RL. The deterministic `format` target teaches the output schema and places the positive item first, but does not teach free-form reasoning or arbitrary negative ordering.

Build schema SFT files:

```powershell
cd D:\SCUT\26_spring\CIKM

python .\teacher\build_sft.py `
  --target-mode format `
  --input .\data\recbench\processed\canonical\movie_train.jsonl `
  --output .\data\recbench\processed\sft\movie_train_schema.parquet

python .\teacher\build_sft.py `
  --target-mode format `
  --input .\data\recbench\processed\canonical\movie_dev.jsonl `
  --output .\data\recbench\processed\sft\movie_dev_schema.parquet
```

Train the schema warm-start checkpoint:

```powershell
cd D:\SCUT\26_spring\CIKM\verl
$env:MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
$env:TRAIN_FILE="../data/recbench/processed/sft/movie_train_schema.parquet"
$env:VAL_FILE="../data/recbench/processed/sft/movie_dev_schema.parquet"
$env:EXPERIMENT_NAME="recbench-schema-sft"

bash ./scripts/train_recbench_sft.sh
```

To run the `SFT -> RL` ablation, launch RL with `verl/scripts/configs/recbench_rl.env` and override `MODEL_PATH` plus `EXPERIMENT_NAME` for that run.

An OpenAI-compatible teacher mode remains available for ablations:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"

python .\teacher\build_sft.py `
  --target-mode teacher `
  --input .\data\recbench\processed\canonical\movie_train.jsonl `
  --output .\data\recbench\processed\sft\movie_train.teacher.parquet `
  --model gpt-4.1-mini
```

## 4. Useful Monitoring Signals

During RL, the key W&B signals are:

- `gdpo/recommendation/mean`: should trend upward over time.
- `gdpo/faithfulness/mean`: should trend upward or at least not collapse while recommendation improves.
- `gdpo/recommendation/std`: may rise when the group contains both good and bad rankings; verify the mean and sample generations together.
- `critic/rewards/mean`: should not stay at a flat failure value for long.
- `actor/kl_loss`: should rise gradually; fast growth suggests lowering `ACTOR_LR` or raising KL control.
- `actor/entropy`: should not collapse early; if it approaches zero too fast, reduce learning rate or increase KL.
- validation generations in W&B: inspect JSON parse rate, full ranking validity, evidence refs, and rationale support IDs.

## Notes

- `data/recbench/books.dat` and `data/recbench/movies.dat` are item metadata files used by this project.
- `scripts/prepare_recbench.py` reads query and task files from the cloned RecBench+ repository and item metadata from `data/recbench`.
- Use `movie_dev.parquet` for validation and keep `movie_test.parquet` for final reporting.
- FaithRL's token-level FAAM path requires a reward model that writes a `sentence_mask` tensor into the rollout batch. This project currently uses response-level RecBench JSON verification through GDPO instead.
