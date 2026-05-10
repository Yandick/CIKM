# FaithRec RecBench

This repository is now kept intentionally small. It only contains the code needed for:

- building ReRec-style RecBench+ reranking data;
- training the main RL-only policy in the local `verl` checkout;
- building deterministic format/top-1 SFT warm-start data only for ablations or format fallback.

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
  scripts/configs/recbench_movie_k10_hybrid.env
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

The RL parquet is the main training input consumed by verl. Regenerate these files whenever the prompt or output schema changes, because the prompt text is stored inside the parquet rows.

## 2. RL-Only Training (Main)

The main experiment follows a FaithRL-style RL-only setup: start from a base instruct model and optimize recommendation correctness together with verifier-style rationale faithfulness on the RL parquet files.

The model must return one JSON object:

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

The `rationale` field is a compact audit object, not free-form chain-of-thought. The reward verifies candidate IDs, support IDs, selected-candidate binding, ranking validity, and evidence grounding. Semantic claim verification is intentionally not a hard gate until a richer item-attribute graph or fact table is available.

The default RL script uses a FaithRL-style hybrid geometric reward mode and GDPO advantage estimation over `recommendation` and `faithfulness`. `recommendation` is a reranking reward, NDCG@`REWARD_K` by default, so moving the target item from rank 1 to rank 2/3/10 changes the reward by the usual logarithmic discount. The FaithRL correctness branch is still top-1 correctness, so a positive item at rank 2 can receive reranking credit without being treated as a correct final recommendation. `BASELINE_CORRECT_RATE` and `BASELINE_UNFAITHFUL_RATE` are the calibration constants analogous to FaithRL's baseline coordinates. Start with the defaults for smoke tests, then replace them with rates measured from a base-model validation run.

This is the part of FaithRL that is directly compatible with the current local verl checkout. FaithRL's token-level FAAM path additionally depends on a reward model that writes a `sentence_mask` tensor into the rollout batch. This project does not yet have that tensor channel for RecBench JSON rationales, so the main implementation uses response-level verifier faithfulness through GDPO instead of pretending to run token-level FAAM.

### Training Framework Usage

The main training launcher is `verl/scripts/train_recbench_rl.sh`. It wraps `verl.trainer.main_ppo` with the RecBench JSON reward hook and these default framework choices:

```text
policy init:        base instruct model, not SFT by default
rollout backend:    vLLM
RL algorithm:       PPO-style verl trainer
advantage:          GDPO over recommendation and faithfulness
reward mode:        FaithRL-style hybrid geometric reward
rerank signal:      NDCG@10 by default; override with REWARD_K
online logging:     console + Weights & Biases
validation samples: 8 generations logged to W&B by default
```

The current implementation uses response-level verifier faithfulness through GDPO. It does not start a FaithRL LLM-as-a-Judge server during training, because token-level FAAM requires an additional `sentence_mask` tensor channel that is not wired into this RecBench JSON reward path yet.

On the data-preparation machine, regenerate the parquet after any prompt or schema change:

```powershell
cd D:\SCUT\26_spring\CIKM
conda activate rec

python .\scripts\prepare_recbench.py `
  --recbench-root D:\SCUT\26_spring\RecBenchPlus-main `
  --item-dir .\data\recbench `
  --output-dir .\data\recbench\processed
```

On the GPU training machine, install/login to W&B once. Omit `WANDB_ENTITY` in the commands below if your W&B account does not use a team/entity name.

```bash
cd /path/to/CIKM/verl
pip install wandb
wandb login
```

Train the movie domain:

```bash
cd /path/to/CIKM/verl

ray stop --force
bash scripts/train_recbench_rl.sh \
  --env-file scripts/configs/recbench_movie_k10_hybrid.env \
  trainer.resume_mode=disable
```

The launcher accepts `--env-file`/`--config` before any verl Hydra overrides. The env file is sourced first, then `train_recbench_rl.sh` applies its built-in defaults, and finally any trailing Hydra overrides are passed directly to `verl.trainer.main_ppo`.

Continue from the latest checkpoint for the same experiment:

```bash
bash scripts/train_recbench_rl.sh \
  --env-file scripts/configs/recbench_movie_k10_hybrid.env \
  trainer.resume_mode=auto
```

Train the book domain by copying the env file and changing only `TRAIN_FILE`, `VAL_FILE`, and `EXPERIMENT_NAME`, or by overriding them at launch:

```bash
TRAIN_FILE=../data/recbench/processed/rl/book_train.parquet \
VAL_FILE=../data/recbench/processed/rl/book_test.parquet \
EXPERIMENT_NAME=recbench-book-rl-k10-hybrid-v2 \
bash scripts/train_recbench_rl.sh \
  --env-file scripts/configs/recbench_movie_k10_hybrid.env \
  trainer.resume_mode=disable
```

For a local smoke test without W&B:

```bash
TRAINER_LOGGER='["console"]' \
WANDB_MODE=disabled \
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=4 \
ROLLOUT_N=2 \
TOTAL_EPOCHS=1 \
TEST_FREQ=1 \
bash scripts/train_recbench_rl.sh \
  --env-file scripts/configs/recbench_movie_k10_hybrid.env \
  trainer.resume_mode=disable
```

Useful training environment variables:

```bash
MODEL_PATH=/path/to/base-instruct-model
TRAIN_FILE=../data/recbench/processed/rl/movie_train.parquet
VAL_FILE=../data/recbench/processed/rl/movie_test.parquet
PROJECT_NAME=faithrec
EXPERIMENT_NAME=recbench-movie-rl-faithrl-gdpo
WANDB_ENTITY=your-wandb-entity
TRAINER_LOGGER='["console","wandb"]'
LOG_VAL_GENERATIONS=8
ROLLOUT_N=8
ACTOR_LR=1e-6
REWARD_MODE=faithrl_hybrid
REWARD_K=10
ADV_ESTIMATOR=gdpo
GDPO_REWARD_KEYS="['recommendation','faithfulness']"
GDPO_REWARD_WEIGHTS="[1.0,0.5]"
BASELINE_CORRECT_RATE=0.5
BASELINE_UNFAITHFUL_RATE=0.5
```

After a base-model validation dump, estimate calibration constants with:

```powershell
python scripts\calibrate_recbench_faithrl.py path\to\validation_dump.jsonl
```

Then rerun training with the printed `BASELINE_CORRECT_RATE` and `BASELINE_UNFAITHFUL_RATE`. The default `REWARD_MODE=faithrl_hybrid` adds a small weighted shaping term on top of the FaithRL-style geometric reward so early runs do not collapse to a completely flat scalar reward when every sample is still unfaithful. Set `REWARD_MODE=faithrl` for the pure geometric reward; `HYBRID_WEIGHT=0.1` is the default hybrid weight.

The RL reward hook is `verl/verl/utils/reward_score/recbench_json.py`. It evaluates the final JSON answer for ranking quality, format validity, evidence-reference validity, structured rationale validity, and FaithRL-style outcome labels.

## 3. Optional SFT Warm-Start Ablation

SFT is not part of the main pipeline. Use it only as an ablation or fallback if the base model's initial JSON/rationale parse rate is too low for stable RL.

Build deterministic SFT targets for output-format learning and top-1 warm start. The positive candidate IDs are used only by the builder; they are not included in the student prompt. The selected positive candidates are placed first, and remaining candidates keep their original A/B/C order to avoid supervising arbitrary negative ordering.

```powershell
python teacher\build_sft.py `
  --input data\recbench\processed\canonical\movie_train.jsonl `
  --output data\recbench\processed\sft\movie_train.parquet
```

Run the same command for `movie_test.jsonl`, `book_train.jsonl`, and `book_test.jsonl` when you need validation files or both domains.

Accepted rows require the positive item at top-1, valid candidate-only ranking, valid evidence references, and valid structured rationale by default.

An OpenAI-compatible teacher remains available for ablations, but it is not the default because token-level SFT would otherwise learn noisy teacher ordering among negative candidates:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"

python teacher\build_sft.py `
  --target-mode teacher `
  --input data\recbench\processed\canonical\movie_train.jsonl `
  --output data\recbench\processed\sft\movie_train.teacher.parquet `
  --model gpt-4.1-mini
```

If you run this ablation, train the SFT checkpoint from inside `verl`:

```bash
cd verl

MODEL_PATH=/path/to/base-model \
TRAIN_FILE=../data/recbench/processed/sft/movie_train.parquet \
VAL_FILE=../data/recbench/processed/sft/movie_test.parquet \
bash scripts/train_recbench_sft.sh
```

Then optionally continue RL from that SFT checkpoint as a separate `SFT -> RL` ablation:

```bash
cd verl

MODEL_PATH=/path/to/sft-checkpoint \
TRAIN_FILE=../data/recbench/processed/rl/movie_train.parquet \
VAL_FILE=../data/recbench/processed/rl/movie_test.parquet \
bash scripts/train_recbench_rl.sh
```

## Notes

- `data/recbench/books.dat` and `data/recbench/movies.dat` are the item metadata files used by this project.
- `scripts/prepare_recbench.py` reads query files from the cloned RecBench+ repository and item metadata from `data/recbench`.
- If the local RecBench+ clone has fewer available rows than `--train-size + --test-size`, the script writes all available rows after the train split.
