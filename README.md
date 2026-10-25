# WorldSense Modality Necessity & Substitutability

CMU 11-777 (Multimodal Machine Learning) midterm project.

## Purpose

We study how much each input modality contributes to omnimodal video question
answering on the [WorldSense](https://arxiv.org/abs/2502.04326) benchmark.

**Research question — modality necessity and substitutability:**
- *Necessity*: how much does accuracy drop when a modality (visual, audio) is removed?
- *Substitutability*: can a textual stand-in (the official subtitle) recover the
  performance lost when raw audio is removed?

Every condition is evaluated on the **same fixed set of WorldSense examples**, so
differences between conditions come from the input modalities alone.

## Dataset

WorldSense is a multiple-choice benchmark of real-world videos with
visual, audio, and text information. Each video has one or more QA tasks
(`task0`, `task1`, ...), labeled with 26 task types in 3 task domains
(Recognition, Understanding, Reasoning).

`src/worldsense_data.py` loads `data/worldsense/worldsense_qa.json` and flattens
every QA task into its own example with UID `"{video_id}::{task_id}"`.

## Input conditions

| # | ID | Model input |
|---|----|-------------|
| 1 | `full` | video + raw audio |
| 2 | `video_only` | video with audio removed |
| 3 | `audio_only` | raw audio only |
| 4 | `video_subtitle` | silent video + official WorldSense subtitle |
| 5 | `subtitle_only` | official WorldSense subtitle only |
| 6 | `question_only` | question + answer choices only |

Every condition also gets the question and answer choices. Subtitles are the
**official WorldSense-provided subtitles**, not ASR transcripts or model-generated
captions.

The planned primary baseline model is Gemini 2.5 Flash-Lite through the CMU AI
Gateway. It is not implemented yet.

## Evaluation protocol (frozen)

- 60 examples total: **12 dev / 48 test**
- Only videos **<= 120 seconds**
- **One QA per video**, so the split is **by video** (no video appears in both dev and test)
- All **26 task types** are covered across the 60 examples
- Random seed **42**
- Selection algorithm: `scripts/create_split.py`

The split in `configs/dev_ids.json` and `configs/test_ids.json` is the
**canonical frozen split**. It is not rebalanced by domain, audio class, or
answer letter. `configs/experiment_manifest.json` records SHA-256 hashes of the
dataset file and both split files. `scripts/create_split.py` refuses to
overwrite existing split files unless you pass `--force`, which you should only
do with explicit agreement from the team.

**The test set must never be used for prompt tuning.** Prompts and parsing
are developed on dev only. Test is run once, for final numbers.

### `video_caption` is forbidden as model input

The WorldSense annotations include a `video_caption` field, a text description
of the video content. It is kept as metadata only and **must never appear in any
prompt or model input**: it would leak visual/audio content and invalidate the
modality comparisons (especially `question_only` and `subtitle_only`).

## Setup and usage

The current code uses only the Python standard library (Python >= 3.10;
developed on 3.14). Run commands from the repository root:

```bash
python -m scripts.inspect_worldsense   # dataset statistics
python -m scripts.validate_split       # check protocol invariants + manifest hashes
```

`validate_split` exits with a nonzero status if any invariant fails.

## Repository layout

```
src/worldsense_data.py          WorldSense loader (flattens QA tasks)
scripts/inspect_worldsense.py   dataset statistics
scripts/create_split.py         split generation (guarded; frozen output)
scripts/validate_split.py       protocol validation
configs/dev_ids.json            frozen dev split (12)
configs/test_ids.json           frozen test split (48)
configs/experiment_manifest.json  hashes, protocol parameters, conditions
```

## Data and outputs

Raw dataset files (`data/`) and model outputs (`outputs/`) are **gitignored**.
Place the WorldSense QA file at `data/worldsense/worldsense_qa.json`, then run
`python -m scripts.validate_split`. It checks that the file's SHA-256 matches the
manifest.

Future model evaluation will save every API response to disk immediately and
support resuming, so analysis can be rerun offline without new API calls.
