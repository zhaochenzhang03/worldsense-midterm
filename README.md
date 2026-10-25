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
**official WorldSense-provided subtitles**. No ASR is run and no captions are
generated. We call this input the "official subtitle" rather than a speech
transcript, because the files can contain non-speech annotations such as
`[Music]`.

The primary baseline model is Gemini 2.5 Flash-Lite through the CMU AI Gateway
(Gateway model ID `gemini/gemini-2.5-flash-lite`).

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

The Python code uses only the standard library (Python >= 3.10; developed on
3.14). The media scripts also need ffmpeg/ffprobe (see "Modality variants" below). Run commands from the repository root:

```bash
python -m scripts.inspect_worldsense   # dataset statistics
python -m scripts.validate_split       # check protocol invariants + manifest hashes
```

`validate_split` exits with a nonzero status if any invariant fails.

## Evaluation harness

The frozen settings are in `configs/eval_config.json`:
- model: Gemini 2.5 Flash-Lite via the CMU AI Gateway
- temperature 0, `max_tokens` 512, prompt `v1`
- the six conditions
- resume on, with results saved immediately

### Final test protocol

The final protocol was frozen before the first TEST request and is recorded in
`configs/final_test_protocol.json`: model, temperature 0, `max_tokens` 512
(added before TEST because one uncapped DEV response ran to the Gateway's
65,536-token limit), prompt `v1`, the six conditions, the parser, the 48-example
test split (288 pairs) and SHA-256 hashes of the protocol files (config, splits,
media manifests, prompt builder, parser, transport, runner).
`python -m scripts.freeze_test_protocol` re-checks the hashes, and
`validate_evaluation` runs the same check. DEV (N=12) served as protocol
development and pilot; it is analyzed separately and never pooled with TEST.

A completion that arrives without text (e.g. blocked, or cut off before any
text) is saved as an empty `raw_response`, parsed as `invalid` and counted as
incorrect; it is a model outcome and is never re-sent. Responses that hit
`max_tokens` (`finish_reason` `length`) are parsed like any other.

**Gateway transport** (`CMUGatewayClient` in `src/model_client.py`): one
`POST /v1/chat/completions` per request, `temperature` 0, `max_tokens` 512, one user message with
the media parts first and then the `v1` prompt text. Media is sent inline as
OpenAI/LiteLLM `file` content parts,
`{"type": "file", "file": {"file_data": "data:<mime>;base64,...", "format": "<mime>"}}`
(`ChatCompletionFileObject` in the Gateway's `/openapi.json`). MP4s go as
`video/mp4`; the `audio_only` M4A goes as `audio/m4a` and is accepted unchanged,
so no audio container conversion is needed. The Gateway's `input_audio` part
only allows wav/mp3 and is not used. The API key comes only from
`CMU_AI_GATEWAY_API_KEY`, is never logged or saved, and is redacted from error
text. The full provider response JSON (including token usage) is saved under
`request_metadata.response_json`.

- `src/prompting.py`: one prompt builder for every condition. The instruction,
  question and choices are identical across conditions. The subtitle conditions
  add a `Subtitles:` block with the normalized official subtitle. The prompt
  never names the condition or a missing modality, and never uses
  `video_caption`.
- `src/answer_parser.py`: maps raw responses to a letter with `parse_status`
  `exact`, `normalized`, `ambiguous` or `invalid`. It never guesses between
  conflicting letters. A LaTeX `\boxed{X}` final answer, which Gemini often
  writes after an explanation, is taken as the answer (several different boxed
  letters: `ambiguous`). Tests: `python -m unittest tests.test_answer_parser`.
- `scripts/run_evaluation.py`: runs on dev by default.
  - `--split test` exits before any API call unless `--allow-test` is also given.
  - Each call is appended to `outputs/<split>/<condition>.jsonl` and fsynced
    right away.
  - Pairs `(uid, condition, model, prompt_version)` that already succeeded are
    skipped, and successes are never duplicated. Failed pairs from earlier runs
    are re-attempted only with `--retry-failed`.
  - Within a run, a request is re-sent (up to `transient_retries` = 2 times,
    after 15 s / 60 s) only after a transient failure that returned no
    completion: HTTP 408/429/5xx, timeout or connection error. Every attempt is
    saved.
  - `--workers N` (1-5) sends up to N requests concurrently; appends are
    serialized.
- `scripts/validate_evaluation.py`: offline checks on saved outputs:
  - only frozen UIDs, in the correct split
  - no duplicate successes
  - prompts match the frozen `v1` prompt
  - no `video_caption`, no API key
  - valid parse statuses
  - test outputs only together with a formal-run marker
  - test records use exactly the frozen model, prompt version, temperature and
    `max_tokens`
  - protocol hashes in `configs/final_test_protocol.json` are unchanged
  - per split and condition: number of pairs with a successful response
  - stored parses match the current parser on the saved raw response
- `scripts/reparse_outputs.py`: after a parser change, re-derives
  `parsed_prediction`, `parse_status` and `correct` from the saved
  `raw_response` (offline; `--write` to rewrite atomically).
- `scripts/freeze_test_protocol.py`: `--write` records the frozen test
  protocol and its file hashes (once, before TEST); without flags it checks them.
- `scripts/analyze_results.py --split {dev,test}`: offline analysis (no API
  calls, saved records never modified) into `outputs/analysis/<split>/`:
  `main_results.csv`, `modality_gaps.json`, `statistical_tests.csv` and
  `bootstrap_results.json` (paired bootstrap, 10,000 resamples, seed 42, plus
  exact McNemar for V+A vs V, A, V+S, S, Q), `item_behavior.csv` (overlapping
  behavior labels), `conditional_necessity.csv`, `subgroup_results.csv`,
  `subtitle_group_results.csv`, `task_domain_results.csv`,
  `task_type_results.csv`, `prediction_agreement.csv`,
  `prediction_letter_distribution.csv`, `parse_status_by_condition.csv`,
  `behavior_cases.md`, `dev_vs_test.csv`, `<split>_summary.md` and SVG figures.
  Missing or unparsed answers count as incorrect.

```bash
python -m scripts.run_evaluation --all-conditions --limit 2 --dry-run   # no API calls
python -m scripts.run_evaluation --condition full --uids <dev uid>        # real call (dev)
python -m scripts.run_evaluation --all-conditions --workers 4             # full dev run, resumable
python -m scripts.validate_evaluation
python -m scripts.analyze_results --split dev                             # offline
python -m scripts.freeze_test_protocol                                    # check frozen test protocol hashes
python -m scripts.run_evaluation --split test --allow-test --all-conditions --workers 4   # final test run
python -m scripts.analyze_results --split test                            # offline final analysis
```

## Repository layout

```
src/worldsense_data.py          WorldSense loader (flattens QA tasks)
scripts/inspect_worldsense.py   dataset statistics
scripts/create_split.py         split generation (guarded; frozen output)
scripts/validate_split.py       protocol validation
src/subtitle_utils.py           frozen subtitle normalization
scripts/process_subtitles.py    writes normalized subtitles for the frozen 60
scripts/inspect_subtitles.py    subtitle coverage + integrity checks
scripts/map_video_archives.py   locate frozen videos in the zip archives
scripts/download_frozen_videos.py  range-download the frozen 60 videos
scripts/validate_media.py       original media integrity checks
src/media_tools.py              ffmpeg/ffprobe helpers
scripts/inspect_media_streams.py  record source codec metadata
scripts/create_modality_variants.py  video_only / audio_only by stream copy
scripts/validate_variants.py    variant integrity checks
configs/video_archive_map.json  video_id -> archive
configs/media_manifest.json     original media hashes
configs/media_stream_metadata.json  source stream metadata
configs/variant_manifest.json   variant hashes, durations, codecs
configs/eval_config.json        frozen evaluation settings (no secrets)
src/prompting.py                single prompt builder for all conditions
src/answer_parser.py            response -> letter + parse_status
src/model_client.py             provider abstraction (Gateway isolated)
scripts/run_evaluation.py       resumable, immediately-saving runner
scripts/validate_evaluation.py  offline checks on saved outputs
scripts/reparse_outputs.py      re-parse saved raw responses (offline)
scripts/analyze_results.py      offline analysis, tables, figures
scripts/freeze_test_protocol.py record / check the frozen test protocol
configs/final_test_protocol.json  frozen test protocol + file hashes
tests/test_answer_parser.py     parser regression tests
configs/dev_ids.json            frozen dev split (12)
configs/test_ids.json           frozen test split (48)
configs/experiment_manifest.json  hashes, protocol parameters, conditions
```

## Data and outputs

Raw dataset files (`data/`) and model outputs (`outputs/`) are **gitignored**.
Place the WorldSense QA file at `data/worldsense/worldsense_qa.json`, then run
`python -m scripts.validate_split`. It checks that the file's SHA-256 matches the
manifest.

### Media (frozen 60 videos only)

`scripts/download_frozen_videos.py` uses HTTP range requests to pull only the 60
frozen `.mp4` files out of the official `worldsense_videos_*.zip` archives
(about 300 MB instead of 18 GB). It checks each file against its zip CRC-32,
writes it atomically, can resume an interrupted run, and records each file's
SHA-256 in `configs/media_manifest.json`. The archive containing each video is
listed in `configs/video_archive_map.json`.

```bash
python -m scripts.download_frozen_videos
python -m scripts.validate_media
```

### Modality variants (frozen)

ffmpeg/ffprobe come from an isolated conda env. The exact version is recorded in
`configs/experiment_manifest.json` under `media_preprocessing.ffmpeg`.

```bash
conda create -n worldsense-ffmpeg -c conda-forge --override-channels ffmpeg
export WORLDSENSE_FFMPEG_DIR=~/miniforge3/envs/worldsense-ffmpeg/bin   # or activate the env

python -m scripts.inspect_media_streams       # -> configs/media_stream_metadata.json
python -m scripts.create_modality_variants    # --split dev|test|all, --force
python -m scripts.validate_variants
```

All 60 originals have one H.264 (High) video stream and one AAC-LC 44.1 kHz
stereo audio stream. Both variants are made by **stream copy only**, with no
re-encoding:

| Variant | File | Contents |
|---|---|---|
| `video_only` | `data/worldsense/variants/video_only/<video_id>.mp4` | original video stream, audio removed |
| `audio_only` | `data/worldsense/variants/audio_only/<video_id>.m4a` | original AAC stream, same sample rate and channels |

Metadata, chapters, subtitle and data streams are dropped. Bitexact flags
make re-runs produce byte-identical files. `validate_variants` checks:
- the file sets and the hashes in `configs/variant_manifest.json`,
- stream presence and absence,
- codecs,
- container and stream duration against the original (tolerance 0.1 s),
- that the copied stream is **packet-identical** to the original,
- that the originals are unchanged.

`create_modality_variants` resumes by skipping outputs that already match the
manifest. It only overwrites them with `--force`.

How each condition is assembled at inference time (all conditions also get the
question and answer choices; none ever gets `video_caption`):

| Condition | Inputs |
|---|---|
| `full` | original `data/worldsense/videos/<video_id>.mp4` |
| `video_only` | `variants/video_only/<video_id>.mp4` |
| `audio_only` | `variants/audio_only/<video_id>.m4a` |
| `video_subtitle` | `variants/video_only/<video_id>.mp4` + `processed_subtitles/<video_id>.txt` as text (not burned into frames) |
| `subtitle_only` | `processed_subtitles/<video_id>.txt` as text |
| `question_only` | nothing else |

### Official subtitles

The subtitle conditions use the official WorldSense subtitle archive. The source
URL and SHA-256 are recorded in `configs/experiment_manifest.json` under
`subtitles`. To obtain the subtitles:

```bash
mkdir -p data/worldsense/raw data/worldsense/subtitles
curl -fL -o data/worldsense/raw/worldsense_subtitles.zip \
  https://huggingface.co/datasets/honglyhly/WorldSense/resolve/main/worldsense_subtitles.zip
shasum -a 256 data/worldsense/raw/worldsense_subtitles.zip   # compare with manifest
unzip -q data/worldsense/raw/worldsense_subtitles.zip -d data/worldsense/subtitles
python -m scripts.process_subtitles   # writes data/worldsense/processed_subtitles/<video_id>.txt
python -m scripts.inspect_subtitles
```

The archive is flat: one `<video_id>.srt` file per WorldSense video.

**Subtitle normalization (frozen).** Model input uses the output of
`src/subtitle_utils.py:normalize_subtitle_text`, which:
- removes SRT sequence numbers and timestamps (a numeric line counts as a
  sequence number only when the next line is a timestamp, so subtitle text such
  as `2001` is kept),
- removes a line only when it exactly repeats the line directly before it
  (rolling captions). Identical text that comes back later in the video, such
  as a second `[Music]`, is **kept**. There is no file-wide deduplication,
- keeps all other text exactly as written, **including bracketed non-speech
  tags** such as `[Music]` and `[Applause]`.

It never summarizes, paraphrases, translates or fills in missing content, and it
never falls back to `video_caption`. Of the frozen 60, 53 subtitles contain real
text and 7 contain only `[Music]`; all 7 are videos without `Speech`.

`inspect_subtitles` reports, for each example: whether the raw file was found,
the normalized status (usable, tags-only, empty, missing), the character count,
and whether the video has `Speech`. It checks that every processed file maps to
exactly one frozen `video_id`, matches a fresh normalization, and matches the
digest recorded in the manifest. It exits nonzero only for structural problems,
not because a non-speech video has no subtitle text.

Future model evaluation will save every API response to disk immediately and
support resuming, so analysis can be rerun offline without new API calls.
