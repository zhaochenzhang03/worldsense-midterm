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

The Python code uses only the standard library (Python >= 3.10; developed on
3.14). The media scripts also need ffmpeg/ffprobe (see "Modality variants" below). Run commands from the repository root:

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
