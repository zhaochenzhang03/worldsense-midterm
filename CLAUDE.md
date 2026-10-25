# CLAUDE.md

This is a **CMU 11-777 Multimodal Machine Learning research project** that studies
modality necessity and substitutability on the WorldSense benchmark. Read
`README.md` and `configs/experiment_manifest.json` for the full protocol.

## Research protocol: hard rules

- **Never silently change the research protocol.** Ask before changing the
  model, prompts, conditions, seed, split, or metrics.
- **Never regenerate or alter the frozen dev/test split** (`configs/dev_ids.json`,
  `configs/test_ids.json`) without explicit permission. Do not run
  `scripts/create_split.py --force` on your own initiative.
- **Never tune prompts on test.** Prompt and parser development uses dev only.
- **Never use `video_caption` as model input**, in any prompt or condition. It is
  metadata only and leaks video content.
- Subtitle conditions use the **official WorldSense subtitles** only, never ASR or
  model-generated captions. Call this input the "official subtitle", not a
  speech transcript: it can contain tags such as `[Music]`.
- Subtitle normalization (`src/subtitle_utils.py:normalize_subtitle_text`) is
  **frozen**. It removes sequence numbers and timestamps and removes only
  *consecutive* exact duplicate lines. Text that recurs at different times is
  kept, with no file-wide dedupe. Bracketed non-speech tags are kept. Never summarize, paraphrase, translate or replace subtitle text. Do not
  change this function without permission. `python -m scripts.inspect_subtitles`
  checks the processed output against the manifest digest.
- **Media preprocessing is frozen** (`configs/experiment_manifest.json` →
  `media_preprocessing`). Originals in `data/worldsense/videos/` are never
  modified. `video_only` (.mp4) and `audio_only` (.m4a) are stream copies with
  no re-encoding. `video_subtitle` reuses the `video_only` file plus the
  processed subtitle as text, never burned in. Do not change codecs,
  containers, ffmpeg flags or the ffmpeg version without permission. Run
  `python -m scripts.validate_media` and `python -m scripts.validate_variants`
  after any media change.
- ffmpeg lives in the conda env `worldsense-ffmpeg`. Set
  `WORLDSENSE_FFMPEG_DIR=~/miniforge3/envs/worldsense-ffmpeg/bin`. The system
  Homebrew is too old for this macOS.
- After touching anything split-related, run `python -m scripts.validate_split`.
  It must pass.

## API evaluation requirements

- Every API evaluation must **support resume**: skip (condition, uid) pairs
  that already have saved results.
- **Save every completed API call immediately** (e.g. append + flush one JSONL
  record per call), not only at the end of a run.
- **Preserve raw model responses** alongside parsed answers, prompts, model
  identifiers, and timestamps.
- Analysis must **run offline from saved outputs** whenever possible, without new
  API calls.
- Do not make API calls or download media unless the user asked for it.

## Conventions

- Run scripts from the repo root as modules: `python -m scripts.<name>`.
- `data/` and `outputs/` are gitignored. Never commit raw dataset files or model outputs.
- The current code uses only the standard library. Do not add dependencies without need.
