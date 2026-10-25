"""
Offline analysis of saved evaluation outputs. Makes no API calls and
never modifies saved records.

Reads outputs/<split>/<condition>.jsonl (one successful record per
(uid, condition) is expected; see scripts/validate_evaluation.py) and
writes report tables, figures and a summary to outputs/analysis/<split>/.

A missing result or an unparsed answer (ambiguous / invalid) counts as
incorrect, so every accuracy uses all frozen examples of the split as
its denominator. DEV and TEST are always analyzed separately and never
pooled.
"""

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

from scripts.run_evaluation import EVAL_CONFIG, SPLIT_FILES, is_success, load_jsonl
from src.prompting import CONDITIONS, PROMPT_VERSION, load_subtitle


BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42

SHORT = {
    "full": "V+A",
    "video_only": "V",
    "audio_only": "A",
    "video_subtitle": "V+S",
    "subtitle_only": "S",
    "question_only": "Q",
}

# name: (condition with the modality, condition without it)
GAPS = {
    "delta_audio": ("full", "video_only"),
    "delta_visual": ("full", "audio_only"),
    "delta_raw_audio": ("full", "video_subtitle"),
}

# Paired comparisons with bootstrap CI and McNemar test.
COMPARISONS = [
    ("full", "video_only"),
    ("full", "audio_only"),
    ("full", "video_subtitle"),
    ("full", "subtitle_only"),
    ("full", "question_only"),
]

# Per-item behavior labels. They overlap: one item can carry several.
BEHAVIORS = {
    "vision_sufficient": (
        "full correct AND video_only correct",
        lambda c: c["full"] and c["video_only"],
    ),
    "audio_sufficient": (
        "full correct AND audio_only correct",
        lambda c: c["full"] and c["audio_only"],
    ),
    "subtitle_substitutable": (
        "full correct AND video_subtitle correct",
        lambda c: c["full"] and c["video_subtitle"],
    ),
    "raw_audio_dependent_candidate": (
        "full correct AND video_subtitle incorrect",
        lambda c: c["full"] and not c["video_subtitle"],
    ),
    "genuine_multimodal_candidate": (
        "full correct AND video_only incorrect AND audio_only incorrect",
        lambda c: c["full"] and not c["video_only"] and not c["audio_only"],
    ),
    "strong_genuine_multimodal_candidate": (
        "full correct AND video_only, audio_only and video_subtitle incorrect",
        lambda c: c["full"] and not c["video_only"] and not c["audio_only"] and not c["video_subtitle"],
    ),
    "negative_audio_interference": (
        "video_only correct AND full incorrect",
        lambda c: c["video_only"] and not c["full"],
    ),
    "negative_visual_interference": (
        "audio_only correct AND full incorrect",
        lambda c: c["audio_only"] and not c["full"],
    ),
    "subtitle_help": (
        "video_subtitle correct AND video_only incorrect",
        lambda c: c["video_subtitle"] and not c["video_only"],
    ),
    "subtitle_hurt": (
        "video_only correct AND video_subtitle incorrect",
        lambda c: c["video_only"] and not c["video_subtitle"],
    ),
    "question_only_shortcut": (
        "question_only correct",
        lambda c: c["question_only"],
    ),
}

# Among examples answered correctly with full input:
# name: (description, rule over per-condition correctness)
CONDITIONAL_NECESSITY = {
    "also_correct_V": ("also correct with V", lambda c: c["video_only"]),
    "also_correct_A": ("also correct with A", lambda c: c["audio_only"]),
    "also_correct_VS": ("also correct with V+S", lambda c: c["video_subtitle"]),
    "requires_raw_audio": ("V+S wrong (raw audio needed beyond subtitle)", lambda c: not c["video_subtitle"]),
    "requires_both_modalities": ("V wrong AND A wrong", lambda c: not c["video_only"] and not c["audio_only"]),
    "solvable_by_some_reduced_input": (
        "correct with at least one of V, A, V+S, S, Q",
        lambda c: any(c[x] for x in CONDITIONS if x != "full"),
    ),
}

# Mechanical refusal / inability detector over raw responses (a
# keyword heuristic, not a judgment of intent). Reported separately for
# unparsed and parsed responses because a response can mention missing
# information and still commit to a letter.
REFUSAL_RE = re.compile(
    r"\bI (?:cannot|can't|can not|am unable|'m unable|am not able|do not have|don't have)\b"
    r"|\bunable to (?:determine|answer|identify|tell|hear|see|view|access|process|watch|listen|analy[sz]e)\b"
    r"|\bnot possible to (?:determine|answer|tell)\b"
    r"|\b(?:cannot|can't|can not) be (?:determined|answered|identified)\b"
    r"|\bimpossible to (?:determine|answer|tell|say|identify|know|analy[sz]e)\b"
    r"|\b(?:does not|doesn't|do not|don't) (?:contain|provide|include|have|describe|mention)\b"
    r"[^.]{0,60}\b(?:information|details|context)\b"
    r"|\benough (?:visual |auditory )?information\b|\binsufficient\b|\bunanswerable\b"
    r"|\bplease provide\b"
    r"|\bno (?:audio|video|visual|sound|image)s? (?:is |was |has been )?(?:provided|available|included|attached)\b"
    r"|\b(?:is|was|were|are) not provided\b",
    re.IGNORECASE,
)

_TAG_RE = re.compile(r"\[[^\]]*\]")

# Reference palette (dataviz skill, light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]
BLUE_RAMP = ["#e8f1fc", "#c6dcf6", "#9cc2ee", "#6aa3e4", "#3d85d9", "#2a6fc0", "#1f5aa0", "#16447a"]


def parse_args():
    parser = argparse.ArgumentParser(description="Offline analysis of saved results.")
    parser.add_argument("--split", choices=list(SPLIT_FILES), default="dev")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Default: outputs/analysis/<split>/",
    )
    return parser.parse_args()


def subtitle_kind(video_id: str) -> str:
    """
    "linguistic" if the normalized official subtitle has any text left
    after removing bracketed tags such as [Music] or [ __ ], else
    "tags_only".
    """
    remainder = _TAG_RE.sub("", load_subtitle(video_id))
    return "linguistic" if re.search(r"\w", remainder) else "tags_only"


def load_items(config: dict, split: str):
    """
    One item per frozen example of the split, with the successful
    record per condition (None if missing). Returns (items, failed API
    attempts per condition, error messages).
    """
    with open(SPLIT_FILES[split], "r", encoding="utf-8") as f:
        examples = json.load(f)

    model = config["model"]["model_id"]
    split_dir = Path(config["output_dir"]) / split
    results = {}
    api_failures = {c: 0 for c in CONDITIONS}
    errors = []

    for condition in CONDITIONS:
        for record in load_jsonl(split_dir / f"{condition}.jsonl"):
            if record["model"] != model or record["prompt_version"] != PROMPT_VERSION:
                continue
            if not is_success(record):
                api_failures[condition] += 1
                errors.append(record["error"] or "")
                continue
            key = (record["uid"], condition)
            if key in results:
                raise SystemExit(f"ERROR: duplicate successful result {key}")
            results[key] = record

    items = []
    for example in examples:
        item = {
            "example": example,
            "subtitle_kind": subtitle_kind(example["video_id"]),
            "speech": "Speech" in example["audio_class"],
            "records": {},
            "prediction": {},
            "parse_status": {},
            "correct": {},
        }
        for condition in CONDITIONS:
            record = results.get((example["uid"], condition))
            item["records"][condition] = record
            item["prediction"][condition] = record["parsed_prediction"] if record else None
            item["parse_status"][condition] = record["parse_status"] if record else "missing"
            item["correct"][condition] = bool(record and record["correct"])
        item["behaviors"] = [
            name for name, (_, rule) in BEHAVIORS.items() if rule(item["correct"])
        ]
        items.append(item)

    return items, api_failures, errors


def accuracy(items: list[dict], condition: str) -> float | None:
    if not items:
        return None
    return sum(item["correct"][condition] for item in items) / len(items)


def mean_gap(items: list[dict], with_condition: str, without_condition: str) -> float | None:
    if not items:
        return None
    return sum(
        item["correct"][with_condition] - item["correct"][without_condition]
        for item in items
    ) / len(items)


def percentile_ci(values: list[float], resamples: int, rng: random.Random) -> tuple[float, float]:
    """Percentile 95% CI of the mean of values under resampling."""
    n = len(values)
    stats = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(resamples)
    )
    return stats[int(0.025 * resamples)], stats[int(0.975 * resamples) - 1]


def paired_bootstrap(items, with_condition, without_condition):
    """
    Resample examples with replacement (paired: both conditions of an
    example move together). Percentile 95% CI.
    """
    diffs = [
        item["correct"][with_condition] - item["correct"][without_condition]
        for item in items
    ]
    low, high = percentile_ci(diffs, BOOTSTRAP_RESAMPLES, random.Random(BOOTSTRAP_SEED))
    return {
        "observed_difference": sum(diffs) / len(diffs),
        "ci95_low": low,
        "ci95_high": high,
        "n": len(diffs),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def mcnemar_exact(items, with_condition, without_condition):
    """
    Two-sided exact McNemar test on the discordant pairs:
    b = correct only with, c = correct only without.
    """
    b = sum(item["correct"][with_condition] and not item["correct"][without_condition] for item in items)
    c = sum(item["correct"][without_condition] and not item["correct"][with_condition] for item in items)
    n = b + c
    if n == 0:
        p = 1.0
    else:
        p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"only_with_correct": b, "only_without_correct": c, "p_value_two_sided": p}


def fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def signed(x, digits=3):
    return "n/a" if x is None else f"{x:+.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finish_reason(record: dict | None) -> str | None:
    return ((record or {}).get("request_metadata") or {}).get("finish_reason")


def is_refusal_like(record: dict | None) -> bool:
    return bool(record and record["raw_response"] and REFUSAL_RE.search(record["raw_response"]))


def group_row(group_type: str, group: str, members: list[dict], with_ci: bool = False) -> dict:
    row = {"group_type": group_type, "group": group, "n": len(members)}
    for c in CONDITIONS:
        row[f"correct_{SHORT[c]}"] = sum(i["correct"][c] for i in members)
    for c in CONDITIONS:
        row[f"acc_{SHORT[c]}"] = round(accuracy(members, c), 4)
    for name, (with_c, without_c) in GAPS.items():
        row[name] = round(mean_gap(members, with_c, without_c), 4)
    if with_ci:
        b = paired_bootstrap(members, "full", "video_subtitle")
        m = mcnemar_exact(members, "full", "video_subtitle")
        row["delta_raw_audio_ci95_low"] = round(b["ci95_low"], 4)
        row["delta_raw_audio_ci95_high"] = round(b["ci95_high"], 4)
        row["delta_raw_audio_mcnemar_p"] = round(m["p_value_two_sided"], 4)
        row["delta_raw_audio_discordant"] = f"{m['only_with_correct']}/{m['only_without_correct']}"
    return row


# ---------------------------------------------------------------------
# Figures: minimal static SVG (standard library only). Each mark has a
# <title> so viewers show its value on hover.
# ---------------------------------------------------------------------

def _svg_open(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{INK}">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="16" y="24" font-size="14" font-weight="bold">{title}</text>',
    ]


def _bar_path(x, y_top, w, h, up=True, r=4):
    """Bar with its data end rounded (r px), the baseline end square."""
    r = min(r, w / 2, h)
    if h <= 0:
        return ""
    if up:
        return (f"M{x:.1f},{y_top + h:.1f} V{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} "
                f"H{x + w - r:.1f} Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f} V{y_top + h:.1f} Z")
    return (f"M{x:.1f},{y_top:.1f} V{y_top + h - r:.1f} Q{x:.1f},{y_top + h:.1f} {x + r:.1f},{y_top + h:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{y_top + h:.1f} {x + w:.1f},{y_top + h - r:.1f} V{y_top:.1f} Z")


def svg_bar_chart(path, title, labels, values, y_min, y_max, y_label,
                  errors=None, series=None, reference=None, value_format="{:.2f}"):
    """
    Vertical bar chart. values is a list of numbers, or, with series,
    a list of per-series lists (grouped bars). errors: per-bar (lo, hi)
    whiskers (single series only). reference: (value, label) dashed line.
    """
    width, height = 680, 380
    left, right, top, bottom = 64, 16, 56 if series else 44, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    groups = values if series else [values]

    def y(v):
        return top + plot_h * (y_max - v) / (y_max - y_min)

    out = _svg_open(width, height, title)

    ticks = 4
    for i in range(ticks + 1):
        v = y_min + (y_max - y_min) * i / ticks
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end" fill="{INK_2}">{v:.2f}</text>')
    out.append(f'<line x1="{left}" x2="{width - right}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="{INK_2}"/>')
    out.append(
        f'<text transform="translate(16,{top + plot_h / 2}) rotate(-90)" text-anchor="middle" fill="{INK_2}">{y_label}</text>'
    )
    if reference:
        rv, rl = reference
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y(rv):.1f}" y2="{y(rv):.1f}" '
                   f'stroke="{INK_2}" stroke-dasharray="4 4"/>')
        out.append(f'<text x="{width - right - 4}" y="{y(rv) + 14:.1f}" text-anchor="end" fill="{INK_2}">{rl}</text>')

    slot = plot_w / len(labels)
    gap = 2
    bar_w = (slot * 0.7 - gap * (len(groups) - 1)) / len(groups)

    for g, group in enumerate(groups):
        for i, v in enumerate(group):
            if v is None:
                continue
            x = left + slot * i + slot * 0.15 + (bar_w + gap) * g
            y0, y1 = sorted((y(0), y(v)))
            name = series[g] + ", " if series else ""
            tip = f"{name}{labels[i].splitlines()[0]}: {value_format.format(v)}"
            if errors and not series:
                tip += f" (95% CI {value_format.format(errors[i][0])} to {value_format.format(errors[i][1])})"
            out.append(
                f'<path d="{_bar_path(x, y0, bar_w, y1 - y0, up=v >= 0)}" fill="{SERIES[g]}"><title>{tip}</title></path>'
            )
            if errors and not series:
                lo, hi = errors[i]
                cx = x + bar_w / 2
                out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}" stroke="{INK}"/>')
                for e in (lo, hi):
                    out.append(f'<line x1="{cx - 5:.1f}" x2="{cx + 5:.1f}" y1="{y(e):.1f}" y2="{y(e):.1f}" stroke="{INK}"/>')
                label_x = x + bar_w / 2 + 8
                out.append(f'<text x="{label_x:.1f}" y="{y(v) + (12 if v < 0 else -4):.1f}" fill="{INK}">'
                           f'{value_format.format(v)}</text>')
            else:
                label_y = y(v) - 4 if v >= 0 else y(v) + 14
                out.append(f'<text x="{x + bar_w / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                           f'font-size="11">{value_format.format(v)}</text>')

    for i, label in enumerate(labels):
        x = left + slot * i + slot / 2
        for j, line in enumerate(label.split("\n")):
            out.append(f'<text x="{x:.1f}" y="{height - bottom + 18 + 14 * j}" text-anchor="middle" '
                       f'fill="{INK if j == 0 else INK_2}">{line}</text>')

    if series:
        lx = left
        for g, name in enumerate(series):
            out.append(f'<rect x="{lx}" y="34" width="10" height="10" rx="2" fill="{SERIES[g]}"/>')
            out.append(f'<text x="{lx + 16}" y="43">{name}</text>')
            lx += 16 + 7 * len(name) + 24

    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def svg_hbar_chart(path, title, labels, values, total, note=""):
    """Horizontal bar chart of counts out of total."""
    row_h = 26
    left, right, top = 260, 64, 48
    width = 720
    height = top + row_h * len(labels) + 40
    plot_w = width - left - right
    vmax = max(max(values), 1)

    out = _svg_open(width, height, title)
    for i, (label, v) in enumerate(zip(labels, values)):
        yc = top + row_h * i + row_h / 2
        w = plot_w * v / vmax
        out.append(f'<text x="{left - 8}" y="{yc + 4:.1f}" text-anchor="end">{label}</text>')
        if v:
            out.append(
                f'<path d="M{left},{yc - 8:.1f} H{left + w - 4:.1f} Q{left + w:.1f},{yc - 8:.1f} {left + w:.1f},{yc - 4:.1f} '
                f'V{yc + 4:.1f} Q{left + w:.1f},{yc + 8:.1f} {left + w - 4:.1f},{yc + 8:.1f} H{left} Z" '
                f'fill="{SERIES[0]}"><title>{label}: {v} / {total}</title></path>'
            )
        out.append(f'<text x="{left + w + 6:.1f}" y="{yc + 4:.1f}" fill="{INK_2}">{v} / {total}</text>')
    out.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + row_h * len(labels)}" stroke="{INK_2}"/>')
    if note:
        out.append(f'<text x="16" y="{height - 14}" fill="{INK_2}" font-size="11">{note}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def svg_heatmap(path, title, labels, matrix, n):
    """Square matrix of counts out of n; one-hue sequential ramp."""
    cell = 64
    left, top = 64, 48
    width = left + cell * len(labels) + 24
    height = top + cell * len(labels) + 64

    out = _svg_open(width, height, title)
    for i, a in enumerate(labels):
        out.append(f'<text x="{left - 8}" y="{top + cell * i + cell / 2 + 4:.1f}" text-anchor="end">{a}</text>')
        out.append(f'<text x="{left + cell * i + cell / 2:.1f}" y="{top + cell * len(labels) + 18}" '
                   f'text-anchor="middle">{a}</text>')
        for j, b in enumerate(labels):
            v = matrix[i][j]
            step = min(len(BLUE_RAMP) - 1, int(v / n * len(BLUE_RAMP)))
            ink = "#ffffff" if step >= 4 else INK
            x, yy = left + cell * j, top + cell * i
            out.append(f'<rect x="{x + 1}" y="{yy + 1}" width="{cell - 2}" height="{cell - 2}" rx="2" '
                       f'fill="{BLUE_RAMP[step]}"><title>{a} vs {b}: {v} / {n} same answer</title></rect>')
            out.append(f'<text x="{x + cell / 2:.1f}" y="{yy + cell / 2:.1f}" text-anchor="middle" fill="{ink}">{v}</text>')
            out.append(f'<text x="{x + cell / 2:.1f}" y="{yy + cell / 2 + 14:.1f}" text-anchor="middle" '
                       f'fill="{ink}" font-size="10">{100 * v / n:.0f}%</text>')
    out.append(f'<text x="16" y="{height - 12}" fill="{INK_2}" font-size="11">'
               f'Cell = examples (of {n}) where both conditions give the same parsed answer.</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    with open(EVAL_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)

    out_dir = args.out_dir or Path(config["output_dir"]) / "analysis" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    split = args.split

    items, api_failures, errors = load_items(config, split)
    n = len(items)
    n_success = sum(r is not None for i in items for r in i["records"].values())
    full_n_pairs = n * len(CONDITIONS)

    # -----------------------------------------------------------------
    # 1. Main accuracies (with a percentile bootstrap CI per condition).
    # -----------------------------------------------------------------
    main_rows = []
    for condition in CONDITIONS:
        correct = sum(item["correct"][condition] for item in items)
        low, high = percentile_ci(
            [float(i["correct"][condition]) for i in items],
            BOOTSTRAP_RESAMPLES, random.Random(BOOTSTRAP_SEED),
        )
        statuses = [item["parse_status"][condition] for item in items]
        main_rows.append({
            "condition": condition,
            "label": SHORT[condition],
            "correct": correct,
            "total": n,
            "accuracy": round(correct / n, 4),
            "ci95_low": round(low, 4),
            "ci95_high": round(high, 4),
            "answered": sum(s in ("exact", "normalized") for s in statuses),
            "unparsed_or_missing": sum(s not in ("exact", "normalized") for s in statuses),
        })
        # Supplementary only: accuracy among parsed answers.
        answered = main_rows[-1]["answered"]
        main_rows[-1]["answered_only_accuracy"] = round(correct / answered, 4) if answered else None
    write_csv(out_dir / "main_results.csv", main_rows)
    acc = {row["condition"]: row["accuracy"] for row in main_rows}

    # -----------------------------------------------------------------
    # 2. Gaps, paired bootstrap, McNemar.
    # -----------------------------------------------------------------
    stats = {}
    for with_c, without_c in COMPARISONS:
        name = f"{SHORT[with_c]} vs {SHORT[without_c]}"
        stats[name] = paired_bootstrap(items, with_c, without_c)
        stats[name]["with"] = with_c
        stats[name]["without"] = without_c
        stats[name]["mcnemar_exact"] = mcnemar_exact(items, with_c, without_c)

    def comparison(with_c, without_c):
        return stats[f"{SHORT[with_c]} vs {SHORT[without_c]}"]

    gaps = {}
    for name, (with_c, without_c) in GAPS.items():
        s = comparison(with_c, without_c)
        gaps[name] = {
            "definition": f"Acc({SHORT[with_c]}) - Acc({SHORT[without_c]})",
            "with": with_c,
            "without": without_c,
            "value": round(s["observed_difference"], 4),
            "ci95": [round(s["ci95_low"], 4), round(s["ci95_high"], 4)],
            "mcnemar_p": round(s["mcnemar_exact"]["p_value_two_sided"], 4),
        }

    with open(out_dir / "modality_gaps.json", "w", encoding="utf-8") as f:
        json.dump({
            "split": split,
            "model": config["model"]["model_id"],
            "gateway_model_id": config["model"]["gateway_model_id"],
            "n_examples": n,
            "accuracy": {SHORT[c]: acc[c] for c in CONDITIONS},
            "gaps": gaps,
        }, f, indent=2)
        f.write("\n")

    with open(out_dir / "bootstrap_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "split": split,
            "method": "paired bootstrap over examples (both conditions of an example resampled together), "
                      f"{BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}, percentile 95% CI; "
                      "exact two-sided McNemar (binomial) on discordant pairs",
            "comparisons": stats,
        }, f, indent=2)
        f.write("\n")

    def unparsed(item, c):
        return item["parse_status"][c] not in ("exact", "normalized")

    stat_rows = []
    for name, s in stats.items():
        m = s["mcnemar_exact"]
        # How many discordant pairs are abstentions (unparsed answers)
        # rather than wrong letters in the condition that lost.
        first_only = [i for i in items if i["correct"][s["with"]] and not i["correct"][s["without"]]]
        second_only = [i for i in items if i["correct"][s["without"]] and not i["correct"][s["with"]]]
        stat_rows.append({
            "comparison": name,
            "acc_first": acc[s["with"]],
            "acc_second": acc[s["without"]],
            "difference": round(s["observed_difference"], 4),
            "ci95_low": round(s["ci95_low"], 4),
            "ci95_high": round(s["ci95_high"], 4),
            "only_first_correct": m["only_with_correct"],
            "only_second_correct": m["only_without_correct"],
            "mcnemar_exact_p": round(m["p_value_two_sided"], 4),
            "n": n,
            "only_first_correct_second_unparsed": sum(unparsed(i, s["without"]) for i in first_only),
            "only_second_correct_first_unparsed": sum(unparsed(i, s["with"]) for i in second_only),
        })
    write_csv(out_dir / "statistical_tests.csv", stat_rows)

    # -----------------------------------------------------------------
    # 3. Per-item behavior table and conditional necessity.
    # -----------------------------------------------------------------
    item_rows = []
    for item in items:
        x = item["example"]
        row = {
            "uid": x["uid"],
            "video_id": x["video_id"],
            "domain": x["domain"],
            "sub_category": x["sub_category"],
            "task_domain": x["task_domain"],
            "task_type": x["task_type"],
            "audio_class": "|".join(x["audio_class"]),
            "speech_present": item["speech"],
            "subtitle_kind": item["subtitle_kind"],
            "gold": x["answer"],
        }
        for c in CONDITIONS:
            row[f"pred_{SHORT[c]}"] = item["prediction"][c] or f"<{item['parse_status'][c]}>"
        for c in CONDITIONS:
            row[f"correct_{SHORT[c]}"] = int(item["correct"][c])
        row["pattern"] = "".join("1" if item["correct"][c] else "0" for c in CONDITIONS)
        for name in BEHAVIORS:
            row[name] = int(name in item["behaviors"])
        item_rows.append(row)
    write_csv(out_dir / "item_behavior.csv", item_rows)

    behavior_counts = {
        name: sum(name in item["behaviors"] for item in items) for name in BEHAVIORS
    }

    full_correct = [i for i in items if i["correct"]["full"]]
    necessity_rows = []
    for name, (description, rule) in CONDITIONAL_NECESSITY.items():
        k = sum(bool(rule(i["correct"])) for i in full_correct)
        necessity_rows.append({
            "measure": name,
            "description": description,
            "count": k,
            "denominator_full_correct": len(full_correct),
            "fraction": round(k / len(full_correct), 4) if full_correct else None,
        })
    write_csv(out_dir / "conditional_necessity.csv", necessity_rows)

    # -----------------------------------------------------------------
    # 4. Subgroups (exploratory).
    # -----------------------------------------------------------------
    def members(rule):
        return [i for i in items if rule(i)]

    subgroup_rows = []
    for label, rule in [("Speech present", lambda i: i["speech"]),
                        ("Speech absent", lambda i: not i["speech"])]:
        group = members(rule)
        if group:
            subgroup_rows.append(group_row("speech", label, group, with_ci=True))
    for cls in sorted({c for i in items for c in i["example"]["audio_class"]}):
        subgroup_rows.append(group_row("audio_class_contains", cls,
                                       members(lambda i, cls=cls: cls in i["example"]["audio_class"])))
    for value in sorted({i["example"]["domain"] for i in items}):
        subgroup_rows.append(group_row("domain", value,
                                       members(lambda i, value=value: i["example"]["domain"] == value)))
    write_csv(out_dir / "subgroup_results.csv", subgroup_rows)

    subtitle_rows = [
        group_row("subtitle_kind", kind, members(lambda i, kind=kind: i["subtitle_kind"] == kind), with_ci=True)
        for kind in ("linguistic", "tags_only")
        if members(lambda i, kind=kind: i["subtitle_kind"] == kind)
    ]
    write_csv(out_dir / "subtitle_group_results.csv", subtitle_rows)

    def by_field(field):
        return [
            group_row(field, value, members(lambda i, value=value: i["example"][field] == value))
            for value in sorted({i["example"][field] for i in items})
        ]

    task_domain_rows = by_field("task_domain")
    task_type_rows = sorted(by_field("task_type"), key=lambda r: (-r["n"], r["group"]))
    write_csv(out_dir / "task_domain_results.csv", task_domain_rows)
    write_csv(out_dir / "task_type_results.csv", task_type_rows)

    # -----------------------------------------------------------------
    # 5. Prediction agreement and letter distribution. Unparsed answers
    #    compare as their parse status.
    # -----------------------------------------------------------------
    def answer(item, c):
        return item["prediction"][c] or f"<{item['parse_status'][c]}>"

    agreement = {
        a: {b: sum(answer(i, a) == answer(i, b) for i in items) for b in CONDITIONS}
        for a in CONDITIONS
    }
    agreement_rows = []
    for ia, a in enumerate(CONDITIONS):
        for b in CONDITIONS[ia + 1:]:
            both_correct = sum(i["correct"][a] and i["correct"][b] for i in items)
            agreement_rows.append({
                "condition_a": SHORT[a],
                "condition_b": SHORT[b],
                "same_answer": agreement[a][b],
                "n": n,
                "proportion": round(agreement[a][b] / n, 4),
                "same_answer_both_correct": both_correct,
                "same_answer_both_wrong": agreement[a][b] - both_correct,
            })
    write_csv(out_dir / "prediction_agreement.csv", agreement_rows)

    letters = sorted({c[0] for i in items for c in i["example"]["candidates"]})
    gold_counts = {l: sum(i["example"]["answer"] == l for i in items) for l in letters}
    letter_rows = [{"source": "gold", **{l: gold_counts[l] for l in letters}, "unparsed": 0, "n": n}]
    letter_counts = {}
    for c in CONDITIONS:
        letter_counts[c] = {l: sum(i["prediction"][c] == l for i in items) for l in letters}
        letter_rows.append({
            "source": SHORT[c],
            **letter_counts[c],
            "unparsed": n - sum(letter_counts[c].values()),
            "n": n,
        })
    for row in letter_rows:
        top = max(letters, key=lambda l: row[l])
        row["most_frequent"] = top
        row["most_frequent_share"] = round(row[top] / n, 4)
    write_csv(out_dir / "prediction_letter_distribution.csv", letter_rows)

    # -----------------------------------------------------------------
    # 6. Parse status / truncation / refusal by condition.
    # -----------------------------------------------------------------
    parse_rows = []
    refusal_examples = {}
    for c in CONDITIONS:
        statuses = [i["parse_status"][c] for i in items]
        records = [i["records"][c] for i in items]
        truncated = [r for r in records if finish_reason(r) == "length"]
        unparsed_records = [r for r in records if r and r["parse_status"] in ("ambiguous", "invalid")]
        refusal_unparsed = [r for r in unparsed_records if is_refusal_like(r)]
        refusal_parsed = [r for r in records if r and r["parse_status"] in ("exact", "normalized") and is_refusal_like(r)]
        refusal_examples[c] = refusal_unparsed
        completion_tokens = [
            ((r.get("request_metadata") or {}).get("usage") or {}).get("completion_tokens")
            for r in records if r
        ]
        completion_tokens = [t for t in completion_tokens if t is not None]
        parse_rows.append({
            "condition": c,
            "label": SHORT[c],
            "n": n,
            "exact": statuses.count("exact"),
            "normalized": statuses.count("normalized"),
            "ambiguous": statuses.count("ambiguous"),
            "invalid": statuses.count("invalid"),
            "missing_no_success": statuses.count("missing"),
            "empty_response": sum(1 for r in records if r and r["raw_response"] == ""),
            "max_tokens_truncated": len(truncated),
            "truncated_without_answer": sum(r["parse_status"] in ("ambiguous", "invalid") for r in truncated),
            "refusal_like_unparsed": len(refusal_unparsed),
            "refusal_like_but_answered": len(refusal_parsed),
            "api_failed_attempts": api_failures[c],
            "median_completion_tokens": sorted(completion_tokens)[len(completion_tokens) // 2] if completion_tokens else None,
        })
    write_csv(out_dir / "parse_status_by_condition.csv", parse_rows)

    # -----------------------------------------------------------------
    # 7. Mechanical case listing (every member of every category).
    # -----------------------------------------------------------------
    lines = [
        f"# Behavior-category cases ({split.upper()} split)",
        "",
        f"Model: `{config['model']['gateway_model_id']}`, prompt `{PROMPT_VERSION}`, temperature "
        f"{config['generation']['temperature']}. N = {n}.",
        "",
        "Generated by `python -m scripts.analyze_results` from saved outputs. Categories are defined "
        "mechanically from per-condition correctness and **overlap**: an example can appear in several "
        "sections. Descriptions state only which conditions were correct; they are not interpretations.",
        "",
        "Raw responses are shown truncated to their last 300 characters.",
        "",
    ]
    for name, (rule, _) in BEHAVIORS.items():
        group = [i for i in items if name in i["behaviors"]]
        lines += [f"## `{name}`: {len(group)} example(s)", "", f"Rule: {rule}.", ""]
        for item in group:
            x = item["example"]
            right = [SHORT[c] for c in CONDITIONS if item["correct"][c]]
            lines += [
                f"### {x['uid']}",
                "",
                f"- {x['domain']} / {x['sub_category']}; {x['task_domain']} / {x['task_type']}; "
                f"audio: {', '.join(x['audio_class'])}; subtitle: {item['subtitle_kind']}",
                f"- Correct in: {', '.join(right) or 'none'}",
                f"- Q: {x['question']}  Choices: " + " / ".join(x["candidates"]) + f"  Gold: **{x['answer']}**",
                "",
                "| condition | prediction | parse | correct | raw response (tail) |",
                "|---|---|---|---|---|",
            ]
            for c in CONDITIONS:
                record = item["records"][c]
                raw = (record["raw_response"] if record else "") or ""
                tail = raw[-300:].replace("|", "\\|").replace("\n", " ")
                if len(raw) > 300:
                    tail = "…" + tail
                lines.append(
                    f"| {SHORT[c]} | {item['prediction'][c] or '-'} | {item['parse_status'][c]} | "
                    f"{'yes' if item['correct'][c] else 'no'} | {tail} |"
                )
            lines.append("")
    (out_dir / "behavior_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -----------------------------------------------------------------
    # 8. DEV vs TEST (descriptive only; never pooled).
    # -----------------------------------------------------------------
    other = "dev" if split == "test" else "test"
    other_dir = Path(config["output_dir"]) / other
    split_rows = []
    if other_dir.exists():
        other_items, _, _ = load_items(config, other)
        by_split = {split: items, other: other_items}
        for c in CONDITIONS:
            row = {"condition": c, "label": SHORT[c]}
            for s in ("dev", "test"):
                group = by_split[s]
                k = sum(i["correct"][c] for i in group)
                row[f"{s}_correct"] = k
                row[f"{s}_n"] = len(group)
                row[f"{s}_accuracy"] = round(k / len(group), 4)
            split_rows.append(row)
        write_csv(out_dir / "dev_vs_test.csv", split_rows)

    # -----------------------------------------------------------------
    # 9. Figures.
    # -----------------------------------------------------------------
    chance = sum(1 / len(i["example"]["candidates"]) for i in items) / n
    svg_bar_chart(
        out_dir / f"{split}_accuracy_by_condition.svg",
        f"{split.upper()} accuracy by condition (N={n}, 95% bootstrap CI)",
        [f"{SHORT[c]}\n{c}" for c in CONDITIONS],
        [acc[c] for c in CONDITIONS],
        0, 1, "accuracy",
        errors=[(r["ci95_low"], r["ci95_high"]) for r in main_rows],
        reference=(chance, f"chance ≈ {chance:.2f}"),
    )
    svg_bar_chart(
        out_dir / f"{split}_modality_gaps.svg",
        f"{split.upper()} paired accuracy differences, V+A minus X (95% bootstrap CI, N={n})",
        [f"{name}\n{'Δ_audio' if s['without'] == 'video_only' else 'Δ_visual' if s['without'] == 'audio_only' else 'Δ_raw_audio' if s['without'] == 'video_subtitle' else ''}"
         for name, s in stats.items()],
        [s["observed_difference"] for s in stats.values()],
        -0.6, 0.6, "accuracy difference",
        errors=[(s["ci95_low"], s["ci95_high"]) for s in stats.values()],
        value_format="{:+.3f}",
    )
    speech_rows = {r["group"]: r for r in subgroup_rows if r["group_type"] == "speech"}
    speech_groups = [g for g in ("Speech present", "Speech absent") if g in speech_rows]
    svg_bar_chart(
        out_dir / f"{split}_accuracy_by_audio_group.svg",
        f"{split.upper()} accuracy by speech presence (exploratory)",
        [f"{SHORT[c]}\n{c}" for c in CONDITIONS],
        [[speech_rows[g][f"acc_{SHORT[c]}"] for c in CONDITIONS] for g in speech_groups],
        0, 1, "accuracy",
        series=[f"{g} (n={speech_rows[g]['n']})" for g in speech_groups],
    )
    svg_hbar_chart(
        out_dir / f"{split}_behavior_counts.svg",
        f"{split.upper()} item behavior categories (overlapping, N={n})",
        list(BEHAVIORS),
        [behavior_counts[b] for b in BEHAVIORS],
        n,
        note="Categories overlap; one example can count in several rows. See item_behavior.csv for rules.",
    )
    svg_heatmap(
        out_dir / f"{split}_prediction_agreement.svg",
        f"{split.upper()} prediction agreement (N={n})",
        [SHORT[c] for c in CONDITIONS],
        [[agreement[a][b] for b in CONDITIONS] for a in CONDITIONS],
        n,
    )

    # -----------------------------------------------------------------
    # 10. Summary.
    # -----------------------------------------------------------------
    error_kinds = {}
    for e in errors:
        kind = re.match(r"(\w+Error: HTTP \d+|\w+)", e)
        key = kind.group(1) if kind else "other"
        error_kinds[key] = error_kinds.get(key, 0) + 1

    s = [
        f"# WorldSense {split.upper()} results summary",
        "",
        ("> **FINAL TEST RESULTS** under the frozen protocol (`configs/final_test_protocol.json`)."
         if split == "test" else
         "> **DEV / PILOT RESULTS (protocol development), NOT FINAL TEST RESULTS.**"),
        f"> N = {n}. One example = {100 / n:.1f} accuracy points.",
        "",
        "Generated by `python -m scripts.analyze_results "
        f"--split {split}` from saved outputs (no API calls).",
        "",
        "## Setup",
        "",
        f"- Model: Gemini 2.5 Flash-Lite, CMU AI Gateway model ID `{config['model']['gateway_model_id']}`",
        f"- Temperature {config['generation']['temperature']}, prompt `{PROMPT_VERSION}` "
        f"(one prompt for all conditions), max_tokens "
        + ("512" if split == "test" else "not set (DEV ran before the cap was introduced)"),
        f"- Frozen {split} split: {n} examples, {len({i['example']['video_id'] for i in items})} unique videos, "
        "one QA per video, videos <= 120 s",
        f"- (example, condition) pairs with a successful response: {n_success} / {full_n_pairs}",
        f"- Failed API attempts recorded: {sum(api_failures.values())}"
        + (f" ({', '.join(f'{k}: {v}' for k, v in sorted(error_kinds.items()))})" if error_kinds else ""),
        "",
        "Conditions: V+A = `full` (original MP4), V = `video_only` (silent MP4), A = `audio_only` (AAC in M4A), "
        "V+S = `video_subtitle` (silent MP4 + normalized official subtitle text), S = `subtitle_only`, "
        "Q = `question_only` (question + choices). Ambiguous, invalid, empty or missing answers count as incorrect.",
        "",
        "## Accuracy",
        "",
        "| condition | correct / N | accuracy | 95% bootstrap CI | answered | answered-only accuracy (supplementary) |",
        "|---|---|---|---|---|---|",
    ]
    for r in main_rows:
        s.append(f"| {r['label']} | {r['correct']} / {r['total']} | {pct(r['accuracy'])} | "
                 f"[{pct(r['ci95_low'])}, {pct(r['ci95_high'])}] | {r['answered']} / {r['total']} | "
                 f"{pct(r['answered_only_accuracy'])} |")
    s += [
        "",
        f"Chance (mean 1 / #choices) = {pct(chance)}.",
        "",
        f"## Paired comparisons (bootstrap {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}; exact McNemar)",
        "",
        "| comparison | difference | 95% CI | McNemar p | discordant (only first / only second correct) "
        "| of which the losing condition abstained (unparsed) |",
        "|---|---|---|---|---|---|",
    ]
    gap_name = {v: k for k, v in GAPS.items()}
    for r in stat_rows:
        st = stats[r["comparison"]]
        tag = gap_name.get((st["with"], st["without"]))
        s.append(
            f"| {r['comparison']}{f' ({tag})' if tag else ''} | {r['difference']:+.3f} | "
            f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}] | {r['mcnemar_exact_p']:.3f} | "
            f"{r['only_first_correct']} / {r['only_second_correct']} | "
            f"{r['only_first_correct_second_unparsed']} / {r['only_second_correct_first_unparsed']} |"
        )
    s += [
        "",
        "## Conditional necessity (among examples correct with V+A)",
        "",
        "| measure | count / full-correct | fraction |",
        "|---|---|---|",
    ]
    for r in necessity_rows:
        s.append(f"| {r['description']} | {r['count']} / {r['denominator_full_correct']} | {pct(r['fraction'])} |")

    def group_table(title, rows, ci=False):
        out = [
            "",
            f"## {title}",
            "",
            "| group | n | " + " | ".join(SHORT[c] for c in CONDITIONS)
            + " | Δ_audio | Δ_visual | Δ_raw_audio |" + (" Δ_raw_audio 95% CI | McNemar p |" if ci else ""),
            "|---|---|" + "---|" * (len(CONDITIONS) + 3) + ("---|---|" if ci else ""),
        ]
        for r in rows:
            out.append(
                f"| {r['group']} | {r['n']} | "
                + " | ".join(f"{r[f'correct_{SHORT[c]}']} ({pct(r[f'acc_{SHORT[c]}'])})" for c in CONDITIONS)
                + f" | {r['delta_audio']:+.3f} | {r['delta_visual']:+.3f} | {r['delta_raw_audio']:+.3f} |"
                + (f" [{r['delta_raw_audio_ci95_low']:+.3f}, {r['delta_raw_audio_ci95_high']:+.3f}] | "
                   f"{r['delta_raw_audio_mcnemar_p']:.3f} |" if ci else "")
            )
        return out

    s += group_table("Speech presence (WorldSense audio_class)", [r for r in subgroup_rows if r["group_type"] == "speech"], ci=True)
    s += group_table("Subtitle type (linguistic vs bracketed tags only)", subtitle_rows, ci=True)
    s += group_table("Task domain (exploratory)", task_domain_rows)
    s += group_table("Audio class contains (exploratory; classes overlap)",
                     [r for r in subgroup_rows if r["group_type"] == "audio_class_contains"])
    s += group_table("Video domain (exploratory)", [r for r in subgroup_rows if r["group_type"] == "domain"])
    s += group_table("Task type (exploratory; tiny groups)", task_type_rows)
    s += [
        "",
        "Cells: correct count (accuracy).",
        "",
        "## Behavior categories (overlapping)",
        "",
        "| category | rule | count / N |",
        "|---|---|---|",
    ]
    for name, (rule, _) in BEHAVIORS.items():
        s.append(f"| `{name}` | {rule} | {behavior_counts[name]} / {n} |")
    s += [
        "",
        "## Parse status, truncation and refusal-like responses",
        "",
        "| condition | exact | normalized | ambiguous | invalid | missing | empty | hit max_tokens | "
        "truncated w/o answer | refusal-like unparsed | refusal-like answered | median completion tokens |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in parse_rows:
        s.append(
            f"| {r['label']} | {r['exact']} | {r['normalized']} | {r['ambiguous']} | {r['invalid']} | "
            f"{r['missing_no_success']} | {r['empty_response']} | {r['max_tokens_truncated']} | "
            f"{r['truncated_without_answer']} | {r['refusal_like_unparsed']} | {r['refusal_like_but_answered']} | "
            f"{r['median_completion_tokens']} |"
        )
    s += [
        "",
        "Refusal-like = raw response matches a fixed keyword pattern (e.g. \"I cannot\", \"unable to determine\", "
        "\"not enough information\"); a mechanical heuristic, see `REFUSAL_RE` in `scripts/analyze_results.py`.",
        "",
        "## Prediction agreement",
        "",
        f"Examples (of {n}) on which two conditions give the same answer (unparsed answers match only the same "
        "parse status).",
        "",
        "| | " + " | ".join(SHORT[c] for c in CONDITIONS) + " |",
        "|---|" + "---|" * len(CONDITIONS),
    ]
    for a in CONDITIONS:
        s.append(f"| {SHORT[a]} | " + " | ".join(str(agreement[a][b]) for b in CONDITIONS) + " |")
    s += [
        "",
        "## Predicted-letter distribution",
        "",
        "| | " + " | ".join(letters) + " | unparsed |",
        "|---|" + "---|" * (len(letters) + 1),
    ]
    for r in letter_rows:
        s.append(f"| {r['source']} | " + " | ".join(str(r[l]) for l in letters) + f" | {r['unparsed']} |")
    if split_rows:
        s += [
            "",
            "## DEV vs TEST accuracy (descriptive only; never pooled)",
            "",
            "| condition | DEV | TEST |",
            "|---|---|---|",
        ]
        for r in split_rows:
            s.append(f"| {r['label']} | {r['dev_correct']} / {r['dev_n']} ({pct(r['dev_accuracy'])}) | "
                     f"{r['test_correct']} / {r['test_n']} ({pct(r['test_accuracy'])}) |")
    s += [
        "",
        "## Notes",
        "",
        f"- N = {n}; subgroup results are exploratory and tiny groups should not be interpreted.",
        "- Categories are behavioral: they show which inputs sufficed for a correct answer, not which "
        "information the model internally used.",
        "",
    ]
    (out_dir / f"{split}_summary.md").write_text("\n".join(s), encoding="utf-8")

    print(f"Analyzed {n} {split} examples, {n_success}/{full_n_pairs} successful results.")
    for r in main_rows:
        print(f"  {r['label']:<4} {r['correct']:>2}/{r['total']}  {r['accuracy']:.3f}")
    for r in stat_rows:
        print(f"  {r['comparison']:<11} {r['difference']:+.3f}  CI [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"
              f"  McNemar p={r['mcnemar_exact_p']:.3f}")
    print(f"Wrote {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
