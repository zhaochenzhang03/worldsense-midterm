"""
Offline analysis of saved evaluation outputs. Makes no API calls.

Reads outputs/<split>/<condition>.jsonl (one successful record per
(uid, condition) is expected; see scripts/validate_evaluation.py) and
writes report tables, figures and summaries to outputs/analysis/.

A missing result or an unparsed answer (ambiguous / invalid) counts as
incorrect, so every accuracy uses all frozen examples of the split as
its denominator.
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


BOOTSTRAP_RESAMPLES = 2000
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
    "negative_audio_interference": (
        "video_only correct AND full incorrect",
        lambda c: c["video_only"] and not c["full"],
    ),
    "negative_visual_interference": (
        "audio_only correct AND full incorrect",
        lambda c: c["audio_only"] and not c["full"],
    ),
    "language_or_option_shortcut": (
        "question_only correct",
        lambda c: c["question_only"],
    ),
}

# Sections of failure_cases.md: (title, behavior labels listed).
FAILURE_SECTIONS = [
    ("Vision sufficient (possible vision shortcut)", ["vision_sufficient"]),
    ("Subtitle substitutable", ["subtitle_substitutable"]),
    ("Raw-audio dependent candidates", ["raw_audio_dependent_candidate"]),
    ("Genuine multimodal candidates", ["genuine_multimodal_candidate"]),
    ("Negative multimodal interference", ["negative_audio_interference", "negative_visual_interference"]),
    ("Question-only (language / option) shortcut", ["language_or_option_shortcut"]),
]

_TAG_RE = re.compile(r"\[[^\]]*\]")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline analysis of saved results.")
    parser.add_argument("--split", choices=list(SPLIT_FILES), default="dev")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/analysis"))
    return parser.parse_args()


def subtitle_kind(video_id: str) -> str:
    """
    "linguistic" if the normalized official subtitle has any text left
    after removing bracketed tags such as [Music] or [ __ ], else
    "tags_only".
    """
    remainder = _TAG_RE.sub("", load_subtitle(video_id))
    return "linguistic" if re.search(r"\w", remainder) else "tags_only"


def accuracy(items: list[dict], condition: str) -> float | None:
    if not items:
        return None
    return sum(item["correct"][condition] for item in items) / len(items)


def mean_gap(items: list[dict], with_condition: str, without_condition: str) -> float:
    return sum(
        item["correct"][with_condition] - item["correct"][without_condition]
        for item in items
    ) / len(items)


def paired_bootstrap(items, with_condition, without_condition):
    """
    Resample examples with replacement (paired: both conditions of an
    example move together). Percentile 95% CI.
    """
    rng = random.Random(BOOTSTRAP_SEED)
    diffs = [
        item["correct"][with_condition] - item["correct"][without_condition]
        for item in items
    ]
    n = len(diffs)
    stats = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "observed_difference": sum(diffs) / n,
        "ci95_low": stats[int(0.025 * BOOTSTRAP_RESAMPLES)],
        "ci95_high": stats[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
        "n": n,
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


# ---------------------------------------------------------------------
# Figures: minimal SVG bar charts (standard library only).
# ---------------------------------------------------------------------

def svg_bar_chart(path, title, labels, values, y_min, y_max, y_label, errors=None, series=None):
    """
    Vertical bar chart. values is a list of numbers, or, with series,
    a list of per-series lists (grouped bars).
    """
    width, height = 640, 360
    left, right, top, bottom = 64, 16, 40, 56
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    groups = values if series else [values]

    def y(v):
        return top + plot_h * (y_max - v) / (y_max - y_min)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>',
    ]

    ticks = 5
    for i in range(ticks + 1):
        v = y_min + (y_max - y_min) * i / ticks
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    out.append(f'<line x1="{left}" x2="{width - right}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#333"/>')
    out.append(
        f'<text transform="translate(16,{top + plot_h / 2}) rotate(-90)" text-anchor="middle">{y_label}</text>'
    )

    slot = plot_w / len(labels)
    bar_w = slot * 0.7 / len(groups)

    for g, group in enumerate(groups):
        for i, v in enumerate(group):
            if v is None:
                continue
            x = left + slot * i + slot * 0.15 + bar_w * g
            y0, y1 = sorted((y(0), y(v)))
            out.append(
                f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" height="{y1 - y0:.1f}" fill="{colors[g % len(colors)]}"/>'
            )
            label_y = y(v) - 4 if v >= 0 else y(v) + 14
            out.append(f'<text x="{x + bar_w / 2:.1f}" y="{label_y:.1f}" text-anchor="middle">{v:+.2f}</text>'
                       if y_min < 0 else
                       f'<text x="{x + bar_w / 2:.1f}" y="{label_y:.1f}" text-anchor="middle">{v:.2f}</text>')
            if errors and not series:
                lo, hi = errors[i]
                cx = x + bar_w / 2
                out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}" stroke="#222"/>')
                for e in (lo, hi):
                    out.append(f'<line x1="{cx - 6:.1f}" x2="{cx + 6:.1f}" y1="{y(e):.1f}" y2="{y(e):.1f}" stroke="#222"/>')

    for i, label in enumerate(labels):
        x = left + slot * i + slot / 2
        for j, line in enumerate(label.split("\n")):
            out.append(f'<text x="{x:.1f}" y="{height - bottom + 18 + 14 * j}" text-anchor="middle">{line}</text>')

    if series:
        for g, name in enumerate(series):
            lx = width - right - 150
            ly = top + 4 + 16 * g
            out.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{colors[g % len(colors)]}"/>')
            out.append(f'<text x="{lx + 16}" y="{ly + 9}">{name}</text>')

    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    with open(EVAL_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(SPLIT_FILES[args.split], "r", encoding="utf-8") as f:
        examples = json.load(f)

    model = config["model"]["model_id"]
    split_dir = Path(config["output_dir"]) / args.split
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Load: the successful record per (uid, condition); count failed
    #    API attempts separately.
    # -----------------------------------------------------------------
    results = {}
    api_failures = {c: 0 for c in CONDITIONS}

    for condition in CONDITIONS:
        for record in load_jsonl(split_dir / f"{condition}.jsonl"):
            if record["model"] != model or record["prompt_version"] != PROMPT_VERSION:
                continue
            if not is_success(record):
                api_failures[condition] += 1
                continue
            key = (record["uid"], condition)
            if key in results:
                print(f"ERROR: duplicate successful result {key}", file=sys.stderr)
                return 1
            results[key] = record

    items = []
    missing = []
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
            if record is None:
                missing.append((example["uid"], condition))
            item["records"][condition] = record
            item["prediction"][condition] = record["parsed_prediction"] if record else None
            item["parse_status"][condition] = record["parse_status"] if record else "missing"
            item["correct"][condition] = bool(record and record["correct"])
        item["behaviors"] = [
            name for name, (_, rule) in BEHAVIORS.items() if rule(item["correct"])
        ]
        items.append(item)

    n = len(items)
    n_success = n * len(CONDITIONS) - len(missing)

    # -----------------------------------------------------------------
    # 2. Main accuracies.
    # -----------------------------------------------------------------
    main_rows = []
    for condition in CONDITIONS:
        correct = sum(item["correct"][condition] for item in items)
        statuses = [item["parse_status"][condition] for item in items]
        main_rows.append({
            "condition": condition,
            "label": f"Acc({SHORT[condition]})",
            "correct": correct,
            "total": n,
            "accuracy": round(correct / n, 4),
            "answered": sum(s in ("exact", "normalized") for s in statuses),
            "parse_exact": statuses.count("exact"),
            "parse_normalized": statuses.count("normalized"),
            "parse_ambiguous": statuses.count("ambiguous"),
            "parse_invalid": statuses.count("invalid"),
            "missing": statuses.count("missing"),
            "api_failed_attempts": api_failures[condition],
        })

    with open(out_dir / "main_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(main_rows[0]))
        writer.writeheader()
        writer.writerows(main_rows)

    acc = {row["condition"]: row["accuracy"] for row in main_rows}

    # -----------------------------------------------------------------
    # 3. Gaps, bootstrap, McNemar.
    # -----------------------------------------------------------------
    gaps = {
        name: {
            "definition": f"Acc({with_c}) - Acc({without_c})",
            "with": with_c,
            "without": without_c,
            "value": round(mean_gap(items, with_c, without_c), 4),
        }
        for name, (with_c, without_c) in GAPS.items()
    }
    subtitle_groups = {}
    for kind in ("linguistic", "tags_only"):
        group = [item for item in items if item["subtitle_kind"] == kind]
        subtitle_groups[kind] = {
            "n": len(group),
            "uids": [item["example"]["uid"] for item in group],
            "acc_full": accuracy(group, "full"),
            "acc_video_subtitle": accuracy(group, "video_subtitle"),
            "delta_raw_audio": mean_gap(group, "full", "video_subtitle") if group else None,
        }

    with open(out_dir / "modality_gaps.json", "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "model": model,
            "gateway_model_id": config["model"]["gateway_model_id"],
            "n_examples": n,
            "accuracy": acc,
            "gaps": gaps,
            "raw_audio_substitution_by_subtitle_kind": subtitle_groups,
            "note": "Exploratory; small sample.",
        }, f, indent=2)
        f.write("\n")

    bootstrap = {}
    for name, (with_c, without_c) in [
        ("full_vs_video_only", ("full", "video_only")),
        ("full_vs_video_subtitle", ("full", "video_subtitle")),
        ("full_vs_audio_only", ("full", "audio_only")),
    ]:
        bootstrap[name] = paired_bootstrap(items, with_c, without_c)
        bootstrap[name]["mcnemar_exact"] = mcnemar_exact(items, with_c, without_c)

    with open(out_dir / "bootstrap_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "method": "paired bootstrap over examples, percentile 95% CI; exact two-sided McNemar on discordant pairs",
            "comparisons": bootstrap,
        }, f, indent=2)
        f.write("\n")

    # -----------------------------------------------------------------
    # 4. Per-item behavior table.
    # -----------------------------------------------------------------
    item_fields = (
        ["uid", "video_id", "domain", "task_domain", "task_type", "audio_class",
         "speech_present", "subtitle_kind", "gold"]
        + [f"pred_{c}" for c in CONDITIONS]
        + [f"correct_{c}" for c in CONDITIONS]
        + ["behaviors"]
    )
    with open(out_dir / "item_behavior.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=item_fields)
        writer.writeheader()
        for item in items:
            x = item["example"]
            row = {
                "uid": x["uid"],
                "video_id": x["video_id"],
                "domain": x["domain"],
                "task_domain": x["task_domain"],
                "task_type": x["task_type"],
                "audio_class": "|".join(x["audio_class"]),
                "speech_present": item["speech"],
                "subtitle_kind": item["subtitle_kind"],
                "gold": x["answer"],
                "behaviors": "|".join(item["behaviors"]),
            }
            for c in CONDITIONS:
                row[f"pred_{c}"] = item["prediction"][c] or f"<{item['parse_status'][c]}>"
                row[f"correct_{c}"] = int(item["correct"][c])
            writer.writerow(row)

    behavior_counts = {
        name: sum(name in item["behaviors"] for item in items) for name in BEHAVIORS
    }

    # Pairwise prediction agreement (unparsed answers compare as their
    # parse status) and predicted-letter distribution: mechanical
    # checks that each condition's inputs change the answers.
    def answer(item, c):
        return item["prediction"][c] or f"<{item['parse_status'][c]}>"

    agreement = {
        a: {b: sum(answer(i, a) == answer(i, b) for i in items) for b in CONDITIONS}
        for a in CONDITIONS
    }
    with open(out_dir / "prediction_agreement.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition"] + CONDITIONS)
        for a in CONDITIONS:
            writer.writerow([a] + [agreement[a][b] for b in CONDITIONS])

    letters = sorted({c[0] for i in items for c in i["example"]["candidates"]})
    letter_counts = {
        c: {l: sum(i["prediction"][c] == l for i in items) for l in letters}
        for c in CONDITIONS
    }
    gold_counts = {l: sum(i["example"]["answer"] == l for i in items) for l in letters}

    # -----------------------------------------------------------------
    # 5. Subgroups (exploratory).
    # -----------------------------------------------------------------
    subgroup_defs = []
    subgroup_defs.append(("speech", "Speech present", lambda i: i["speech"]))
    subgroup_defs.append(("speech", "Speech absent", lambda i: not i["speech"]))
    for cls in sorted({c for i in items for c in i["example"]["audio_class"]}):
        subgroup_defs.append(("audio_class_contains", cls, lambda i, cls=cls: cls in i["example"]["audio_class"]))
    for key, group_type in [("task_domain", "task_domain"), ("domain", "domain")]:
        for value in sorted({i["example"][key] for i in items}):
            subgroup_defs.append((group_type, value, lambda i, key=key, value=value: i["example"][key] == value))
    for kind in ("linguistic", "tags_only"):
        subgroup_defs.append(("subtitle_kind", kind, lambda i, kind=kind: i["subtitle_kind"] == kind))

    subgroup_rows = []
    for group_type, group, rule in subgroup_defs:
        members = [i for i in items if rule(i)]
        if not members:
            continue
        row = {"group_type": group_type, "group": group, "n": len(members)}
        for c in CONDITIONS:
            row[f"acc_{c}"] = round(accuracy(members, c), 4)
        for name, (with_c, without_c) in GAPS.items():
            row[name] = round(mean_gap(members, with_c, without_c), 4)
        subgroup_rows.append(row)

    with open(out_dir / "subgroup_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(subgroup_rows[0]))
        writer.writeheader()
        writer.writerows(subgroup_rows)

    # -----------------------------------------------------------------
    # 6. failure_cases.md (mechanical; no interpretation).
    # -----------------------------------------------------------------
    lines = [
        f"# Behavior-category cases ({args.split.upper()} split)",
        "",
        f"Model: `{config['model']['gateway_model_id']}`, prompt `{PROMPT_VERSION}`, temperature "
        f"{config['generation']['temperature']}. N = {n}.",
        "",
        "Generated by `python -m scripts.analyze_results` from saved outputs. "
        "Categories are defined mechanically from per-condition correctness and "
        "**overlap**: an example can appear in several sections. Every member of "
        "each category is listed (the split is small). Descriptions state only "
        "which conditions were correct; they are not interpretations.",
        "",
        "Raw responses are shown truncated to their last 300 characters.",
        "",
    ]
    for title, names in FAILURE_SECTIONS:
        lines.append(f"## {title}")
        lines.append("")
        for name in names:
            members = [i for i in items if name in i["behaviors"]]
            lines.append(f"### `{name}`: {len(members)} example(s)")
            lines.append("")
            lines.append(f"Rule: {BEHAVIORS[name][0]}.")
            lines.append("")
            for item in members:
                x = item["example"]
                right = [SHORT[c] for c in CONDITIONS if item["correct"][c]]
                wrong = [SHORT[c] for c in CONDITIONS if not item["correct"][c]]
                lines += [
                    f"#### {x['uid']}",
                    "",
                    f"- Domain: {x['domain']} / {x['sub_category']}; task: {x['task_domain']} / {x['task_type']}",
                    f"- Audio class: {', '.join(x['audio_class'])}; subtitle: {item['subtitle_kind']}",
                    f"- Why listed: {BEHAVIORS[name][0]}. Correct in: {', '.join(right) or 'none'}; "
                    f"incorrect in: {', '.join(wrong) or 'none'}.",
                    f"- Question: {x['question']}",
                    "- Choices: " + " / ".join(x["candidates"]),
                    f"- Gold: **{x['answer']}**",
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
                        f"| {c} | {item['prediction'][c] or '-'} | {item['parse_status'][c]} | "
                        f"{'yes' if item['correct'][c] else 'no'} | {tail} |"
                    )
                lines.append("")
    (out_dir / "failure_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -----------------------------------------------------------------
    # 7. Figures.
    # -----------------------------------------------------------------
    svg_bar_chart(
        out_dir / f"{args.split}_accuracy_by_condition.svg",
        f"{args.split.upper()} accuracy by condition (N={n})",
        [f"{c}\n{SHORT[c]}" for c in CONDITIONS],
        [acc[c] for c in CONDITIONS],
        0, 1, "accuracy",
    )
    gap_names = list(GAPS)
    gap_ci = {
        "delta_audio": bootstrap["full_vs_video_only"],
        "delta_visual": bootstrap["full_vs_audio_only"],
        "delta_raw_audio": bootstrap["full_vs_video_subtitle"],
    }
    svg_bar_chart(
        out_dir / f"{args.split}_modality_gaps.svg",
        f"{args.split.upper()} modality gaps with 95% bootstrap CI (N={n})",
        [f"{g}\n{gaps[g]['definition']}" for g in gap_names],
        [gaps[g]["value"] for g in gap_names],
        -1, 1, "accuracy difference",
        errors=[(gap_ci[g]["ci95_low"], gap_ci[g]["ci95_high"]) for g in gap_names],
    )
    speech_rows = {r["group"]: r for r in subgroup_rows if r["group_type"] == "speech"}
    speech_groups = [g for g in ("Speech present", "Speech absent") if g in speech_rows]
    svg_bar_chart(
        out_dir / f"{args.split}_accuracy_by_audio_group.svg",
        f"{args.split.upper()} accuracy by speech presence (exploratory)",
        [f"{c}\n{SHORT[c]}" for c in CONDITIONS],
        [[speech_rows[g][f"acc_{c}"] for c in CONDITIONS] for g in speech_groups],
        0, 1, "accuracy",
        series=[f"{g} (n={speech_rows[g]['n']})" for g in speech_groups],
    )

    # -----------------------------------------------------------------
    # 8. Summary.
    # -----------------------------------------------------------------
    usage_audio_video_only = [
        ((i["records"]["video_only"] or {}).get("request_metadata", {}).get("usage") or {})
        .get("prompt_tokens_details", {}).get("audio_tokens")
        for i in items
    ]
    reported_audio = sum(1 for t in usage_audio_video_only if t)

    s = [
        f"# WorldSense {args.split.upper()} experiment summary",
        "",
        f"> **THESE ARE {args.split.upper()} RESULTS, NOT FINAL TEST RESULTS.**"
        if args.split == "dev" else "> Formal test results.",
        f"> N = {n}. Every number below is exploratory and has wide uncertainty.",
        "",
        "Generated by `python -m scripts.analyze_results` from saved outputs (no API calls).",
        "",
        "## Setup",
        "",
        f"- Model: Gemini 2.5 Flash-Lite, CMU AI Gateway model ID `{config['model']['gateway_model_id']}`",
        f"- Temperature {config['generation']['temperature']}, prompt `{PROMPT_VERSION}` (one prompt for all conditions)",
        f"- Frozen {args.split} split: {n} examples, one QA per video, videos <= 120 s",
        f"- Successful (example, condition) results: {n_success} / {n * len(CONDITIONS)}",
        f"- Failed API attempts recorded: {sum(api_failures.values())}",
        f"- Unparsed answers (ambiguous + invalid): "
        f"{sum(r['parse_ambiguous'] + r['parse_invalid'] for r in main_rows)}",
        "",
        "Conditions: `full` (original MP4, video + audio), `video_only` (silent MP4, video stream copy), "
        "`audio_only` (AAC stream in M4A), `video_subtitle` (silent MP4 + normalized official subtitle text), "
        "`subtitle_only` (normalized official subtitle text), `question_only` (question + choices).",
        "",
        "## Accuracy",
        "",
        "| condition | | correct / N | accuracy | parsed exact / normalized / ambiguous / invalid |",
        "|---|---|---|---|---|",
    ]
    for r in main_rows:
        s.append(
            f"| `{r['condition']}` | {r['label']} | {r['correct']} / {r['total']} | {pct(r['accuracy'])} | "
            f"{r['parse_exact']} / {r['parse_normalized']} / {r['parse_ambiguous']} / {r['parse_invalid']} |"
        )
    s += [
        "",
        "## Modality gaps (paired bootstrap, 2000 resamples, seed 42)",
        "",
        "| gap | definition | observed | 95% CI | McNemar exact p (discordant b / c) |",
        "|---|---|---|---|---|",
    ]
    for g in gap_names:
        b = gap_ci[g]
        m = b["mcnemar_exact"]
        s.append(
            f"| {g} | {gaps[g]['definition']} | {gaps[g]['value']:+.3f} | "
            f"[{b['ci95_low']:+.3f}, {b['ci95_high']:+.3f}] | "
            f"{m['p_value_two_sided']:.3f} ({m['only_with_correct']} / {m['only_without_correct']}) |"
        )
    s += [
        "",
        "b = examples correct only with the first condition; c = correct only with the second.",
        "",
        "## Raw-audio substitution by subtitle kind",
        "",
        "| subtitle kind | n | Acc(full) | Acc(video_subtitle) | Delta_raw_audio |",
        "|---|---|---|---|---|",
    ]
    for kind, g in subtitle_groups.items():
        delta = g["delta_raw_audio"]
        s.append(
            f"| {kind} | {g['n']} | {pct(g['acc_full'])} | {pct(g['acc_video_subtitle'])} | "
            f"{'n/a' if delta is None else f'{delta:+.3f}'} |"
        )
    s += [
        "",
        "`tags_only` = the normalized official subtitle contains only bracketed tags such as `[Music]`.",
        "",
        "## Behavior categories (overlapping)",
        "",
        "| category | rule | count |",
        "|---|---|---|",
    ]
    for name, (rule, _) in BEHAVIORS.items():
        s.append(f"| `{name}` | {rule} | {behavior_counts[name]} |")
    s += [
        "",
        "Per-example labels: `item_behavior.csv`. Listed cases: `failure_cases.md`.",
        "",
        "## Subgroups (exploratory, very small n)",
        "",
        "| group | n | " + " | ".join(SHORT[c] for c in CONDITIONS) + " | Delta_audio | Delta_visual | Delta_raw_audio |",
        "|---|---|" + "---|" * (len(CONDITIONS) + 3),
    ]
    for r in subgroup_rows:
        s.append(
            f"| {r['group_type']}: {r['group']} | {r['n']} | "
            + " | ".join(pct(r[f"acc_{c}"]) for c in CONDITIONS)
            + f" | {r['delta_audio']:+.2f} | {r['delta_visual']:+.2f} | {r['delta_raw_audio']:+.2f} |"
        )
    s += [
        "",
        "## Prediction agreement between conditions",
        "",
        f"Number of the {n} examples on which two conditions give the same answer "
        "(an unparsed answer matches only the same parse status). Full table: `prediction_agreement.csv`.",
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
        "| gold | " + " | ".join(str(gold_counts[l]) for l in letters) + " | |",
    ]
    for c in CONDITIONS:
        s.append(
            f"| {SHORT[c]} | " + " | ".join(str(letter_counts[c][l]) for l in letters)
            + f" | {n - sum(letter_counts[c].values())} |"
        )
    s += [
        "",
        "## Limitations",
        "",
        f"- N = {n}: one example changes an accuracy by {100 / n:.1f} points. Confidence intervals are wide "
        "and no difference here should be called significant or used as a final claim.",
        "- Subgroups have even fewer examples; they only indicate what to look at on test.",
        "- The model often ignores \"Return only the letter\" and writes a long explanation. Such answers are "
        "parsed mechanically (e.g. `\\boxed{X}`, \"the answer is X\"); conflicting or missing letters count as incorrect.",
        f"- Gemini reports nonzero `audio_tokens` for {reported_audio} / {n} `video_only` requests although "
        "those MP4s contain no audio stream (checked with ffprobe). This looks like per-second token accounting "
        "for video input; the Gateway does not expose more detail.",
        "- The Gateway applied implicit prompt caching to repeated video inputs (`cached_tokens` in usage). "
        "This is a provider-side optimization; requests and prompts were not changed.",
        "- Single model, single prompt, temperature 0, one run: no estimate of run-to-run variance.",
        "",
    ]
    (out_dir / f"{args.split}_summary.md").write_text("\n".join(s), encoding="utf-8")

    print(f"Analyzed {n} {args.split} examples, {n_success}/{n * len(CONDITIONS)} successful results.")
    for r in main_rows:
        print(f"  {r['label']:<10} {r['correct']:>2}/{r['total']}  {r['accuracy']:.3f}")
    for g in gap_names:
        b = gap_ci[g]
        print(f"  {g:<16} {gaps[g]['value']:+.3f}  CI [{b['ci95_low']:+.3f}, {b['ci95_high']:+.3f}]")
    print(f"Wrote {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
