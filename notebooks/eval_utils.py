"""
eval_utils.py — shared helpers for all three eval notebooks.
Place this file alongside the notebooks (in the notebooks/ directory).
"""

import json
import re
from collections import Counter


# ── Text helpers ──────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:64]


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 \-]", "", text)
    return text


# ── Output parsing ────────────────────────────────────────────────────────────

_HRS_PATTERN = re.compile(r"-?\b[0-3]\b")

def extract_hrs_prediction(output_obj) -> str:
    """
    Pull the raw generated string out of whatever the pipeline returns,
    then find the first integer in {0, -1, -2, -3}.
    Returns the matched string (e.g. '-2') or '' if nothing matched.
    """
    raw = _raw_text(output_obj)
    # Look for the last assistant turn if the pipeline returns the full conversation
    if isinstance(output_obj, list):
        for item in reversed(output_obj):
            if isinstance(item, dict):
                content = item.get("generated_text", "")
                if isinstance(content, list):
                    # chat-style: list of role/content dicts
                    for turn in reversed(content):
                        if isinstance(turn, dict) and turn.get("role") == "assistant":
                            raw = _stringify_content(turn.get("content", ""))
                            break
                    break

    m = _HRS_PATTERN.search(raw)
    return m.group(0) if m else ""


def _raw_text(output_obj) -> str:
    if isinstance(output_obj, str):
        return output_obj.strip()
    if isinstance(output_obj, list) and output_obj:
        first = output_obj[0]
        if isinstance(first, dict):
            for k in ["generated_text", "text", "answer", "content"]:
                if k in first:
                    val = first[k]
                    if isinstance(val, str):
                        return val.strip()
                    if isinstance(val, list):
                        return _stringify_content(val)
    if isinstance(output_obj, dict):
        for k in ["generated_text", "text", "answer", "content"]:
            if k in output_obj and isinstance(output_obj[k], str):
                return output_obj[k].strip()
    return str(output_obj).strip()


def _stringify_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text", ""))
            else:
                parts.append(str(c))
        return " ".join(parts)
    return str(content)


# ── Metrics ───────────────────────────────────────────────────────────────────

VALID_HRS = {"0", "-1", "-2", "-3"}

def compute_metrics(results: list[dict]) -> dict:
    """
    Given a list of result dicts (each with 'gold_answer' and 'prediction_text'),
    compute accuracy, per-class recall, and parse-failure rate.
    """
    total = len(results)
    if total == 0:
        return {}

    exact, normalized, parse_fail = 0, 0, 0
    per_class = {k: {"tp": 0, "support": 0} for k in VALID_HRS}

    for r in results:
        gold = str(r["gold_answer"]).strip()
        pred = str(r["prediction_text"]).strip()

        if pred not in VALID_HRS:
            parse_fail += 1

        if pred == gold:
            exact += 1
        if normalize_text(pred) == normalize_text(gold):
            normalized += 1

        if gold in per_class:
            per_class[gold]["support"] += 1
            if pred == gold:
                per_class[gold]["tp"] += 1

    per_class_recall = {}
    for cls, counts in per_class.items():
        sup = counts["support"]
        per_class_recall[f"recall_hrs_{cls}"] = (
            round(counts["tp"] / sup, 4) if sup > 0 else None
        )

    return {
        "num_samples":      total,
        "exact_match":      round(exact / total, 4),
        "normalized_match": round(normalized / total, 4),
        "parse_failure_rate": round(parse_fail / total, 4),
        **per_class_recall,
    }
