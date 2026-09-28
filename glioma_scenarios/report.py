"""Reporting: JSON-safe results and a plain-language Markdown summary.

Every report states what is patient-specific (anatomy, observations) and what
is population-informed (kinetics, drug, immune), and repeats the claims the
framework must not make.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CANNOT_CLAIM = [
    "a single MRI directly measures immune activity",
    "a single MRI directly measures drug concentration in brain tissue",
    "drug sensitivity has been individually calibrated for this patient",
    "the exact biological age of the tumour is known",
    "the best clinical drug schedule for this individual has been determined",
    "MRI segmentations are direct measurements of cycling, quiescent and necrotic cells",
]

SCOPE = ("Research tool for uncertainty-aware treatment-scenario analysis. "
         "Not a clinical decision system; outputs are not individual prescriptions.")


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if hasattr(x, "to_dict"):
        return to_jsonable(x.to_dict())
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def save_json(obj, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)


def markdown_report(title: str, provenance: Dict[str, str], identifiability: List[dict],
                    outcomes: List[dict], equivalence: dict, seeding: Optional[List[dict]] = None,
                    seeding_text: Optional[str] = None) -> str:
    L = [f"# {title}", "", f"_{SCOPE}_", "", "## What comes from where", ""]
    for k, v in provenance.items():
        L.append(f"- **{k}**: {v}")
    L += ["", "## Identifiability", "", "| parameter | verdict | posterior/prior variance reduction |",
          "|---|---|---|"]
    for r in identifiability:
        L.append(f"| {r['name']} | {r['verdict']} | {r['contraction']:.2f} |")
    combos = sorted({c for r in identifiability for c in r.get("combinations", [])})
    if combos:
        L += ["", "Identifiable combinations (log-linear): " + "; ".join(f"`{c}`" for c in combos)]
    L += ["", "## Scenario outcomes (median [5%, 95%])", "",
          "| schedule | total dose (mg) | mean viable burden | core volume at horizon (mL) | time to progression (d) |",
          "|---|---|---|---|---|"]
    fmt = lambda d: f"{d['median']:.3g} [{d['q05']:.3g}, {d['q95']:.3g}]"  # noqa: E731
    for r in outcomes:
        L.append(f"| {r['schedule']} | {r['total_dose_mg']:.0f} | {fmt(r['auc_viable'])} | "
                 f"{fmt(r['final_core_ml'])} | {fmt(r['ttp_days'])} |")
    L += ["", f"## Treatment-equivalence classes (metric: {equivalence['metric']})", ""]
    for i, c in enumerate(equivalence["classes"], 1):
        L.append(f"{i}. " + ", ".join(f"'{n}'" for n in c))
    L += ["", "### Pairwise statements", ""]
    for p in equivalence["pairs"]:
        L.append(f"- [{p['verdict']}] {p['statement']}")
    if seeding:
        L += ["", "## Effective time since seeding (model-dependent)", "",
              "| assumption | median | 90% interval |", "|---|---|---|"]
        for r in seeding:
            s = r["T_seed_days"]
            L.append(f"| {r['assumption']} | {s['median']:.0f} d | {s['q05']:.0f}-{s['q95']:.0f} d |")
        if seeding_text:
            L += ["", seeding_text]
    L += ["", "## Claims this analysis does NOT make", ""] + [f"- It does not claim that {c}." for c in CANNOT_CLAIM]
    return "\n".join(L) + "\n"
