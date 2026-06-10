#!/usr/bin/env python3
"""Generate an offline HTML visualization of CyberAdoAgg ROLE -> TARGET flows."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "data/raw/xlsx/CyberAdoAgg_gold_global_total_latest.xlsx"
DEFAULT_OUTPUT = BASE_DIR / "results/cyberado_role_target_flows.html"

ROLE_ORDER = ["victim", "victim_support", "conciliator", "bully_support", "bully"]
ROLE_META = {
    "victim": {
        "label": "Victime",
        "short": "Victime",
        "color": "#1f78a8",
    },
    "victim_support": {
        "label": "Soutien victime",
        "short": "Soutien victime",
        "color": "#3b9f72",
    },
    "conciliator": {
        "label": "Conciliateur",
        "short": "Conciliateur",
        "color": "#c69422",
    },
    "bully_support": {
        "label": "Soutien harceleur",
        "short": "Soutien harc.",
        "color": "#d86c32",
    },
    "bully": {
        "label": "Harceleur",
        "short": "Harceleur",
        "color": "#b64f66",
    },
}

ROLE_ALIASES = {
    "victim": [
        "victim",
        "victime",
        "target",
        "cible",
        "harcele",
        "harcelee",
        "harcele_e",
        "personne_harcelee",
    ],
    "victim_support": [
        "victim_support",
        "victim support",
        "support_victim",
        "victimsupport",
        "victime_support",
        "soutien_victim",
        "soutien_victime",
        "supporter_victime",
        "defenseur_victime",
        "defense_victime",
    ],
    "conciliator": [
        "conciliator",
        "conciliateur",
        "conciliatrice",
        "conciliation",
        "mediateur",
        "mediatrice",
        "moderateur",
        "moderatrice",
    ],
    "bully_support": [
        "bully_support",
        "bully support",
        "support_bully",
        "bullysupport",
        "harceleur_support",
        "soutien_bully",
        "soutien_harceleur",
        "supporter_harceleur",
        "pro_harceleur",
    ],
    "bully": [
        "bully",
        "harceleur",
        "harceleuse",
        "agresseur",
        "agresseuse",
        "attaquant",
        "attaquante",
    ],
}


def normalize_key(value: Any) -> str:
    """Return a lowercase ASCII-ish key robust to spacing, case, accents and dashes."""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("'", "_")
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_/]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


ALIAS_TO_ROLE: dict[str, str] = {}
COMPACT_ALIAS_TO_ROLE: dict[str, str] = {}
for canonical_role, aliases in ROLE_ALIASES.items():
    for alias in aliases + [canonical_role]:
        key = normalize_key(alias)
        ALIAS_TO_ROLE[key] = canonical_role
        COMPACT_ALIAS_TO_ROLE[key.replace("_", "")] = canonical_role


def cell_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_missing_cell(value: Any) -> bool:
    text = cell_text(value)
    return not text or text.lower() in {"nan", "none", "null", "<na>"}


def is_noise_target(value: Any) -> bool:
    text = cell_text(value).lower()
    return text.startswith("file:") or "candidates:" in text or "majority:" in text


def canonicalize_role(value: Any) -> tuple[str | None, str]:
    """Map a role-ish cell to the five canonical roles, with conservative fuzzy repair."""
    if is_missing_cell(value):
        return None, "missing"

    key = normalize_key(value)
    compact = key.replace("_", "")
    if key in ALIAS_TO_ROLE:
        return ALIAS_TO_ROLE[key], "exact"
    if compact in COMPACT_ALIAS_TO_ROLE:
        return COMPACT_ALIAS_TO_ROLE[compact], "compact"

    candidates = list(ALIAS_TO_ROLE.keys())
    best_key = ""
    best_score = 0.0
    for candidate in candidates:
        score = SequenceMatcher(None, key, candidate).ratio()
        if score > best_score:
            best_key = candidate
            best_score = score

    if len(key) >= 4 and best_score >= 0.88:
        return ALIAS_TO_ROLE[best_key], f"fuzzy:{best_score:.2f}"
    return None, "unmatched"


def canonicalize_targets(value: Any) -> tuple[list[str], str]:
    """Split and normalize TARGET cells. Annotation diagnostics are treated as noise."""
    if is_missing_cell(value):
        return [], "missing"
    if is_noise_target(value):
        return [], "noise"

    text = cell_text(value)
    pieces = [
        piece
        for piece in re.split(r"\s*(?:/|;|,|\||\+|&|\bet\b|\band\b)\s*", text, flags=re.I)
        if piece.strip()
    ]
    roles: list[str] = []
    methods: list[str] = []
    for piece in pieces:
        role, method = canonicalize_role(piece)
        if role is not None:
            roles.append(role)
            methods.append(method)

    unique_roles = []
    seen_roles = set()
    for role in roles:
        if role not in seen_roles:
            unique_roles.append(role)
            seen_roles.add(role)
    if unique_roles:
        status = "exact" if all(method == "exact" for method in methods) else "repaired"
        return unique_roles, status
    return [], "unmatched"


def raw_value(value: Any) -> str:
    return "<NA>" if is_missing_cell(value) else cell_text(value)


def clean_participant_name(value: Any) -> str:
    text = cell_text(value).strip("<> ")
    text = re.sub(r"\d+$", "", text)
    text = re.split(r"[-_]", text)[0]
    return text.strip() or "sans nom"


def detect_threads(df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[int, str]]:
    ids = pd.to_numeric(df["ID"], errors="coerce")
    starts = [0]
    for row_index in range(1, len(df)):
        current = ids.iloc[row_index]
        previous = ids.iloc[row_index - 1]
        if pd.notna(current) and pd.notna(previous) and current <= previous:
            starts.append(row_index)
    starts.append(len(df))

    threads: list[dict[str, Any]] = []
    row_to_thread: dict[int, str] = {}
    for number, (start, end) in enumerate(zip(starts, starts[1:]), start=1):
        sub = df.iloc[start:end]
        victim_names = Counter()
        for _, row in sub.iterrows():
            role, _ = canonicalize_role(row.get("ROLE"))
            if role == "victim":
                victim_names[clean_participant_name(row.get("NAME"))] += 1
        victim_name = victim_names.most_common(1)[0][0] if victim_names else ""
        label = f"Fil {number}" + (f" - {victim_name}" if victim_name else "")
        thread_id = f"thread_{number}"
        threads.append(
            {
                "id": thread_id,
                "label": label,
                "start_row": int(start),
                "end_row": int(end - 1),
                "row_count": int(end - start),
            }
        )
        for row_index in range(start, end):
            row_to_thread[row_index] = thread_id
    return threads, row_to_thread


def blank_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "valid_source_rows": 0,
        "missing_source_rows": 0,
        "unmatched_source_rows": 0,
        "valid_target_rows": 0,
        "missing_target_rows": 0,
        "noise_target_rows": 0,
        "unmatched_target_rows": 0,
        "visualized_rows": 0,
        "relations": 0,
    }


def add_stats(total: dict[str, Any], part: dict[str, Any]) -> None:
    for key, value in part.items():
        if isinstance(value, int):
            total[key] += value


def build_payload(input_path: Path) -> dict[str, Any]:
    df = pd.read_excel(input_path)
    missing_columns = {"ID", "NAME", "TIME", "TEXT", "ROLE", "TARGET"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {', '.join(sorted(missing_columns))}")

    threads, row_to_thread = detect_threads(df)
    thread_stats = {thread["id"]: blank_stats() for thread in threads}
    relations: list[dict[str, Any]] = []

    role_raw_counter: Counter[str] = Counter()
    target_raw_counter: Counter[str] = Counter()
    source_mapping_counter: Counter[tuple[str, str, str]] = Counter()
    target_mapping_counter: Counter[tuple[str, str, str]] = Counter()

    for row_index, row in df.iterrows():
        thread_id = row_to_thread[int(row_index)]
        stats = thread_stats[thread_id]
        stats["rows"] += 1

        raw_role = raw_value(row.get("ROLE"))
        raw_target = raw_value(row.get("TARGET"))
        role_raw_counter[raw_role] += 1
        target_raw_counter[raw_target] += 1

        source_role, source_status = canonicalize_role(row.get("ROLE"))
        source_mapping_counter[(raw_role, source_role or "", source_status)] += 1
        if source_role:
            stats["valid_source_rows"] += 1
        elif source_status == "missing":
            stats["missing_source_rows"] += 1
        else:
            stats["unmatched_source_rows"] += 1

        target_roles, target_status = canonicalize_targets(row.get("TARGET"))
        target_mapping_counter[(raw_target, "/".join(target_roles), target_status)] += 1
        if target_roles:
            stats["valid_target_rows"] += 1
        elif target_status == "missing":
            stats["missing_target_rows"] += 1
        elif target_status == "noise":
            stats["noise_target_rows"] += 1
        else:
            stats["unmatched_target_rows"] += 1

        if not source_role or not target_roles:
            continue

        stats["visualized_rows"] += 1
        for target_role in target_roles:
            stats["relations"] += 1
            relations.append(
                {
                    "row_index": int(row_index),
                    "message_id": cell_text(row.get("ID")),
                    "thread_id": thread_id,
                    "name": cell_text(row.get("NAME")),
                    "time": cell_text(row.get("TIME")),
                    "text": cell_text(row.get("TEXT")),
                    "source": source_role,
                    "target": target_role,
                    "raw_role": raw_role,
                    "raw_target": raw_target,
                    "target_count": len(target_roles),
                }
            )

    all_stats = blank_stats()
    for stats in thread_stats.values():
        add_stats(all_stats, stats)
    thread_stats["all"] = all_stats

    all_thread = {
        "id": "all",
        "label": "Tous les fils",
        "start_row": 0,
        "end_row": int(len(df) - 1),
        "row_count": int(len(df)),
    }

    target_noise_samples = [
        {"raw": raw, "count": count}
        for raw, count in target_raw_counter.most_common()
        if raw.startswith("File:")
    ][:10]

    source_audit = [
        {"raw": raw, "mapped": mapped or "-", "status": status, "count": count}
        for (raw, mapped, status), count in sorted(
            source_mapping_counter.items(), key=lambda item: (-item[1], item[0][0])
        )
    ]
    target_audit = [
        {"raw": raw, "mapped": mapped or "-", "status": status, "count": count}
        for (raw, mapped, status), count in sorted(
            target_mapping_counter.items(), key=lambda item: (-item[1], item[0][0])
        )
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(input_path),
        "roles": [
            {
                "id": role_id,
                "label": ROLE_META[role_id]["label"],
                "short": ROLE_META[role_id]["short"],
                "color": ROLE_META[role_id]["color"],
            }
            for role_id in ROLE_ORDER
        ],
        "threads": [all_thread] + threads,
        "thread_stats": thread_stats,
        "relations": relations,
        "audit": {
            "source": source_audit,
            "target": target_audit,
            "noise_target_samples": target_noise_samples,
            "notes": [
                "Les roles sont normalises sans tenir compte de la casse, des accents, des espaces et des tirets.",
                "Les cibles multiples separees par /, ;, virgule, +, &, et/and sont decomposees en plusieurs relations.",
                "Les cellules TARGET de diagnostic commencant par File: ou contenant Candidates/Majority sont ignorees.",
                "Les corrections floues ne sont acceptees qu'a partir d'un score de similarite >= 0.88.",
            ],
        },
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CyberAdoAgg - flux ROLE vers TARGET</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f8;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d9dee7;
      --panel: #ffffff;
      --panel-soft: #fafbfc;
      --focus: #275d8c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.45;
      letter-spacing: 0;
    }
    main {
      width: min(1500px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 36px;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: end;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 26px;
      line-height: 1.15;
      font-weight: 760;
    }
    .subtitle {
      margin: 0;
      color: var(--muted);
      max-width: 820px;
    }
    .source {
      color: var(--muted);
      font-size: 12px;
      text-align: right;
      max-width: 420px;
      overflow-wrap: anywhere;
    }
    .thread-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 16px;
    }
    button.thread-tab {
      min-height: 36px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 8px;
      padding: 7px 12px;
      font: inherit;
      cursor: pointer;
    }
    button.thread-tab[aria-pressed="true"] {
      background: #244761;
      border-color: #244761;
      color: white;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 8px;
      margin-bottom: 16px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      min-height: 68px;
    }
    .metric-value {
      display: block;
      font-size: 21px;
      line-height: 1.15;
      font-weight: 760;
    }
    .metric-label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }
    .workspace {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .svg-wrap {
      width: 100%;
      overflow-x: auto;
      background: #fcfcfd;
    }
    svg {
      display: block;
      width: 100%;
      min-width: 920px;
      height: auto;
      min-height: 520px;
    }
    .column-title {
      fill: #4b5563;
      font-size: 16px;
      font-weight: 760;
    }
    .node-rect {
      fill: white;
      stroke-width: 2.5;
      rx: 8;
      ry: 8;
    }
    .node-label {
      fill: #111827;
      font-size: 16px;
      font-weight: 760;
    }
    .node-count {
      fill: #667085;
      font-size: 12px;
    }
    .edge-path {
      fill: none;
      opacity: 0.48;
      cursor: pointer;
      transition: opacity 140ms ease, stroke-width 140ms ease;
    }
    .edge-hit {
      fill: none;
      stroke: transparent;
      stroke-width: 20;
      cursor: pointer;
    }
    .edge-group:hover .edge-path,
    .edge-group.is-selected .edge-path {
      opacity: 0.92;
    }
    .edge-group.is-muted .edge-path {
      opacity: 0.16;
    }
    .legend-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-top: 1px solid var(--line);
      background: var(--panel-soft);
      color: var(--muted);
      font-size: 13px;
    }
    .legend-item {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      white-space: nowrap;
    }
    .swatch {
      width: 11px;
      height: 11px;
      border-radius: 3px;
      display: inline-block;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: minmax(340px, 0.95fr) minmax(420px, 1.35fr);
      gap: 16px;
      margin-top: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-width: 0;
    }
    .panel h2 {
      margin: 0;
      padding: 14px 16px 10px;
      font-size: 17px;
      line-height: 1.2;
    }
    .panel-note {
      color: var(--muted);
      font-size: 12px;
      padding: 0 16px 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 9px 10px;
      border-top: 1px solid #edf0f4;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: #4b5563;
      background: #fafbfc;
      font-weight: 760;
    }
    tr.edge-row {
      cursor: pointer;
    }
    tr.edge-row:hover,
    tr.edge-row.is-selected {
      background: #f0f6fb;
    }
    .role-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
      font-weight: 680;
    }
    .role-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      flex: 0 0 auto;
    }
    .examples {
      max-height: 460px;
      overflow: auto;
    }
    .message {
      border-top: 1px solid #edf0f4;
      padding: 12px 16px;
    }
    .message:first-child {
      border-top: 0;
    }
    .message-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    .message-text {
      margin: 0;
      overflow-wrap: anywhere;
    }
    .raw-target {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
      overflow-wrap: anywhere;
    }
    details.audit {
      margin-top: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0;
    }
    details.audit summary {
      cursor: pointer;
      padding: 14px 16px;
      font-weight: 760;
    }
    .audit-body {
      border-top: 1px solid var(--line);
      padding: 0 0 10px;
    }
    .audit-notes {
      margin: 12px 16px;
      color: var(--muted);
      font-size: 13px;
    }
    .audit-tables {
      display: grid;
      grid-template-columns: 1fr 1.3fr;
      gap: 16px;
      padding: 0 16px 14px;
    }
    .table-scroll {
      max-height: 360px;
      overflow: auto;
      border: 1px solid #edf0f4;
      border-radius: 8px;
    }
    .tooltip {
      position: fixed;
      pointer-events: none;
      z-index: 20;
      max-width: 340px;
      padding: 9px 10px;
      border-radius: 8px;
      background: #1f2933;
      color: white;
      box-shadow: 0 12px 28px rgba(16, 24, 40, 0.22);
      font-size: 13px;
      opacity: 0;
      transform: translate(-9999px, -9999px);
      transition: opacity 80ms ease;
    }
    .tooltip.is-visible {
      opacity: 1;
    }
    @media (max-width: 1050px) {
      main {
        width: min(100vw - 20px, 1500px);
        padding-top: 16px;
      }
      .topbar {
        grid-template-columns: 1fr;
      }
      .source {
        text-align: left;
      }
      .metrics {
        grid-template-columns: repeat(3, minmax(120px, 1fr));
      }
      .detail-grid,
      .audit-tables {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 620px) {
      .metrics {
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }
      h1 {
        font-size: 22px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header class="topbar">
      <div>
        <h1>Flux des messages par role</h1>
        <p class="subtitle">Chaque fleche relie le role de l'auteur du message a un role cible issu de <code>TARGET</code>. Les cibles multiples sont separees en plusieurs relations.</p>
      </div>
      <div class="source" id="sourceInfo"></div>
    </header>

    <nav class="thread-tabs" id="threadTabs" aria-label="Filtrer par fil WhatsApp"></nav>
    <section class="metrics" id="metrics" aria-label="Statistiques du filtre courant"></section>

    <section class="workspace" aria-label="Graphe oriente des roles">
      <div class="svg-wrap">
        <svg id="flowSvg" viewBox="0 0 1200 650" role="img" aria-labelledby="svgTitle svgDesc">
          <title id="svgTitle">Flux ROLE vers TARGET</title>
          <desc id="svgDesc">Graphe dirige entre roles auteurs a gauche et roles cibles a droite.</desc>
        </svg>
      </div>
      <div class="legend-row" id="legend"></div>
    </section>

    <section class="detail-grid">
      <article class="panel">
        <h2>Fleches affichees</h2>
        <div class="panel-note">Selectionner une ligne ou une fleche pour afficher des messages exemples.</div>
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Source</th>
                <th>Cible</th>
                <th>Relations</th>
                <th>%</th>
              </tr>
            </thead>
            <tbody id="edgeTable"></tbody>
          </table>
        </div>
      </article>

      <article class="panel">
        <h2 id="examplesTitle">Messages exemples</h2>
        <div class="panel-note" id="examplesNote"></div>
        <div class="examples" id="examples"></div>
      </article>
    </section>

    <details class="audit">
      <summary>Audit de normalisation</summary>
      <div class="audit-body">
        <div class="audit-notes" id="auditNotes"></div>
        <div class="audit-tables">
          <section>
            <h2>ROLE brut -> role canonique</h2>
            <div class="table-scroll">
              <table>
                <thead><tr><th>Valeur brute</th><th>Canonique</th><th>Statut</th><th>n</th></tr></thead>
                <tbody id="sourceAudit"></tbody>
              </table>
            </div>
          </section>
          <section>
            <h2>TARGET brut -> cible canonique</h2>
            <div class="table-scroll">
              <table>
                <thead><tr><th>Valeur brute</th><th>Canonique</th><th>Statut</th><th>n</th></tr></thead>
                <tbody id="targetAudit"></tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    </details>
  </main>

  <div class="tooltip" id="tooltip"></div>

  <script>
    const DATA = __DATA_JSON__;
    const roleOrder = DATA.roles.map(role => role.id);
    const roleById = new Map(DATA.roles.map(role => [role.id, role]));
    let selectedThread = "all";
    let selectedEdge = null;

    const svg = document.getElementById("flowSvg");
    const tooltip = document.getElementById("tooltip");

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[char]));
    }

    function fmt(value) {
      return new Intl.NumberFormat("fr-FR").format(value);
    }

    function roleLabel(roleId) {
      return roleById.get(roleId)?.label ?? roleId;
    }

    function roleChip(roleId) {
      const role = roleById.get(roleId);
      return `<span class="role-chip"><span class="role-dot" style="background:${role.color}"></span>${esc(role.label)}</span>`;
    }

    function relationsForCurrentThread() {
      if (selectedThread === "all") return DATA.relations;
      return DATA.relations.filter(relation => relation.thread_id === selectedThread);
    }

    function aggregateEdges(relations) {
      const edgeMap = new Map();
      for (const relation of relations) {
        const key = `${relation.source}->${relation.target}`;
        if (!edgeMap.has(key)) {
          edgeMap.set(key, {
            key,
            source: relation.source,
            target: relation.target,
            count: 0,
          });
        }
        edgeMap.get(key).count += 1;
      }
      return Array.from(edgeMap.values()).sort((a, b) => b.count - a.count || a.key.localeCompare(b.key));
    }

    function isSelected(edge) {
      return selectedEdge && selectedEdge.source === edge.source && selectedEdge.target === edge.target;
    }

    function ensureSelectedEdge(edges) {
      if (!edges.length) {
        selectedEdge = null;
        return;
      }
      const exists = selectedEdge && edges.some(edge => isSelected(edge));
      if (!exists) {
        selectedEdge = { source: edges[0].source, target: edges[0].target };
      }
    }

    function setSelectedEdge(source, target) {
      selectedEdge = { source, target };
      render();
    }

    function drawTabs() {
      const tabs = document.getElementById("threadTabs");
      tabs.innerHTML = DATA.threads.map(thread => `
        <button class="thread-tab" type="button" data-thread="${esc(thread.id)}" aria-pressed="${thread.id === selectedThread}">
          ${esc(thread.label)}
        </button>
      `).join("");
      tabs.querySelectorAll("button").forEach(button => {
        button.addEventListener("click", () => {
          selectedThread = button.dataset.thread;
          selectedEdge = null;
          render();
        });
      });
    }

    function drawMetrics(edges) {
      const stats = DATA.thread_stats[selectedThread];
      const edgeCount = edges.length;
      const visualizedRatio = stats.rows ? Math.round((stats.visualized_rows / stats.rows) * 100) : 0;
      const ignoredTargets = stats.missing_target_rows + stats.noise_target_rows + stats.unmatched_target_rows;
      const metrics = [
        [stats.rows, "messages du filtre"],
        [stats.visualized_rows, "messages avec cible exploitable"],
        [stats.relations, "relations role-cible"],
        [edgeCount, "fleches non nulles"],
        [ignoredTargets, "cibles ignorees"],
        [`${visualizedRatio} %`, "messages visualises"],
      ];
      document.getElementById("metrics").innerHTML = metrics.map(([value, label]) => `
        <div class="metric">
          <span class="metric-value">${typeof value === "number" ? fmt(value) : esc(value)}</span>
          <span class="metric-label">${esc(label)}</span>
        </div>
      `).join("");
    }

    function drawLegend() {
      document.getElementById("legend").innerHTML = DATA.roles.map(role => `
        <span class="legend-item"><span class="swatch" style="background:${role.color}"></span>${esc(role.label)}</span>
      `).join("") + `<span class="legend-item">Largeur = nombre de relations</span>`;
    }

    function drawSvg(edges) {
      svg.innerHTML = "";
      const width = 1200;
      const left = { x: 54, w: 270, anchor: 324 };
      const right = { x: 876, w: 270, anchor: 876 };
      const nodeH = 68;
      const yByRole = new Map(roleOrder.map((roleId, index) => [roleId, 114 + index * 108]));
      const maxCount = Math.max(1, ...edges.map(edge => edge.count));

      const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      for (const role of DATA.roles) {
        const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
        marker.setAttribute("id", `arrow-${role.id}`);
        marker.setAttribute("viewBox", "0 0 10 10");
        marker.setAttribute("refX", "9");
        marker.setAttribute("refY", "5");
        marker.setAttribute("markerWidth", "8");
        marker.setAttribute("markerHeight", "8");
        marker.setAttribute("orient", "auto-start-reverse");
        const markerPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
        markerPath.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
        markerPath.setAttribute("fill", role.color);
        marker.appendChild(markerPath);
        defs.appendChild(marker);
      }
      svg.appendChild(defs);

      const leftTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
      leftTitle.setAttribute("x", left.x);
      leftTitle.setAttribute("y", "48");
      leftTitle.setAttribute("class", "column-title");
      leftTitle.textContent = "ROLE auteur";
      svg.appendChild(leftTitle);

      const rightTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
      rightTitle.setAttribute("x", right.x);
      rightTitle.setAttribute("y", "48");
      rightTitle.setAttribute("class", "column-title");
      rightTitle.textContent = "TARGET cible";
      svg.appendChild(rightTitle);

      const sourceTotals = Object.fromEntries(roleOrder.map(roleId => [roleId, 0]));
      const targetTotals = Object.fromEntries(roleOrder.map(roleId => [roleId, 0]));
      for (const edge of edges) {
        sourceTotals[edge.source] += edge.count;
        targetTotals[edge.target] += edge.count;
      }

      const edgesLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      edgesLayer.setAttribute("aria-label", "fleches");
      svg.appendChild(edgesLayer);

      const sortedForDraw = [...edges].sort((a, b) => a.count - b.count);
      for (const edge of sortedForDraw) {
        const sourceRole = roleById.get(edge.source);
        const sy = yByRole.get(edge.source);
        const ty = yByRole.get(edge.target);
        const strokeWidth = 1.6 + Math.sqrt(edge.count / maxCount) * 15;
        const pathD = `M ${left.anchor} ${sy} C 500 ${sy}, 700 ${ty}, ${right.anchor} ${ty}`;

        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("class", [
          "edge-group",
          selectedEdge && !isSelected(edge) ? "is-muted" : "",
          isSelected(edge) ? "is-selected" : "",
        ].join(" ").trim());

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", pathD);
        path.setAttribute("class", "edge-path");
        path.setAttribute("stroke", sourceRole.color);
        path.setAttribute("stroke-width", String(strokeWidth));
        path.setAttribute("marker-end", `url(#arrow-${edge.source})`);
        group.appendChild(path);

        const hit = document.createElementNS("http://www.w3.org/2000/svg", "path");
        hit.setAttribute("d", pathD);
        hit.setAttribute("class", "edge-hit");
        group.appendChild(hit);

        const show = event => {
          tooltip.innerHTML = `<strong>${esc(roleLabel(edge.source))} -> ${esc(roleLabel(edge.target))}</strong><br>${fmt(edge.count)} relation${edge.count > 1 ? "s" : ""}`;
          tooltip.classList.add("is-visible");
          moveTooltip(event);
        };
        const hide = () => tooltip.classList.remove("is-visible");
        const move = event => moveTooltip(event);
        group.addEventListener("mouseenter", show);
        group.addEventListener("mousemove", move);
        group.addEventListener("mouseleave", hide);
        group.addEventListener("click", () => setSelectedEdge(edge.source, edge.target));
        edgesLayer.appendChild(group);
      }

      const nodesLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      nodesLayer.setAttribute("aria-label", "roles");
      svg.appendChild(nodesLayer);
      for (const role of DATA.roles) {
        drawNode(nodesLayer, left.x, yByRole.get(role.id), left.w, nodeH, role, `${fmt(sourceTotals[role.id])} relations sortantes`);
        drawNode(nodesLayer, right.x, yByRole.get(role.id), right.w, nodeH, role, `${fmt(targetTotals[role.id])} relations entrantes`);
      }
    }

    function drawNode(layer, x, centerY, width, height, role, countLabel) {
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", String(x));
      rect.setAttribute("y", String(centerY - height / 2));
      rect.setAttribute("width", String(width));
      rect.setAttribute("height", String(height));
      rect.setAttribute("class", "node-rect");
      rect.setAttribute("stroke", role.color);
      group.appendChild(rect);

      const accent = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      accent.setAttribute("x", String(x));
      accent.setAttribute("y", String(centerY - height / 2));
      accent.setAttribute("width", "8");
      accent.setAttribute("height", String(height));
      accent.setAttribute("fill", role.color);
      accent.setAttribute("rx", "6");
      accent.setAttribute("ry", "6");
      group.appendChild(accent);

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", String(x + 22));
      label.setAttribute("y", String(centerY - 6));
      label.setAttribute("class", "node-label");
      label.textContent = role.label;
      group.appendChild(label);

      const count = document.createElementNS("http://www.w3.org/2000/svg", "text");
      count.setAttribute("x", String(x + 22));
      count.setAttribute("y", String(centerY + 17));
      count.setAttribute("class", "node-count");
      count.textContent = countLabel;
      group.appendChild(count);

      layer.appendChild(group);
    }

    function moveTooltip(event) {
      const offset = 14;
      tooltip.style.transform = `translate(${event.clientX + offset}px, ${event.clientY + offset}px)`;
    }

    function drawEdgeTable(edges) {
      const total = edges.reduce((sum, edge) => sum + edge.count, 0);
      const tbody = document.getElementById("edgeTable");
      tbody.innerHTML = edges.map(edge => {
        const pct = total ? `${Math.round((edge.count / total) * 1000) / 10} %` : "0 %";
        return `
          <tr class="edge-row ${isSelected(edge) ? "is-selected" : ""}" data-source="${esc(edge.source)}" data-target="${esc(edge.target)}">
            <td>${roleChip(edge.source)}</td>
            <td>${roleChip(edge.target)}</td>
            <td>${fmt(edge.count)}</td>
            <td>${pct}</td>
          </tr>
        `;
      }).join("");
      tbody.querySelectorAll("tr").forEach(row => {
        row.addEventListener("click", () => setSelectedEdge(row.dataset.source, row.dataset.target));
      });
    }

    function drawExamples(relations) {
      const container = document.getElementById("examples");
      const title = document.getElementById("examplesTitle");
      const note = document.getElementById("examplesNote");
      if (!selectedEdge) {
        title.textContent = "Messages exemples";
        note.textContent = "Aucune relation disponible pour ce filtre.";
        container.innerHTML = "";
        return;
      }
      const matching = relations.filter(relation => relation.source === selectedEdge.source && relation.target === selectedEdge.target);
      title.textContent = `${roleLabel(selectedEdge.source)} -> ${roleLabel(selectedEdge.target)}`;
      note.textContent = `${fmt(matching.length)} relation(s), affichage des 20 premiers messages dans l'ordre du fichier.`;
      container.innerHTML = matching.slice(0, 20).map(relation => `
        <div class="message">
          <div class="message-meta">
            <span>${esc(DATA.threads.find(thread => thread.id === relation.thread_id)?.label ?? relation.thread_id)}</span>
            <span>ligne ${fmt(relation.row_index + 2)}</span>
            <span>ID ${esc(relation.message_id)}</span>
            <span>${esc(relation.time)}</span>
            <span>${esc(relation.name)}</span>
          </div>
          <p class="message-text">${esc(relation.text)}</p>
          <div class="raw-target">TARGET brut: ${esc(relation.raw_target)}</div>
        </div>
      `).join("");
    }

    function drawAudit() {
      document.getElementById("auditNotes").innerHTML = DATA.audit.notes.map(note => `<p>${esc(note)}</p>`).join("");
      document.getElementById("sourceAudit").innerHTML = DATA.audit.source.map(item => `
        <tr><td>${esc(item.raw)}</td><td>${esc(item.mapped)}</td><td>${esc(item.status)}</td><td>${fmt(item.count)}</td></tr>
      `).join("");
      document.getElementById("targetAudit").innerHTML = DATA.audit.target.map(item => `
        <tr><td>${esc(item.raw)}</td><td>${esc(item.mapped)}</td><td>${esc(item.status)}</td><td>${fmt(item.count)}</td></tr>
      `).join("");
    }

    function render() {
      drawTabs();
      const relations = relationsForCurrentThread();
      const edges = aggregateEdges(relations);
      ensureSelectedEdge(edges);
      drawMetrics(edges);
      drawSvg(edges);
      drawLegend();
      drawEdgeTable(edges);
      drawExamples(relations);
    }

    document.getElementById("sourceInfo").innerHTML = `
      Source: <code>${esc(DATA.source_file)}</code><br>
      Generation: <code>${esc(DATA.generated_at)}</code>
    `;
    drawAudit();
    render();
  </script>
</body>
</html>
"""


def generate_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML visualization of CyberAdoAgg ROLE -> TARGET flows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input XLSX file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output HTML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate_html(payload), encoding="utf-8")
    stats = payload["thread_stats"]["all"]
    print(f"Wrote {args.output}")
    print(
        "Rows: {rows} | visualized messages: {visualized_rows} | relations: {relations} | "
        "ignored targets: {ignored}".format(
            rows=stats["rows"],
            visualized_rows=stats["visualized_rows"],
            relations=stats["relations"],
            ignored=stats["missing_target_rows"]
            + stats["noise_target_rows"]
            + stats["unmatched_target_rows"],
        )
    )


if __name__ == "__main__":
    main()
