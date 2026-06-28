# Frontmatter export - per-file context cards from the knowledge graph.
#
# For each source file that has AST-derived architectural design (nodes/edges
# extracted by tree-sitter + clustering), emit a markdown doc:
#   YAML frontmatter (file metadata) + Few-shot references + Example uses + Guardrails.
#
# These are non-destructive sidecars (written to graphify-out/frontmatter/, never
# into the source files themselves) meant for an AI assistant to read right before
# it edits a given file.
from __future__ import annotations
from collections import Counter
from pathlib import Path
import networkx as nx

from graphify.analyze import _is_concept_node, _is_file_node, god_nodes, surprising_connections
from graphify.ingest import _yaml_str
from graphify.wiki import _safe_filename

# Structural relations are mechanical (import/contains) and uninteresting as
# few-shot examples or usage hints - they mirror file layout, not design intent.
_STRUCTURAL_RELATIONS = frozenset({"imports", "imports_from", "contains", "method"})


def _is_real_source_file(source_file: str) -> bool:
    """A real source path: non-empty and the basename has an extension."""
    return bool(source_file) and "." in source_file.split("/")[-1]


def source_files(G: nx.Graph) -> list[str]:
    """Every real source file that contributed nodes to the graph, sorted."""
    files = {
        d.get("source_file", "")
        for _, d in G.nodes(data=True)
        if _is_real_source_file(d.get("source_file", ""))
    }
    return sorted(files)


def _file_nodes(G: nx.Graph, source_file: str) -> list[str]:
    """Node ids whose source_file is this file."""
    return [n for n, d in G.nodes(data=True) if d.get("source_file", "") == source_file]


def _real_nodes(G: nx.Graph, node_ids: list[str]) -> list[str]:
    """Drop synthetic file-hub and concept nodes, keep real entities."""
    return [n for n in node_ids if not _is_file_node(G, n) and not _is_concept_node(G, n)]


def _label(G: nx.Graph, nid: str) -> str:
    return G.nodes[nid].get("label", nid)


def _anchor(G: nx.Graph, nid: str) -> str:
    """`source_file:Lxx` anchor an agent can open, falling back gracefully."""
    d = G.nodes[nid]
    src = d.get("source_file", "")
    loc = d.get("source_location", "")
    if src and loc:
        return f"{src}:{loc}"
    return src or "unknown"


def build_file_frontmatter(
    G: nx.Graph,
    source_file: str,
    communities: dict[int, list[str]] | None = None,
    community_labels: dict[int, str] | None = None,
    gods: list[dict] | None = None,
    surprises: list[dict] | None = None,
) -> str:
    """Return a markdown doc (YAML frontmatter + three sections) for one source file.

    Sections, all derived from the AST/graph:
      - Few-shot references: related cross-file entities to study as exemplars.
      - Example uses: who calls/uses this file's entities (incoming edges).
      - Guardrails: blast-radius god nodes, AMBIGUOUS/INFERRED links to verify.

    `gods` / `surprises` may be precomputed once by the caller and reused across
    files; if omitted they are computed from G.
    """
    communities = communities or {}
    community_labels = community_labels or {}
    gods = gods if gods is not None else god_nodes(G, top_n=10)
    surprises = surprises if surprises is not None else surprising_connections(G, communities, top_n=50)

    nodes_here = _file_nodes(G, source_file)
    real_here = _real_nodes(G, nodes_here)
    here_set = set(nodes_here)
    god_ids = {g["id"] for g in gods}

    # ---- Frontmatter metadata -------------------------------------------------
    file_type = ""
    cid_counter: Counter = Counter()
    for nid in nodes_here:
        d = G.nodes[nid]
        file_type = file_type or d.get("file_type", "")
        c = d.get("community")
        if c is not None:
            cid_counter[int(c)] += 1
    community_id = cid_counter.most_common(1)[0][0] if cid_counter else None
    community_name = (
        community_labels.get(community_id, f"Community {community_id}")
        if community_id is not None
        else ""
    )

    # Top entities in the file by degree.
    top_entities = sorted(real_here, key=lambda n: G.degree(n), reverse=True)[:10]

    # God nodes defined in this file (blast-radius warnings).
    gods_here = [g for g in gods if g["id"] in here_set]

    # Confidence breakdown across edges incident to this file's entities.
    conf_counts: Counter = Counter()
    for nid in real_here:
        for nbr in G.neighbors(nid):
            conf_counts[G.edges[nid, nbr].get("confidence", "EXTRACTED")] += 1

    fm: list[str] = ["---"]
    fm.append(f'source_file: "{_yaml_str(source_file)}"')
    if file_type:
        fm.append(f'file_type: "{_yaml_str(file_type)}"')
    if community_name:
        fm.append(f'community: "{_yaml_str(community_name)}"')
    fm.append(f"entity_count: {len(real_here)}")
    if top_entities:
        fm.append("entities:")
        for nid in top_entities:
            loc = G.nodes[nid].get("source_location", "")
            suffix = f" ({loc})" if loc else ""
            fm.append(f'  - "{_yaml_str(_label(G, nid) + suffix)}"')
    if gods_here:
        fm.append("god_nodes:")
        for g in gods_here:
            fm.append(f'  - "{_yaml_str(g["label"])} ({g["degree"]} connections)"')
    fm.append("confidence:")
    for conf in ("EXTRACTED", "INFERRED", "AMBIGUOUS"):
        fm.append(f"  {conf.lower()}: {conf_counts.get(conf, 0)}")
    fm.append("generated_by: graphify")
    fm.append("---")

    lines: list[str] = fm + ["", f"# {Path(source_file).name}", ""]
    if community_name:
        lines += [f"> Part of **{community_name}** · {len(real_here)} entities · see [[index]]", ""]

    # ---- Few-shot references --------------------------------------------------
    # Cross-file entities most related to this file: concrete exemplars to open
    # and pattern-match against. Sourced from surprising_connections (ranked) and
    # backfilled with the strongest cross-file neighbours of this file's gods.
    lines += [
        "## Few-shot references",
        "",
        "_Related entities elsewhere in the corpus - open these as worked examples._",
        "",
    ]
    fewshot: list[str] = []
    seen_refs: set[str] = set()

    here_files = {source_file}
    for s in surprises:
        files = s.get("source_files", []) or []
        if source_file not in files:
            continue
        # The "other" side of the connection is the exemplar.
        other = s["target"] if (files and files[0] == source_file) else s["source"]
        other_file = next((f for f in files if f != source_file), files[0] if files else "")
        key = f"{other}|{other_file}"
        if key in seen_refs:
            continue
        seen_refs.add(key)
        why = s.get("why") or s.get("note") or "related across files"
        anchor = f" in `{other_file}`" if other_file else ""
        fewshot.append(f"- [[{other}]]{anchor} — {why}")

    # Backfill from cross-file neighbours of this file's god nodes.
    if len(fewshot) < 8:
        for nid in sorted(real_here, key=lambda n: G.degree(n), reverse=True):
            for nbr in sorted(G.neighbors(nid), key=lambda n: G.degree(n), reverse=True):
                if nbr in here_set or _is_file_node(G, nbr) or _is_concept_node(G, nbr):
                    continue
                rel = G.edges[nid, nbr].get("relation", "related")
                if rel in _STRUCTURAL_RELATIONS:
                    continue
                key = f"{_label(G, nbr)}|{_anchor(G, nbr)}"
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                fewshot.append(f"- [[{_label(G, nbr)}]] in `{_anchor(G, nbr)}` — {rel} `{_label(G, nid)}`")
                if len(fewshot) >= 8:
                    break
            if len(fewshot) >= 8:
                break

    lines += (fewshot[:8] or ["- _No cross-file references found._"]) + [""]

    # ---- Example uses --------------------------------------------------------
    # Incoming edges: external entities that call/use this file's entities.
    # Direction is recovered from the _src/_tgt fields the builder preserves on
    # otherwise-undirected edges.
    lines += [
        "## Example uses",
        "",
        "_How this file's entities are used elsewhere (callers / dependents)._",
        "",
    ]
    uses: list[str] = []
    seen_uses: set[str] = set()
    for nid in real_here:
        for nbr in G.neighbors(nid):
            if nbr in here_set or _is_file_node(G, nbr):
                continue
            ed = G.edges[nid, nbr]
            rel = ed.get("relation", "related")
            if rel in _STRUCTURAL_RELATIONS:
                continue
            # Keep edges that point INTO this file's entity (external uses it).
            tgt = ed.get("_tgt")
            if tgt is not None and tgt != nid:
                continue
            key = f"{nbr}|{nid}|{rel}"
            if key in seen_uses:
                continue
            seen_uses.add(key)
            uses.append(f"- `{_label(G, nbr)}` ({_anchor(G, nbr)}) --{rel}--> `{_label(G, nid)}`")
            if len(uses) >= 12:
                break
        if len(uses) >= 12:
            break
    lines += (uses or ["- _No external callers detected in the graph._"]) + [""]

    # ---- Guardrails ----------------------------------------------------------
    lines += [
        "## Guardrails",
        "",
        "_Edit with care - what breaks if you change this file._",
        "",
    ]
    guards: list[str] = []
    for g in gods_here:
        guards.append(
            f"- ⚠️ **{g['label']}** is a god node ({g['degree']} connections) — "
            f"changes here ripple widely; check dependents before refactoring."
        )
    # AMBIGUOUS / INFERRED edges touching this file - verify before relying on them.
    seen_warn: set[str] = set()
    for nid in real_here:
        for nbr in G.neighbors(nid):
            ed = G.edges[nid, nbr]
            conf = ed.get("confidence", "EXTRACTED")
            if conf not in ("AMBIGUOUS", "INFERRED"):
                continue
            rel = ed.get("relation", "related")
            key = f"{nid}|{nbr}|{rel}"
            if key in seen_warn:
                continue
            seen_warn.add(key)
            verb = "is unverified" if conf == "AMBIGUOUS" else "was inferred, not stated"
            guards.append(
                f"- `{_label(G, nid)}` --{rel}--> `{_label(G, nbr)}` {verb} "
                f"({conf}) — confirm before depending on it."
            )
            if len(guards) >= 12:
                break
        if len(guards) >= 12:
            break
    lines += (guards or ["- _No high-risk connections flagged for this file._"]) + [""]

    lines += ["---", "", "*Generated by graphify. Non-authoritative: verify against source.*"]
    return "\n".join(lines)


def to_frontmatter(
    G: nx.Graph,
    communities: dict[int, list[str]] | None = None,
    output_dir: str | Path = "graphify-out/frontmatter",
    community_labels: dict[int, str] | None = None,
) -> int:
    """Write one frontmatter card per source file. Returns the number written."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    communities = communities or {}
    # Compute the expensive graph-wide analyses once, reuse per file.
    gods = god_nodes(G, top_n=10)
    surprises = surprising_connections(G, communities, top_n=50)

    count = 0
    for sf in source_files(G):
        doc = build_file_frontmatter(
            G, sf, communities, community_labels, gods=gods, surprises=surprises
        )
        (out / f"{_safe_filename(sf)}.md").write_text(doc, encoding="utf-8")
        count += 1
    return count
