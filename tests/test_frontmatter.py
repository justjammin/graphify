"""Tests for graphify.frontmatter — per-file context cards from the graph."""
import networkx as nx
from graphify.frontmatter import build_file_frontmatter, source_files, to_frontmatter


def _make_graph():
    """Two files. 'Engine' (core.py) is a god node, used by 'App' (app.py),
    with one AMBIGUOUS cross-file edge to 'Runner'."""
    G = nx.Graph()
    G.add_node("n1", label="Engine", file_type="code", source_file="core.py", source_location="L10", community=0)
    G.add_node("n2", label="helper", file_type="code", source_file="core.py", source_location="L40", community=0)
    G.add_node("n3", label="App", file_type="code", source_file="app.py", source_location="L5", community=1)
    G.add_node("n4", label="Runner", file_type="code", source_file="app.py", source_location="L22", community=1)
    # within-file structural-ish call
    G.add_edge("n1", "n2", relation="calls", confidence="EXTRACTED", weight=1.0)
    # App uses Engine: an incoming use of core.py (direction via _src/_tgt)
    G.add_edge("n3", "n1", relation="uses", confidence="EXTRACTED", weight=1.0, _src="n3", _tgt="n1")
    # AMBIGUOUS cross-file edge from Engine
    G.add_edge("n1", "n4", relation="references", confidence="AMBIGUOUS", weight=1.0, _src="n1", _tgt="n4")
    return G


COMMUNITIES = {0: ["n1", "n2"], 1: ["n3", "n4"]}
LABELS = {0: "Core", 1: "Application"}


def test_source_files_lists_real_files():
    G = _make_graph()
    assert source_files(G) == ["app.py", "core.py"]


def test_frontmatter_has_yaml_block():
    G = _make_graph()
    doc = build_file_frontmatter(G, "core.py", COMMUNITIES, LABELS)
    assert doc.startswith("---")
    assert 'source_file: "core.py"' in doc
    assert "generated_by: graphify" in doc


def test_frontmatter_lists_entities_and_community():
    G = _make_graph()
    doc = build_file_frontmatter(G, "core.py", COMMUNITIES, LABELS)
    assert '"Engine (L10)"' in doc
    assert 'community: "Core"' in doc


def test_example_uses_lists_incoming_caller():
    G = _make_graph()
    doc = build_file_frontmatter(G, "core.py", COMMUNITIES, LABELS)
    section = doc.split("## Example uses", 1)[1].split("## Guardrails", 1)[0]
    # App (app.py) uses Engine -> shows up as an incoming use of core.py
    assert "App" in section
    assert "--uses-->" in section
    assert "Engine" in section


def test_few_shot_references_have_anchors():
    G = _make_graph()
    doc = build_file_frontmatter(G, "core.py", COMMUNITIES, LABELS)
    section = doc.split("## Few-shot references", 1)[1].split("## Example uses", 1)[0]
    # cross-file neighbours in app.py are offered as worked examples
    assert "app.py" in section
    assert "[[" in section


def test_guardrails_flag_god_node_and_ambiguous_edge():
    G = _make_graph()
    doc = build_file_frontmatter(G, "core.py", COMMUNITIES, LABELS)
    section = doc.split("## Guardrails", 1)[1]
    assert "god node" in section
    assert "Engine" in section
    assert "AMBIGUOUS" in section


def test_empty_sections_have_fallbacks():
    # A single isolated file: no cross-file refs and no external callers.
    G = nx.Graph()
    G.add_node("a", label="Solo", file_type="code", source_file="solo.py", source_location="L1", community=0)
    G.add_node("b", label="mate", file_type="code", source_file="solo.py", source_location="L9", community=0)
    G.add_edge("a", "b", relation="calls", confidence="EXTRACTED", weight=1.0)
    doc = build_file_frontmatter(G, "solo.py", {0: ["a", "b"]})
    fewshot = doc.split("## Few-shot references", 1)[1].split("## Example uses", 1)[0]
    uses = doc.split("## Example uses", 1)[1].split("## Guardrails", 1)[0]
    assert "No cross-file references" in fewshot
    assert "No external callers" in uses


def test_to_frontmatter_writes_one_card_per_file(tmp_path):
    G = _make_graph()
    n = to_frontmatter(G, COMMUNITIES, tmp_path, community_labels=LABELS)
    assert n == 2
    assert (tmp_path / "core.py.md").exists()
    assert (tmp_path / "app.py.md").exists()


def test_to_frontmatter_card_is_valid(tmp_path):
    G = _make_graph()
    to_frontmatter(G, COMMUNITIES, tmp_path, community_labels=LABELS)
    text = (tmp_path / "core.py.md").read_text()
    assert "## Few-shot references" in text
    assert "## Example uses" in text
    assert "## Guardrails" in text
